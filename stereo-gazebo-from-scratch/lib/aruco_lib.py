from cv2 import aruco
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


VERBOSE = False


def get_euler_angles(rvec):
    """
    Transforms a rotation vector into the equivalent euler angles.
    """
    R, _ = cv2.Rodrigues(rvec)
    
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    # Convert to degrees
    return np.degrees(x), np.degrees(y), np.degrees(z)

def create_homogeneous_matrix(rvec, tvec):
    """
    Converts OpenCV rvec and tvec into a 4x4 homogeneous transformation matrix.
    """
    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create 4x4 matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return T

def fuse_stereo_aruco_poses(rvec_1, tvec_1, T_C_1, rvec_2, tvec_2, T_C_2):
    """
    Transforms left and right camera ArUco detections to the center frame 
    and averages them using translation mean and quaternion SLERP.
    
    Args:
        rvec_L, tvec_L: Detection from Left Camera
        T_C_L: 4x4 matrix of Left Camera relative to Center
        rvec_R, tvec_R: Detection from Right Camera
        T_C_R: 4x4 matrix of Right Camera relative to Center
        
    Returns:
        rvec_avg, tvec_avg: The fused pose in the Center coordinate frame
    """
    
    # 1. Convert local camera detections to 4x4 matrices
    T_L_M = create_homogeneous_matrix(rvec_1, tvec_1)
    T_R_M = create_homogeneous_matrix(rvec_2, tvec_2)
    
    # 2. Transform both detections to the Center frame
    # T_Center_Marker = T_Center_Camera * T_Camera_Marker
    T_1 = T_C_1 @ T_L_M 
    T_2 = T_C_2 @ T_R_M
    
    # 3. Average Translation (Position)
    t1 = T_1[0:3, 3]
    t2 = T_2[0:3, 3]
    t_avg = (t1 + t2) / 2.0
    
    # 4. Average Rotation using SLERP (Quaternions)
    # Extract 3x3 rotation matrices
    R1_mat = T_1[0:3, 0:3]
    R2_mat = T_2[0:3, 0:3]
    
    # Convert to SciPy Rotation objects
    rot1 = Rotation.from_matrix(R1_mat)
    rot2 = Rotation.from_matrix(R2_mat)
    
    # Set up SLERP between the two rotations at "time" 0 and 1
    key_rots = Rotation.concatenate([rot1, rot2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    
    # Evaluate exactly halfway between them (t = 0.5)
    rot_avg = slerp([0.5])[0]
    
    # 5. Convert fused rotation back to OpenCV rvec format
    R_avg_mat = rot_avg.as_matrix()
    rvec_avg, _ = cv2.Rodrigues(R_avg_mat)
    
    # Ensure tvec shape matches OpenCV standards (3x1)
    tvec_avg = t_avg.reshape((3, 1))
    
    return rvec_avg, tvec_avg

def get_gazebo_camera_matrix(width, height, horizontal_fov_rad, vertical_fov_rad):
    fx = (width / 2.0) / np.tan(horizontal_fov_rad / 2.0)
    fy = (height / 2.0) / np.tan(vertical_fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    
    mtx = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    
    return mtx

class ArucoDetector():
    def __init__(self, mtx, dist, dict=aruco.DICT_ARUCO_ORIGINAL):

        self.dictionary = aruco.getPredefinedDictionary(dict)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

        self.mtx = np.array(mtx, dtype=np.float32).reshape(3, 3)
        self.dist = np.array(dist, dtype=np.float32)

    def detectMarkers(self, frame):
        """
        Detection of aruco markers in a frame.

        :param MatLike frame: Frame to process
        :return corners ([np.array..]): corners of the detected arucos
        :return ids ([int..]): ids of the detected arucos
        :return rejected: rejected img points
        """

        corners, ids, rejected = self.detector.detectMarkers(frame)
        return corners, ids, rejected
    
    def estimate_pose(self, current_marker_corners, marker_size):
        """
        Estimates the position and rotation of an aruco detection relative to the image of a single marker.
        
        :param list corners: Corner points from detector.detectMarkers()
        :param float marker_size: Physical size of the marker in meters (e.g., 0.2)
        :param np.array mtx: Camera Matrix (Intrinsics)
        :param np.array dist: Distortion Coefficients
            
        :returns rvec (np.array): Rotation vector
        :returns tvec (np.array): Translation vector
        """

        marker_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)

        image_points = current_marker_corners.reshape((4, 2))
        
        if VERBOSE: print("solving...")

        _, rvec, tvec = cv2.solvePnP(
            marker_points,
            image_points,
            self.mtx,
            self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return rvec, tvec
    
    def full_prediction(self, frame, do_draw:bool=True):
        """
        Full detection and processing of aruco markers.

        :param MatLike frame: frame to be processed
        :param bool (optional) do_draw: Activate or deactivate function to draw detections onto the frame
        :return frame (MatLike): Original frame or frame with detections
        :return position ((float, float, float)): Position of the detected aruco from the camera
        :return rotation ((float, float, float)): Rotation of the detected aruco with respect to the camera
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detectMarkers(gray)

            if ids is not None:
                if VERBOSE: print(f"Found {len(ids)} markers.")
                
                for i in range(len(ids)):

                    if VERBOSE: print("--------------------------------------------",
                                    "\nShape:", corners[i].shape)

                    rvec, tvec = self.estimate_pose(corners[i], 0.2)

                    distance = np.linalg.norm(tvec)
                    if VERBOSE: print(f"Distance: {distance:.3f} meters")

                    x_offset = tvec[0][0]
                    y_offset = tvec[1][0]
                    z_depth  = tvec[2][0]

                    if VERBOSE: print(f"Position: x={x_offset:.2f}, y={y_offset:.2f}, z={z_depth:.2f}")

                    pitch, yaw, roll = get_euler_angles(rvec)
                    if VERBOSE: print(f"Orientation: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}", 
                                    "\n--------------------------------------------")

                    if do_draw: aruco.drawDetectedMarkers(frame, corners, ids)

                    return (frame, (x_offset, y_offset, z_depth), (pitch, yaw, roll))
        except Exception as e: print("'ArucoDetector:full_detection' error:", e)
        
        return None
