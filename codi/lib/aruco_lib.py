from cv2 import aruco
import cv2
import numpy as np

VERBOSE = True


def average_pos(positions):
    xs = 0
    ys = 0
    zs = 0

    for p in positions:
        x, y, z = p
        xs+=x
        ys+=y
        zs+=z
    
    xs /= len(positions)
    ys /= len(positions)
    zs /= len(positions)

    return xs, ys, zs

def get_euler_angles(rvec):
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

class ArucoDetector():
    def __init__(self, mtx, dist, dict=aruco.DICT_ARUCO_ORIGINAL):

        self.dictionary = aruco.getPredefinedDictionary(dict)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

        self.mtx = np.array(mtx, dtype=np.float32).reshape(3, 3)
        self.dist = np.array(dist, dtype=np.float32)

    def detectMarkers(self, frame):
        corners, ids, rejected = self.detector.detectMarkers(frame)
        return corners, ids, rejected
    
    def estimate_pose(self, current_marker_corners, marker_size):
        """
        Estimates the pose of a single marker.
        
        Args:
            corners (list): Corner points from detector.detectMarkers()
            marker_size (float): Physical size of the marker in meters (e.g., 0.2)
            mtx (np.array): Camera Matrix (Intrinsics)
            dist (np.array): Distortion Coefficients
            
        Returns:
            rvec, tvec: Rotation and Translation vectors
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
    
    def full_prediction(self, frame):
        """
        Fa el processat sencer de deteccions aruco. 
        Retorna (frame, (offset_x, offset_y, offset_z), (pitch, yaw, roll)) del primer aruco detectat.
        Si no es fa cap detecció retorna None.
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


                    aruco.drawDetectedMarkers(frame, corners, ids)

                    return (frame, (x_offset, y_offset, z_depth), (pitch, yaw, roll))
        except Exception as e: print("'ArucoDetector:full_detection' error:", e)
        
        return None
