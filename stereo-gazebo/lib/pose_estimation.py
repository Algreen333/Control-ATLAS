import cv2
import numpy as np
from cv2 import aruco

# Codi desenvolupat per a la estimació de la posició dels ArUco


def estimate_pose(current_marker_corners, marker_size, mtx, dist):
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
    mtx = np.array(mtx, dtype=np.float32).reshape(3, 3)

    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float32)
    else:
        dist = np.array(dist, dtype=np.float32)

    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    image_points = current_marker_corners.reshape((4, 2))
    
    _, rvec, tvec = cv2.solvePnP(
        marker_points,
        image_points,
        mtx,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    return rvec, tvec

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

if __name__ == "__main__":
    mtx = [[1.51886749e+03, 0.00000000e+00, 9.58667845e+02],
           [0.00000000e+00, 1.51891963e+03, 5.46500289e+02],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    dst = [[-1.42885973e-01, 1.56710837e+00, 3.70984276e-04, -1.01880722e-03, -5.00183593e+00]]

    cap = cv2.VideoCapture(0)


    dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    while cv2.waitKey(30) != ord('q'):
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # corners: A list of arrays. Each item corresponds to one detected marker.
            # ids: A list of IDs corresponding to the corners.
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is not None:
                print(f"Found {len(ids)} markers.")
                
                for i in range(len(ids)):
                    try:
                        print(corners[i].shape)

                        rvec, tvec = estimate_pose(corners[i], 0.2, mtx, dst)

                        distance = np.linalg.norm(tvec)
                        print(f"Distance: {distance:.3f} meters")

                        x_offset = tvec[0][0]
                        y_offset = tvec[1][0]
                        z_depth  = tvec[2][0]

                        print(f"Position: x={x_offset:.2f}, y={y_offset:.2f}, z={z_depth:.2f}")

                        pitch, yaw, roll = get_euler_angles(rvec)
                        print(f"Orientation: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}")

                    except Exception as e: print(e)

                    aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.imshow("frame", frame)


