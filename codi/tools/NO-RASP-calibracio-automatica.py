
import numpy as np
import cv2
from cv2 import aruco

import sys
sys.path.append("main")

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
all_obj_points = [] # 3d point in real world space
all_img_points = [] # 2d points in image plane.

SQUARE_LENGTH = 0.02  
MARKER_LENGTH = 0.015
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
CHARUCO_BOARD = aruco.CharucoBoard((9,7), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)

charuco_detector = aruco.CharucoDetector(CHARUCO_BOARD)


size = int(input("Width: "))
size = (size, int(input("Height: ")))

cap = cv2.VideoCapture(0)


print("FENT FOTOS ... prem Ctrl+C per aturar el programa i obtenir la calibracio")
enabled = True

while cv2.waitKey(30) != ord('q'):
    ret, img = cap.read()

    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)


        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 6:
            obj_points, img_points = CHARUCO_BOARD.matchImagePoints(charuco_corners, charuco_ids)
            if obj_points is not None and img_points is not None and len(img_points) > 3:
                all_obj_points.append(obj_points)
                all_img_points.append(img_points)
                
                img = aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

    cv2.imshow("img", img)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, gray.shape[::-1], None, None)
cv2.destroyAllWindows()
cv2.waitKey(1)

print(mtx)

print()

print(dist)