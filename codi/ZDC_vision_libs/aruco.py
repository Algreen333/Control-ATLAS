import cv2
import numpy as np
import logging

class Aruco:
    def __init__(self, marker_size_m, camera_matrix, dist_coeffs, dict_id=cv2.aruco.DICT_4):
        self.marker_size_m = marker_size_m 
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        half = self.marker_size_m/2.0
        self.obj_points = np.array([
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(corners) == 0:
            return False, 0.0, 0.0, 0.0

        target_corners = corners[0]

        target_corners = cv2.cornerSubPix(
            gray,
            target_corners,
            winSize=(5,5),
            zeroZone=(-1,-1),
            criteria=self.subpix_criteria
        )

        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            target_corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags = cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return False, 0.0, 0.0, 0.0

        x_m = float(tvec[0][0])
        y_m = float(tvec[1][0])
        z_m = float(tvec[2][0])

        return True, x_m, y_m, z_m