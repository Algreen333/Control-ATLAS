import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger("ArucoDetector")


class Aruco:
    def __init__(
        self,
        marker_size_m: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        target_id: int = 26,
        dict_id: int = cv2.aruco.DICT_4X4_50
    ):
        self.marker_size_m = float(marker_size_m)
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)
        self.target_id = int(target_id)

        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        half = self.marker_size_m / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0]
        ], dtype=np.float32)

        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.001
        )

    def process_frame(
        self,
        frame: np.ndarray,
        target_id: Optional[int] = None,
        draw: bool = False
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if frame is None:
            return False, None, None, frame

        active_id = self.target_id if target_id is None else target_id
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(corners) == 0:
            return False, None, None, frame

        flat_ids = np.ravel(ids)
        target_idx = None
        for i, detected_id in enumerate(flat_ids):
            if detected_id == active_id:
                target_idx = i
                break

        if target_idx is None:
            return False, None, None, frame

        target_corners = corners[target_idx].copy()
        try:
            target_corners = cv2.cornerSubPix(
                gray,
                target_corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=self.subpix_criteria
            )
        except cv2.error:
            pass  # Fall back to raw contour corners if subpixel fails

        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            target_corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return False, None, None, frame

        if draw:
            cv2.aruco.drawDetectedMarkers(frame, [target_corners], np.array([[active_id]]))
            cv2.drawFrameAxes(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.marker_size_m * 0.5
            )

        return True, rvec, tvec, frame