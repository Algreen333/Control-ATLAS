import time
import cv2
import numpy as np
from lib.camera_lib_light import FlaskPreviewServer, StereoCapture
from lib.pid_controller import PIDController

# Mock a StereoCapture class so isinstance(capture, StereoCapture) returns True
class MockStereoCapture(StereoCapture):
    def __init__(self):
        self.cam1_cfg = {"ExposureTime": 700, "AeEnable": False}
        self.cam2_cfg = {"ExposureTime": 700, "AeEnable": False}
        self.web_cam = cv2.VideoCapture(0) # Uses your PC webcam if available

    def set_exposure(self, cam_idx: int, exposure_time: int):
        print(f"[MOCK HW] Setting Cam {cam_idx} exposure to {exposure_time} us")
        if cam_idx == 1:
            self.cam1_cfg["ExposureTime"] = exposure_time
        else:
            self.cam2_cfg["ExposureTime"] = exposure_time

    def get_test_frame(self):
        ret, frame = self.web_cam.read()
        if not ret or frame is None:
            # Generate a dummy synthetic frame if no webcam is plugged in
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Webcam - Synthetic Feed", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Simulate brightness/exposure change visually for testing
        exp1 = self.cam1_cfg["ExposureTime"]
        exp2 = self.cam2_cfg["ExposureTime"]
        f1 = cv2.convertScaleAbs(frame, alpha=max(0.2, min(3.0, exp1 / 700.0)), beta=0)
        f2 = cv2.convertScaleAbs(frame, alpha=max(0.2, min(3.0, exp2 / 700.0)), beta=0)
        
        cv2.putText(f1, f"Cam 1 ({exp1} us)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(f2, f"Cam 2 ({exp2} us)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return cv2.hconcat([f1, f2])

if __name__ == "__main__":
    mock_cam = MockStereoCapture()
    pid_x = PIDController(kp=0.15, ki=0.05, kd=0.06)
    pid_y = PIDController(kp=0.15, ki=0.05, kd=0.06)

    preview = FlaskPreviewServer(
        host="127.0.0.1",
        port=5000,
        max_fps=15,
        jpeg_quality=70,
        tuning_enabled=True,
        pid_controllers=[pid_x, pid_y],
        cam_tuning_enabled=True,
        capture=mock_cam,
        config_path="mission_config.json"
    )
    preview.start()
    print("UI Server running at: http://127.0.0.1:5000")

    try:
        while True:
            frame = mock_cam.get_test_frame()
            preview.update_frame(frame, drift_ms=1.23)
            time.sleep(1 / 15)
    except KeyboardInterrupt:
        preview.stop()