import cv2
import threading
import time
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger("CameraHAL")

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except (ImportError, RuntimeError):
    HAS_PICAMERA = False


class Camera:
    def __init__(
        self,
        source: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        timeout: float = 0.5,
        AeEnable: bool = False,
        ExposureTime: int = 50000,
        AnalogueGain: float = 2.0,
        Brightness: float = 0.1,
        format: str = "BGR888"
    ):
        self.source = source
        self.timeout = timeout
        self.resolution = resolution
        self.fps = fps
        self.has_picamera = bool(HAS_PICAMERA)
        self.format = format

        self._lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.isHealthy: bool = False
        self.isRunning: bool = True
        self.last_frame_time: float = 0.0

        if self.has_picamera:
            try:
                self.cap = Picamera2(self.source)
                config = self.cap.create_preview_configuration(
                    main={"size": self.resolution, "format": self.format}
                )
                self.cap.configure(config)
                frame_duration = int(1e6 / self.fps)
                self.cap.set_controls({
                    "AeEnable": AeEnable,
                    "FrameDurationLimits": (frame_duration, frame_duration),
                    "AnalogueGain": float(AnalogueGain),
                    "Brightness": float(Brightness),
                    "ExposureTime": int(ExposureTime)
                })
                self.cap.start()
                test_frame = self.cap.capture_array()
                if test_frame is not None:
                    self.frame = test_frame
                    self.isHealthy = True
                    self.last_frame_time = time.time()
            except Exception as e:
                logger.error(f"[HAL] Critical: Picamera2 initialization failed on index {self.source}: {e}")
                raise RuntimeError(f"Picamera2 init failed on {self.source}")
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.cap.isOpened():
                self.isRunning = False
                logger.error(f"[HAL] Critical: Failed to open OpenCV video capture source {self.source}")
                raise RuntimeError(f"Camera source {self.source} unreachable.")

            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                self.frame = test_frame
                self.isHealthy = True
                self.last_frame_time = time.time()

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        while self.isRunning:
            ret = False
            captured_frame = None

            try:
                if self.has_picamera:
                    captured_frame = self.cap.capture_array()
                    ret = captured_frame is not None
                else:
                    ret, captured_frame = self.cap.read()
            except Exception as e:
                logger.error(f"[HAL] Error reading frame from source {self.source}: {e}")
                ret = False

            now = time.time()
            with self._lock:
                if ret and captured_frame is not None:
                    self.frame = captured_frame
                    self.last_frame_time = now
                    self.isHealthy = True
                else:
                    self.isHealthy = False

            if not ret:
                time.sleep(0.01)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Non-blocking snapshot retrieval."""
        with self._lock:
            healthy = self.isHealthy and ((time.time() - self.last_frame_time) <= self.timeout)
            frame = self.frame.copy() if (self.frame is not None and healthy) else None
            return healthy, frame

    def stop(self):
        self.isRunning = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.has_picamera:
            try:
                self.cap.stop()
                self.cap.close()
            except Exception:
                pass
        else:
            try:
                self.cap.release()
            except Exception:
                pass

        logger.info(f"[HAL] Camera source {self.source} safely shut down.")