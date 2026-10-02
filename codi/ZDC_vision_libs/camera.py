import cv2
import threading
import time
import logging

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except (ImportError, RuntimeError):
    HAS_PICAMERA = False

class Camera:
    def __init__(self, source, resolution=(640, 480), fps=30, timeout=0.5, AeEnable=False, ExposureTime=50000, AnalogueGain=2.0, Brightness=0.1, format="BGR888"):
        self.source = source
        self.timeout = timeout
        self.resolution = resolution
        self.fps = fps
        self.has_picamera = bool(HAS_PICAMERA)
        self.format = format
        self.lock = threading.Lock()

        self.frame = None
        self.isHealthy = False
        self.isRunning = True
        self.last_frame_time = time.time()

        if self.has_picamera:
            try:
                self.cap = Picamera2(self.source)
                config = self.cap.create_preview_configuration(main={"size": self.resolution, "format": self.format})
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
                # Primer frame de prueba con Picamera2
                test_frame = self.cap.capture_array()
                if test_frame is not None:
                    self.frame = test_frame
                    self.isHealthy = True
            except Exception as e:
                logging.error(f"[HAL] CRITICAL: Failed to initialize Picamera2: {e}")
                raise RuntimeError(f"Picamera2 hardware unreachable on source {self.source}")
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.cap.isOpened():
                self.isRunning = False
                logging.error(f"[HAL] CRITICAL: Failed to open camera source {self.source}")
                raise RuntimeError(f"Camera Hardware {self.source} unreachable.")

            ret, test_frame = self.cap.read()
            if ret:
                self.frame = test_frame
                self.isHealthy = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.isRunning:
            ret = False
            frame = None

            try:
                if self.has_picamera:
                    frame = self.cap.capture_array()
                    ret = frame is not None
                else:
                    ret, frame = self.cap.read()
            except Exception as e:
                logging.error(f"[HAL] Error reading frame from source {self.source}: {e}")
                ret = False

            with self.lock:
                if ret:
                    self.frame = frame
                    self.last_frame_time = time.time()
                    self.isHealthy = True
                else:
                    self.isHealthy = False

            if not ret:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if time.time() - self.last_frame_time > self.timeout:
                if self.isHealthy:
                    logging.warning(f"[HAL] WARNING: Camera {self.source} timed out.")
                self.isHealthy = False

            return self.isHealthy, self.frame

    def stop(self):
        self.isRunning = False
        if self.thread.is_alive():
            self.thread.join()

        if self.has_picamera:
            self.cap.stop()
            self.cap.close()
        else:
            self.cap.release()

        logging.info(f"[HAL] Camera {self.source} released.")