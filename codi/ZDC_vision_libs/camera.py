import cv2
import threading
import time
import logging
import json

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

        if self.has_picamera:
            try:
                self.cap = Picamera2(self.source)
                config = self.cap.create_preview_configuration(main={"size": self.resolution, "format": self.format})
                self.cap.configure(config)
                frame_duration = int(1e6/self.fps)
                self.cap.set_controls({
                    "AeEnable": AeEnable,
                    "FrameDurationLimits": (frame_duration,frame_duration),
                    "AnalogueGain": float(AnalogueGain),
                    "Brightness": float(Brightness),
                    "ExposureTime": int(ExposureTime)
                })

                self.cap.start()
            except Exception as e:
                logging.error(f"[HAL] CRITICAL: Failed to initialize Picamera2: {e}")
                raise RuntimeError(f"Picamera2 hardware unreachable on source {self.source}")
            
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.isHealthy = False
            self.isRunning = False
            logging.error(f"[HAL] CRITICAL: Failed to open camera source {self.source}")
            raise RuntimeError(f"Camera Hardware {self.source} unreachable.")

        self.frame = None
        self.isHealthy = True
        self.isRunning = True
        self.last_frame_time = time.time() 

        self.lock = threading.Lock()

        #Comprova amb el primer frame per veure si la cam va
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
        else:
            self.isHealthy = False

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.isRunning:
            ret, frame = self.cap.read()

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
            #Comprovo si la cap esta ben sincronitzada 
            if time.time() - self.last_frame_time > self.timeout:
                if self.isHealthy:
                    logging.warning(f"[HAL] WARNING: Camera {self.source} timed out.")
                self.isHealthy = False

            return self.isHealthy, self.frame

    def stop(self):
        self.isRunning = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
        logging.info(f"[HAL] Camera {self.source} released.")

    