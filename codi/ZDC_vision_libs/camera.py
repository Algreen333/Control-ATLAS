import cv2
import threading
import time
import logging

class Camera:
    def __init__(self, source, width=640, height=480, fps=30, timeout=0.5):
        self.source = source
        self.timeout = timeout

        self.cap = cv2.VideoCapture(self.source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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

    