import sys
import time
import logging
import cv2
import numpy as np
from flask import Flask, Response
import aruco

from visionOrchestrator import VisionOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

CAM1_MATRIX = [[
            1291.167154,
            0.0,
            806.142608
        ],
        [
            0.0,
            1291.77973,
            587.53442
        ],
        [
            0.0,
            0.0,
            1.0
        ]]
CAM1_DIST = [
            0.17131639,
            -0.31106833,
            -0.00193605,
            -0.00056888,
            0.06746523
            ]

CAM0_MATRIX = np.array([[500.0, 0.0, 320.0],
                        [0.0, 500.0, 240.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
CAM0_DIST = np.zeros((5, 1), dtype=np.float32)

CAM0_ID = 0
CAM1_ID = 1
MARKER_SIZE = 0.10

app = Flask(__name__)
orchestrator = None

def generate_frames():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    while True:
        payload = orchestrator.get_vision_data()

        h0, f0 = orchestrator.cam0.get_frame()
        h1, f1 = orchestrator.cam1.get_frame()

        img0 = f0.copy() if (h0 and f0 is not None) else blank.copy()
        img1 = f1.copy() if (h1 and f1 is not None) else blank.copy()

        cv2.putText(img0, f"CAM 0 ({'OK' if h0 else 'TIMEOUT'})", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if h0 else (0, 0, 255), 2)
        cv2.putText(img1, f"CAM 1 ({'OK' if h1 else 'TIMEOUT'})", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if h1 else (0, 0, 255), 2)

        combined = np.hstack((img0, img1))

        status_color = (0, 255, 0) if payload.target_acquired else (0, 0, 255)
        status_txt = "TARGET ACQUIRED" if payload.target_acquired else "SEARCHING"
        info_txt = (f"[{status_txt}] Cams: {payload.active_cameras}/2 | "
                    f"X: {payload.x_offset_m:+.2f}m  Y: {payload.y_offset_m:+.2f}m  Z: {payload.z_distance_m:+.2f}m")

        cv2.rectangle(combined, (0, 0), (combined.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(combined, info_txt, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        ret, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logging.info("[MAIN] Iniciando VisionOrchestrator...")
    orchestrator = VisionOrchestrator(
        cam0_idx=CAM0_ID,
        cam1_idx=CAM1_ID,
        marker_size_m=MARKER_SIZE,
        cam0_matrix=CAM0_MATRIX,
        cam0_dist=CAM0_DIST,
        cam1_matrix=CAM1_MATRIX,
        cam1_dist=CAM1_DIST,
        cam0_offset_m=(0.0, 0.0, 0.0),
        cam1_offset_m=(0.0, 0.0, 0.0)
    )

    logging.info("[WEB] Preview disponible en http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        orchestrator.stop()