from flask import Flask, Response
from picamera2 import Picamera2
from libcamera import controls
import cv2
import time
import numpy as np
import threading

app = Flask(__name__)

# Global variables for thread-safe frame sharing
latest_frame = None
frame_lock = threading.Lock()

def setup_video_sync_node(cam_index, sync_mode):
    """Initializes a camera for continuous video capture with sync controls."""
    picam = Picamera2(cam_index)
    config = picam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    
    # Crucial for minimizing latency: prevents libcamera from queueing old frames [cite: 66, 67]
    config["main"]["queue"] = False
    config["main"]["buffer_count"] = 6
    
    picam.configure(config)
    
    if sync_mode == "server":
        picam.set_controls({"SyncMode": controls.rpi.SyncModeEnum.Server})
        print(f"Hardware node {cam_index} initialized as IPA SYNC SERVER.")
    elif sync_mode == "client":
        picam.set_controls({"SyncMode": controls.rpi.SyncModeEnum.Client})
        print(f"Hardware node {cam_index} initialized as IPA SYNC CLIENT.")
        
    picam.start()
    return picam

def sensor_acquisition_loop(cam_server, cam_client):
    """
    HIGH-PRIORITY THREAD:
    Continuously pulls requests from the hardware, calculates drift, 
    and updates the global frame buffer. Drops older frames inherently.
    """
    global latest_frame
    
    print("Acquisition thread running. Allowing IPA feedback loop to converge...")
    time.sleep(2)
    
    while True:
        try:
            req_server = cam_server.capture_request()
            req_client = cam_client.capture_request()
            
            ts_server = req_server.get_metadata().get('SensorTimestamp', 0)
            ts_client = req_client.get_metadata().get('SensorTimestamp', 0)
            
            drift_ns = abs(ts_server - ts_client)
            drift_ms = drift_ns / 1_000_000.0
            
            img_server_rgb = req_server.make_array("main")
            img_client_rgb = req_client.make_array("main")
            
            img_server = img_server_rgb[:, :, ::-1]
            img_client = img_client_rgb[:, :, ::-1]

            # Immediately release requests back to the system pool to prevent stalling
            req_server.release()
            req_client.release()


            status_color = (0, 255, 0) if drift_ms < 1.0 else (0, 0, 255)
            cv2.putText(img_server, f"SERVER | Drift: {drift_ms:.3f} ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(img_client, f"CLIENT | Drift: {drift_ms:.3f} ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            combined_frame = cv2.hconcat([img_server, img_client])

            # Safely update the global frame reference
            with frame_lock:
                latest_frame = combined_frame
                
        except Exception as e:
            print(f"Acquisition fault: {e}")
            break

def generate_telemetry_stream():
    """
    LOWER-PRIORITY GENERATOR:
    Pulls the most recent frame from the acquisition thread and streams it to the web.
    """
    global latest_frame
    
    while True:
        with frame_lock:
            if latest_frame is None:
                frame_to_encode = None
            else:
                frame_to_encode = latest_frame.copy()
        
        if frame_to_encode is None:
            time.sleep(0.1)
            continue
            
        # Compress the frame for web transmission
        ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        # Optional: Throttle the web stream to ~15-20 FPS to save CPU for drone SLAM
        time.sleep(0.05)

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>UAV Stereo Vision Telemetry</title>
            <style>
                body { background-color: #121212; color: #ffffff; text-align: center; font-family: sans-serif; }
                img { max-width: 100%; height: auto; border: 2px solid #333; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h2>Synchronized Optical Feed</h2>
            <p>Real-time telemetry view. Dropped frames prioritized over latency.</p>
            <img src="/video_feed" />
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_telemetry_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    cam_server = None
    cam_client = None
    try:
        print("Initializing optical sensors...")
        cam_server = setup_video_sync_node(0, "server")
        cam_client = setup_video_sync_node(1, "client")
        
        # Start the high-speed background acquisition thread
        acquisition_thread = threading.Thread(
            target=sensor_acquisition_loop, 
            args=(cam_server, cam_client),
            daemon=True # Ensures the thread dies when the main program exits
        )
        acquisition_thread.start()
        
        print("Starting GCS telemetry server on port 5000...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
        
    finally:
        print("Shutting down MIPI receivers safely...")
        if cam_server:
            cam_server.stop()
        if cam_client:
            cam_client.stop()