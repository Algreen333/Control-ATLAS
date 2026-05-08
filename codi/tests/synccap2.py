from flask import Flask, Response
from picamera2 import Picamera2
from libcamera import controls
import cv2
import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation, Slerp


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (main)
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lib.camera_lib import CalibrationConfig
from lib.aruco_lib import ArucoDetector

calconfWide = CalibrationConfig.from_path("../configs/1536x864-v3.conf")
calconfNarr = CalibrationConfig.from_path("../configs/1640x1232-v2.conf")

detWide = ArucoDetector(calconfWide.mtx, calconfWide.dist)
detNarr = ArucoDetector(calconfNarr.mtx, calconfNarr.dist)


T_C_WIDE = np.eye(4)
T_C_WIDE[0, 3] = 65
T_C_NARR = np.eye(4)
T_C_NARR[0, 3] = -85


SERVER_ID = 0
CLIENT_ID = 1

DRAW_ARUCOS = True

app = Flask(__name__)

# Global variables for thread-safe frame sharing
latest_frame_wide = None
latest_frame_narr = None
latest_drift_ms = None

frame_lock = threading.Lock()

def create_homogeneous_matrix(rvec, tvec):
    """
    Converts OpenCV rvec and tvec into a 4x4 homogeneous transformation matrix.
    """
    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create 4x4 matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return T

def fuse_stereo_aruco_poses(rvec_1, tvec_1, T_C_1, rvec_2, tvec_2, T_C_2):
    """
    Transforms left and right camera ArUco detections to the center frame 
    and averages them using translation mean and quaternion SLERP.
    
    Args:
        rvec_L, tvec_L: Detection from Left Camera
        T_C_L: 4x4 matrix of Left Camera relative to Center
        rvec_R, tvec_R: Detection from Right Camera
        T_C_R: 4x4 matrix of Right Camera relative to Center
        
    Returns:
        rvec_avg, tvec_avg: The fused pose in the Center coordinate frame
    """
    
    # 1. Convert local camera detections to 4x4 matrices
    T_L_M = create_homogeneous_matrix(rvec_1, tvec_1)
    T_R_M = create_homogeneous_matrix(rvec_2, tvec_2)
    
    # 2. Transform both detections to the Center frame
    # T_Center_Marker = T_Center_Camera * T_Camera_Marker
    T_1 = T_C_1 @ T_L_M 
    T_2 = T_C_2 @ T_R_M
    
    # 3. Average Translation (Position)
    t1 = T_1[0:3, 3]
    t2 = T_2[0:3, 3]
    t_avg = (t1 + t2) / 2.0
    
    # 4. Average Rotation using SLERP (Quaternions)
    # Extract 3x3 rotation matrices
    R1_mat = T_1[0:3, 0:3]
    R2_mat = T_2[0:3, 0:3]
    
    # Convert to SciPy Rotation objects
    rot1 = Rotation.from_matrix(R1_mat)
    rot2 = Rotation.from_matrix(R2_mat)
    
    # Set up SLERP between the two rotations at "time" 0 and 1
    key_rots = Rotation.concatenate([rot1, rot2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    
    # Evaluate exactly halfway between them (t = 0.5)
    rot_avg = slerp([0.5])[0]
    
    # 5. Convert fused rotation back to OpenCV rvec format
    R_avg_mat = rot_avg.as_matrix()
    rvec_avg, _ = cv2.Rodrigues(R_avg_mat)
    
    # Ensure tvec shape matches OpenCV standards (3x1)
    tvec_avg = t_avg.reshape((3, 1))
    
    return rvec_avg, tvec_avg

def joint_display(img_server, img_client, drift_ms):
    # --- DYNAMIC DIMENSION MATCHING (ZERO-PADDING) ---
    h_server, w_server = img_server.shape[:2]
    h_client, w_client = img_client.shape[:2]
    
    # If heights mismatch, pad the bottom of the shorter image with zeros (black)
    if h_server > h_client:
        padding = h_server - h_client
        img_client = cv2.copyMakeBorder(img_client, 0, padding, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif h_client > h_server:
        padding = h_client - h_server
        img_server = cv2.copyMakeBorder(img_server, 0, padding, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])

    status_color = (0, 255, 0) if drift_ms < 1.0 else (0, 0, 255)
    cv2.putText(img_server, f"SERVER | Drift: {drift_ms:.3f} ms", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(img_client, f"CLIENT | Drift: {drift_ms:.3f} ms", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    combined_frame = cv2.hconcat([img_server, img_client])
    resized = cv2.resize(combined_frame, (0,0), fx = 0.5, fy = 0.5)

    # Safely update the global frame reference
    return resized

def setup_video_sync_node(cam_index, sync_mode, size=(640, 480), format="RGB888", framerate=30.0):
    """Initializes a camera for continuous video capture with sync controls."""
    picam = Picamera2(cam_index)
    config = picam.create_video_configuration(main={"size": size, "format": format})
    
    # Crucial for minimizing latency: prevents libcamera from queueing old frames [cite: 66, 67]
    config["main"]["queue"] = False
    config["main"]["buffer_count"] = 6
    config["main"]["framerate"] = framerate
    
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
    global latest_frame_wide
    global latest_frame_narr
    global latest_drift_ms
    
    print("Acquisition thread running. Allowing IPA feedback loop to converge...")
    time.sleep(2)

    # --- TEMPORAL QUEUE ALIGNMENT BLOCK ---
    print("Initiating temporal queue alignment to break harmonic locks...")
    req_server = cam_server.capture_request()
    req_client = cam_client.capture_request()

    while True:
        ts_server = req_server.get_metadata().get('SensorTimestamp', 0)
        ts_client = req_client.get_metadata().get('SensorTimestamp', 0)
        
        drift_ns = ts_server - ts_client
        drift_ms = abs(drift_ns) / 1_000_000.0
        
        # If drift is less than 15ms (half a frame), we are on the correct frame boundary
        if drift_ms < 15.0:
            print(f"Queues perfectly aligned. Startup drift: {drift_ms:.3f} ms")
            req_server.release()
            req_client.release()
            break
            
        print(f"1-Frame offset detected ({drift_ms:.3f} ms). Flushing older frame...")
        if drift_ns < 0:
            # Server frame is older, drop it and pull a new one to catch up
            req_server.release()
            req_server = cam_server.capture_request()
        else:
            # Client frame is older, drop it and pull a new one to catch up
            req_client.release()
            req_client = cam_client.capture_request()
    
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
            
            img_server = cv2.cvtColor(img_server_rgb, cv2.COLOR_BGR2RGB)
            img_client = cv2.cvtColor(img_client_rgb, cv2.COLOR_BGR2RGB)

            # Immediately release requests back to the system pool to prevent stalling
            req_server.release()
            req_client.release()

            with frame_lock:
                latest_drift_ms = drift_ms
                if SERVER_ID == 0: 
                    latest_frame_wide = img_server
                    latest_frame_narr = img_client
                else:
                    latest_frame_wide = img_client
                    latest_frame_narr = img_server
                
        except Exception as e:
            print(f"Acquisition fault: {e}")
            break

def generate_telemetry_stream():
    """
    LOWER-PRIORITY GENERATOR:
    Pulls the most recent frame from the acquisition thread and streams it to the web.
    """
    global latest_frame_wide
    global latest_frame_narr
    global latest_drift_ms
    
    while True:
        with frame_lock:
            if latest_frame_wide is None or latest_frame_narr is None:
                frame_narr = None
                frame_wide = None
                drift_ms = None
            else:
                frame_wide = latest_frame_wide.copy()
                frame_narr = latest_frame_narr.copy()
                drift_ms = latest_drift_ms
        
        if frame_wide is None or frame_narr is None:
            time.sleep(0.1)
            continue

        if SERVER_ID==0: frame_to_encode = joint_display(frame_wide, frame_narr, drift_ms)    
        else: frame_to_encode = joint_display(frame_narr, frame_wide, drift_ms)    
        
        # Compress the frame for web transmission
        ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        # Optional: Throttle the web stream to ~15-20 FPS to save CPU for drone SLAM
        time.sleep(0.05)

def generate_aruco_stream():
    global latest_frame_wide
    global latest_frame_narr
    global latest_drift_ms
    
    while True:
        with frame_lock:
            if latest_frame_wide is None or latest_frame_narr is None:
                frame_narr = None
                frame_wide = None
                drift_ms = None
            else:
                frame_wide = latest_frame_wide.copy()
                frame_narr = latest_frame_narr.copy()
                drift_ms = latest_drift_ms
        
        if frame_wide is None or frame_narr is None:
            time.sleep(0.1)
            continue

        gray_wide = cv2.cvtColor(frame_wide, cv2.COLOR_BGR2GRAY)
        corners_wide, ids_wide, rejected_wide = detWide.detectMarkers(gray_wide)
        
        gray_narr = cv2.cvtColor(frame_narr, cv2.COLOR_BGR2GRAY)
        corners_narr, ids_narr, rejected_narr = detNarr.detectMarkers(gray_narr)

        if ids_wide is not None and ids_narr is not None:
            rvec_wide, tvec_wide = detWide.estimate_pose(corners_wide[0], 180)
            rvec_narr, tvec_narr = detNarr.estimate_pose(corners_narr[0], 180)
            rvec, tvec = fuse_stereo_aruco_poses(rvec_wide, tvec_wide, T_C_WIDE, rvec_narr, tvec_narr, T_C_NARR)
            print("----------------RVEC----------------")
            print(rvec)
            print("----------------TVEC----------------")
            print(tvec)

        if SERVER_ID==0: frame_to_encode = joint_display(frame_wide, frame_narr, drift_ms)
        else: frame_to_encode = joint_display(frame_narr, frame_wide, drift_ms)

        # Compress the frame for web transmission
        ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
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

@app.route('/aruco')
def aruco():
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
            <img src="/aruco_detection" />
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_telemetry_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/aruco_detection')
def aruco_feed():
    return Response(generate_aruco_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    cam_server = None
    cam_client = None
    try:
        print("Initializing optical sensors...")
        cam_server = setup_video_sync_node(SERVER_ID , "server", size=(1536,864), framerate=(20,30))
        cam_client = setup_video_sync_node(CLIENT_ID, "client", size=(1640, 1232), framerate=(20, 35))
        
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