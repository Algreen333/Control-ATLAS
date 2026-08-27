import time
import cv2
from flask import Flask, Response, redirect, request
from threading import Thread, Lock
import numpy as np
import json

from typing import Optional, Tuple


from lib.aruco_lib import *

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except (ImportError, RuntimeError):
    HAS_PICAMERA = False

def joint_display(img_server, img_client, drift_ms = 0, tvec=None):
        """Formats, dynamically pads, and annotates frames for combined visualization."""
        h_server, w_server = img_server.shape[:2]
        h_client, w_client = img_client.shape[:2]
        
        # Pad to match heights if resolutions differ
        if h_server > h_client:
            padding = h_server - h_client
            img_client = cv2.copyMakeBorder(img_client, 0, padding, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif h_client > h_server:
            padding = h_client - h_server
            img_server = cv2.copyMakeBorder(img_server, 0, padding, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # PiCamera2 usually outputs RGB. OpenCV imencode expects BGR.
        img_server_bgr = cv2.cvtColor(img_server, cv2.COLOR_RGB2BGR)
        img_client_bgr = cv2.cvtColor(img_client, cv2.COLOR_RGB2BGR)

        # Draw drift data
        drift_text = f"{drift_ms:.3f} ms" if drift_ms is not None else "N/A"
        status_color = (0, 255, 0) if drift_ms is not None and drift_ms < 1.0 else (0, 0, 255)
        
        cv2.putText(img_server_bgr, f"SERVER | Drift: {drift_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(img_client_bgr, f"CLIENT | Drift: {drift_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Draw pose data if available
        if tvec is not None:
            pose_text = f"Pose [X: {tvec[0][0]:.2f} Y: {tvec[1][0]:.2f} Z: {tvec[2][0]:.2f}]"
            cv2.putText(img_server_bgr, pose_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Concatenate horizontally and scale down
        combined_frame = cv2.hconcat([img_server_bgr, img_client_bgr])
        resized = cv2.resize(combined_frame, (0, 0), fx=0.5, fy=0.5)

        return resized

class VideoCapture:
    def __init__(self, capture_source = 0, resolution=(640, 480), format="RGB888", fps=30.0):
        """
        :param any capture_source: Source of the capture. Irrelevant when using Picamera2
        :param (int,int) resolution: Resolution of the capture. By default (640,480)
        :param str format: Format of the capture. Only relevant when using Picamera2. By default "RGB888"
        :param float fps: Fps of the capture. By default 30.0
        """
        
        self.resolution = resolution
        self.fps = fps
        self.format = format
        self.capture_source = capture_source
        
        if HAS_PICAMERA:
            self.capture = Picamera2(capture_source)
            config = self.capture.create_preview_configuration(main={"size": self.resolution, "format": self.format})
            self.capture.configure(config)

        else:
            self.capture = cv2.VideoCapture(self.capture_source)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        # For recording
        self.is_recording = False
        self.video_writer = None    

        self.frame = None
        self.stopped = False
        self.lock = Lock()
        
        self.capture_thread = Thread(target=self._update, daemon=True)

    def start(self):
        """
        Inicializes capture.

        :return self (VideoCapture):
        """


        if HAS_PICAMERA: self.capture.start()
        self.capture_thread.start()

        time.sleep(1) 

        print("[*] Video capture started")
        return self

    def _update(self):
        while not self.stopped:
            if HAS_PICAMERA:
                new_frame = self.capture.capture_array()
            else:
                ret, new_frame = self.capture.read()
                if not ret:
                    continue
                
            with self.lock:
                self.frame = new_frame

    def read(self):
        """
        Read a frame.
        
        :return ret (bool): Whether the frame has been read successfully
        :return frame (MatLike): The captured frame 
        """
        try:
            with self.lock:
                return True, self.frame
        except: 
            return False, None

    def toggle_record(self, filename=None):
        """
        Toggles record state. If it is not recording it creates opens a new VideoWriter and starts writing a video.
        
        :param str (optional) filename: Filename of the video. If left on None the video will be created with name "rec_time.avi"
        """
        if not self.is_recording:
            self.is_recording = True
            if filename is None: filename = f"rec_{int(time.time())}.avi"
            self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.resolution)
            Thread(target=self._record_loop, daemon=True).start()
        else:
            self.is_recording = False
            self.video_writer.release()

    def _record_loop(self):
        while self.is_recording:
            ret, f = self.read()
            if ret:    
                if f is not None: self.video_writer.write(f)
                time.sleep(1/self.fps)

        self.video_writer.release()

    def stop(self):
        """
        Stops video capture
        """

        self.stopped = True
        if self.capture_thread.is_alive():
            self.capture_thread.join()

        if HAS_PICAMERA:
            self.capture.stop()
            self.capture.close()
        else: 
            self.capture.release()


class VideoServer:
    def __init__(self, streamer:VideoCapture, host='0.0.0.0', port=5000):
        """
        Starts video server for a video capture. It is a Flask server which shows the capture in real time.

        :param VideoCapture streamer: VideoCapture whcih to show
        :param str host: Host address. By default "0.0.0.0"
        :param int port: Host port. By default 5000
        """

        self.streamer:VideoCapture = streamer
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self.video_writer = None
        self.is_recording = False

        self.custom_endpoints = {"/video_feed": "Raw Stream"}
        
        # Register routes
        self.app.add_url_rule('/', 'index', lambda: self._render_page("/video_feed_raw"))
        self.app.add_url_rule('/video_feed', 'video_feed', lambda: self._render_page("/video_feed_raw"))
        self.app.add_url_rule('/video_feed_raw', 'video_feed_raw', self.video_feed_raw)
        self.app.add_url_rule('/toggle_record', 'toggle_record', self.toggle_record, methods=['POST'])

    def _render_page(self, stream_url):
        """The Master Template: Fullscreen with Hover UI"""
        rec_color = "#ff4444" if self.streamer.is_recording else "#44bb44"
        rec_text = "Stop Recording" if self.streamer.is_recording else "Start Recording"
        
        # Build navigation links for the overlay
        nav_links = "".join([f'<a href="{url}" style="color:white; margin-right:15px; text-decoration:none; font-size:14px; opacity:0.7;">{name}</a>' 
                             for url, name in self.custom_endpoints.items()])

        return f"""
        <html>
            <head>
                <style>
                    body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: black; font-family: sans-serif; }}
                    .viewport {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }}
                    .overlay {{ position: absolute; top: 0; left: 0; width: 100%; padding: 20px; box-sizing: border-box;
                                background: linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, transparent 100%);
                                color: white; opacity: 0; transition: opacity 0.3s; z-index: 10; display: flex; justify-content: space-between; }}
                    .overlay:hover {{ opacity: 1; }}
                    .btn {{ padding: 10px 20px; background: {rec_color}; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }}
                </style>
            </head>
            <body>
                <img class="viewport" src="{stream_url}">
                <div class="overlay">
                    <div>
                        <h2 style="margin:0 0 10px 0;">PiStream Portal</h2>
                        {nav_links}
                    </div>
                    <form action="/toggle_record" method="post" style="margin:0;">
                        <input type="hidden" name="next" value="{stream_url}">
                        <button class="btn">{rec_text}</button>
                    </form>
                </div>
            </body>
        </html>
        """

    def toggle_record(self, filename = None):
        """
        Toggles record state. If it is not recording it creates opens a new VideoWriter and starts writing a video.
        
        :param str (optional) filename: Filename of the video. If left on None the video will be created with name "rec_time.avi"
        """

        if not self.is_recording:
            self.is_recording = True
            filename = f"recordings/rec_{int(time.time())}.avi"
            self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.streamer.fps, self.streamer.resolution)
        else:
            self.is_recording = False
            self.video_writer.release()
        return redirect(request.form.get('next', '/'))

    def _generate_frames(self, processor=None):
        prev_frame = None
        start_time = time.time()

        while True:
            try:
                ret, frame = self.streamer.read()

                if not ret or frame is None:
                    # Si hi ha un error o el frame no és vàlid.
                    # Si està gravant escriurà un altre cop el frame anterior
                    if self.is_recording and self.video_writer is not None:
                        time.sleep(max(0, (time.time() - start_time - (1/self.streamer.fps))))
                        if prev_frame is not None: self.video_writer.write(prev_frame)
                        start_time = time.time()
                    continue
                
                if processor is not None: frame = processor(frame)


                # Convert NumPy array to JPEG
                _, buffer = cv2.imencode('.jpg', frame)

                if self.is_recording and self.video_writer is not None:
                    time.sleep(max(0, (time.time() - start_time - (1/self.streamer.fps))))
                    self.video_writer.write(prev_frame)
                    start_time = time.time()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print("'_generate_frames' exception:", e)

                if prev_frame is not None:
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def index(self):
        return Response(self._generate_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def start(self):
        server_thread = Thread(
            target=lambda: self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False),
            daemon=True
        )
        server_thread.start()
        print(f"[*] Web server live at http://{self.host}:{self.port}")

    def create_route(self, rule, endpoint_name, generator):
        """
        Exemple: create_route('/aruco_detector', 'aruco_detector', aruco_detector), 
        on aruco_detector() és una funció aruco_detector(frame) -> frame_processat
        """
        self.custom_endpoints[rule] = endpoint_name
        self.app.add_url_rule(f"{rule}_raw", f"{endpoint_name}_raw", 
                              lambda: Response(self._generate_frames(processor=generator), 
                                    mimetype='multipart/x-mixed-replace; boundary=frame')
        )
        self.app.add_url_rule(rule, endpoint_name, 
                              lambda: self._render_page(f"{rule}_raw"))
    
    def video_feed_raw(self):
        return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


class CalibrationConfig:
    """
    Can be initialized with:
    - CalibrationConfig(mtx:np.ndarray, dist:np.ndarray) 
    - CalibrationConfig.from_path(path): path is the config file path
    - CalibrationConfig.from_lists(mtx:list, dist:list)
    """

    def __init__(self, mtx: np.ndarray, dist: np.ndarray, image_size:tuple=(640, 480), alpha=0):
        # The main constructor only accepts the finalized numpy arrays
        self.mtx = mtx
        self.dist = dist
        self.image_size = image_size

        self.new_cameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, self.image_size, alpha, self.image_size
        )
        
        # Pre-compute the undistortion and rectification transformation maps 
        # cv2.CV_16SC2 is used for the map type as it represents a highly efficient fixed-point format for cv2.remap
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.new_cameramtx, self.image_size, cv2.CV_16SC2
        )
        
        # Extract Region of Interest (ROI) to optionally crop black borders later
        self.x, self.y, self.w, self.h = self.roi

    def process(self, frame, crop=False):
        """
        Applies the pre-computed undistortion maps to a frame.
        
        :param frame: The distorted BGR/RGB numpy array.
        :param crop: Boolean. If True, crops the image to the valid Region of Interest.
        :return: Undistorted numpy array.
        """
        # Apply the pre-computed spatial mapping
        # INTER_LINEAR is the best balance between visual fidelity and processing speed
        undistorted_frame = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
        
        if crop and self.w > 0 and self.h > 0:
            undistorted_frame = undistorted_frame[self.y:self.y+self.h, self.x:self.x+self.w]
            
        return undistorted_frame

    @classmethod
    def from_path(self, path: str):
        # Read the file, extract matrices
        with open(path, "r") as f:
            data = json.load(f)

        loaded_mtx = np.array(data["mtx"])
        loaded_dist = np.array(data["dist"])
        loaded_image_size = (data["size"][0], data["size"][1])

        return self(loaded_mtx, loaded_dist, image_size = loaded_image_size)

    @classmethod
    def from_lists(self, mtx: list, dist: list):
        return self(np.array(mtx), np.array(dist))
    
    def save(self, path:str):
        config_data = {
            "mtx": self.mtx.tolist(),
            "dist": self.dist.tolist(),
            "size": self.image_size.tolist()
        }
        with open(path, "w") as f:
            json.dump(config_data, f, indent=4)

    
class TelemetryServer:
    def __init__(self, capture_instance, host='0.0.0.0', port=5000):
        """
        Initializes the web server using an existing StereoCapture instance.
        
        :param capture_instance: Initialized and running StereoCapture object.
        """

        self.capture = capture_instance
        self.host = host
        self.port = port
        self.currentPhase = "Non Autonomous"
        self.currentAltitude = None

        self.app = Flask(__name__)
        self.setup_routes()

    def setPhase(self, phase: str):
        self.currentPhase = phase

    def setAltitude(self, alt_m: float):
        self.currentAltitude = round(alt_m, 2)

    def telemetry(self):
        from flask import jsonify
        return jsonify({
            'phase': self.currentPhase,
            'altitude': self.currentAltitude
        })

    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/telemetry', 'telemetry', self.telemetry)

    def joint_display(self, img_server, img_client, drift_ms, tvec=None):
        """Formats, dynamically pads, and annotates frames for combined visualization."""
        h_server, w_server = img_server.shape[:2]
        h_client, w_client = img_client.shape[:2]
        
        # Pad to match heights if resolutions differ
        if h_server > h_client:
            padding = h_server - h_client
            img_client = cv2.copyMakeBorder(img_client, 0, padding, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif h_client > h_server:
            padding = h_client - h_server
            img_server = cv2.copyMakeBorder(img_server, 0, padding, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # PiCamera2 usually outputs RGB. OpenCV imencode expects BGR.
        img_server_bgr = cv2.cvtColor(img_server, cv2.COLOR_RGB2BGR)
        img_client_bgr = cv2.cvtColor(img_client, cv2.COLOR_RGB2BGR)

        # Draw drift data
        drift_text = f"{drift_ms:.3f} ms" if drift_ms is not None else "N/A"
        status_color = (0, 255, 0) if drift_ms is not None and drift_ms < 1.0 else (0, 0, 255)
        
        cv2.putText(img_server_bgr, f"SERVER | Drift: {drift_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(img_client_bgr, f"CLIENT | Drift: {drift_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Draw pose data if available
        if tvec is not None:
            pose_text = f"Pose [X: {tvec[0][0]:.2f} Y: {tvec[1][0]:.2f} Z: {tvec[2][0]:.2f}]"
            cv2.putText(img_server_bgr, pose_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Concatenate horizontally and scale down
        combined_frame = cv2.hconcat([img_server_bgr, img_client_bgr])
        resized = cv2.resize(combined_frame, (0, 0), fx=0.5, fy=0.5)

        return resized

    def generate_stream(self):
        """Continuously pulls frames from the capture object and streams them."""
        while True:
            # 1. Poll the capture object for new frames
            serv, clnt, drift_ms = self.capture.read()
            ret_s, frame_s = serv
            ret_c, frame_c = clnt

            # Wait if hardware hasn't pushed frames to the buffer yet
            if not ret_s or not ret_c or frame_s is None or frame_c is None:
                time.sleep(0.05)
                continue

            # 2. Fetch latest pose metrics from the capture object's process loop
            pose = self.capture.get_latest_pose()
            tvec, rvec = pose if pose else (None, None)

            # 3. Assemble the UI
            frame_to_encode = joint_display(frame_s, frame_c, drift_ms, tvec)

            # 4. Compress to JPEG
            ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            # Throttle web stream slightly to save CPU for the actual ArUco math
            time.sleep(0.05)

    def index(self):
        return '''
        <html>
        <head>
            <title>UAV Telemetry</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { background: #000; font-family: monospace; overflow: hidden; height: 100vh; }

                /* Video fills the screen */
                #stream { width: 100%; height: 100vh; object-fit: contain; display: block; }

                /* HUD overlay — always visible, top-left corner */
                #hud {
                    position: fixed; top: 0; left: 0;
                    padding: 14px 18px;
                    background: linear-gradient(135deg, rgba(0,0,0,0.75), rgba(0,0,0,0.4));
                    color: #0f0; border-bottom-right-radius: 10px;
                    z-index: 99;
                }
                #hud .label { font-size: 11px; color: #888; letter-spacing: 1px; text-transform: uppercase; }
                #hud .value { font-size: 22px; font-weight: bold; margin-bottom: 10px; }
                #hud .value.phase { color: #0ff; }
                #hud .value.alt   { color: #0f0; }

                /* Dot that blinks to show live data */
                #live-dot {
                    position: fixed; top: 12px; right: 14px;
                    width: 10px; height: 10px; border-radius: 50%;
                    background: #f00; z-index: 99;
                    animation: blink 1s infinite;
                }
                @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
            </style>
        </head>
        <body>
            <div id="hud">
                <div class="label">Phase</div>
                <div class="value phase" id="phase-val">---</div>
                <div class="label">Altitude</div>
                <div class="value alt" id="alt-val">--- m</div>
            </div>
            <div id="live-dot"></div>
            <img id="stream" src="/video_feed" />

            <script>
                async function poll() {
                    try {
                        const r = await fetch('/telemetry');
                        const d = await r.json();
                        document.getElementById('phase-val').textContent = d.phase ?? '---';
                        document.getElementById('alt-val').textContent =
                            d.altitude !== null ? d.altitude.toFixed(1) + ' m' : '--- m';
                    } catch(e) {}
                }
                poll();
                setInterval(poll, 500);
            </script>
        </body>
        </html>
        '''

    def video_feed(self):
        return Response(self.generate_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def start(self):
        """Starts the Flask server. This is a blocking call."""
        print(f"[TELEMETRY] Starting web server on {self.host}:{self.port}...")
        self.app.run(host=self.host, port=self.port, threaded=True)

    def start_background(self):
        """Starts the Flask server in a non-blocking background thread."""
        print(f"[TELEMETRY] Starting background web server on {self.host}:{self.port}...")
        
        # We wrap app.run in a thread. 
        # use_reloader=False is REQUIRED when running Flask outside the main thread.
        self.server_thread = Thread(
            target=self.app.run,
            kwargs={'host': self.host, 'port': self.port, 'threaded': True, 'use_reloader': False},
            daemon=True # Daemon ensures this thread dies automatically when your main script ends
        )
        self.server_thread.start()
        
        return self.server_thread

class StereoCapture:
    def __init__(self, idx_server, size_server, idx_client, size_client, 
                 T_C_WIDE=None, T_C_NARR=None, arucoDICT = aruco.DICT_4X4_50, marker_size = 1, marker_id = 49,
                 configWide:CalibrationConfig = None, configNarr:CalibrationConfig = None):
        self.capture_server = self.setup_video_sync_node_picamera(idx_server, "server", size=size_server)
        self.capture_client = self.setup_video_sync_node_picamera(idx_client, "client", size=size_client)

        self.lock = Lock()
        self.latest_frame_server = None
        self.latest_frame_client = None
        self.latest_drift_ms = None

        self.running = False
        self.thread = None
        self.latest_tvec = None
        self.latest_rvec = None
        self.latest_display_frame = None

        self.T_C_WIDE = T_C_WIDE
        self.T_C_NARR = T_C_NARR

        self.MARKER_SIZE = marker_size
        self.MARKER_ID = marker_id

        if configWide is not None:
            self.detWide = ArucoDetector(configWide.mtx, configWide.dist, arucoDICT)
        if configNarr is not None:
            self.detNarr = ArucoDetector(configNarr.mtx, configNarr.dist, arucoDICT)
        
        self.do_process = False

    def setup_video_sync_node_picamera(self, cam_index, sync_mode, size=(640, 480), format="RGB888", framerate=30.0):
        from picamera2 import Picamera2
        from libcamera import controls
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

        picam.set_controls({
            "NoiseReductionMode": 2,
            "ExposureTime": 200,       # microseconds — go as low as lighting allows
            "AnalogueGain": 60.0,        # compensate brightness with gain
            "AeEnable": False,          # disable auto-exposure so it doesn't fight you
        })

            
        picam.start()
        return picam

    def _update_loop(self):
        """Pulls continuous synchronized frames and monitors temporal drift."""
        time.sleep(2)
                
        try:
            while self.running:
                # Capture continuous requests from the hardware [cite: 60]
                req_server = self.capture_server.capture_request()
                req_client = self.capture_client.capture_request()
                
                # Extract the precise nanosecond the sensor began reading out the frame [cite: 63]
                ts_server = req_server.get_metadata().get('SensorTimestamp', 0)
                ts_client = req_client.get_metadata().get('SensorTimestamp', 0)
                
                drift_ns = abs(ts_server - ts_client)
                self.latest_drift_ms = drift_ns / 1_000_000.0

                    
                #print(f"1-Frame offset detected ({self.latest_drift_ms:.3f} ms). Flushing older frame...")
                if drift_ns < 0:
                    # Server frame is older, drop it and pull a new one to catch up
                    req_server.release()
                    req_server = self.capture_server.capture_request()
                else:
                    # Client frame is older, drop it and pull a new one to catch up
                    req_client.release()
                    req_client = self.capture_client.capture_request()
                
                # Extract image buffers
                img_server = req_server.make_array("main")
                img_client = req_client.make_array("main")

                img_client = cv2.rotate(img_client, cv2.ROTATE_180)
                
                # Release requests to prevent memory exhaustion
                req_server.release()
                req_client.release()

                with self.lock:
                    self.latest_frame_server = img_server.copy()
                    self.latest_frame_client = img_client.copy()

                if self.do_process:
                    self.process(True, img_server, True, img_client)
                    
        finally:
            cv2.destroyAllWindows()

    def start(self, do_process=False):
        self.do_process = do_process
        self.running = True
        self.thread = Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("[SYNC-CAP] Sync camera thread started.")
        return self
    
    def read(self):
        """
        Returns latest frames captured
        
        :return frame_server: (ret, frame) of the server capture
        :return frame_client: (ret, frame) of the client capture
        :return drift_ms (float | None): Time drift (in ms) of the captures.
        """
        with self.lock:
            if self.latest_frame_server is not None: serv = (True, self.latest_frame_server)
            else: serv = (False, None)

            if self.latest_frame_client is not None: clnt = (True, self.latest_frame_client)
            else: clnt = (False, None)

            if self.latest_drift_ms is not None: drft = self.latest_drift_ms
            else: drft = None

            return serv, clnt, drft
    
    def process(self, ret_w, frame_wide, ret_n, frame_narr):
        rvec_wide = None
        tvec_wide = None
        rvec_narr = None
        tvec_narr = None
        rvec = None
        tvec = None

        if ret_w:
            gray = cv2.cvtColor(frame_wide, cv2.COLOR_BGR2GRAY)
            corners_wide, ids_wide, rejected_wide = self.detWide.detectMarkers(gray)
            if ids_wide is not None:                                          
                indices = np.where(ids_wide == self.MARKER_ID)[0]
                if len(indices) > 0:
                    rvec_wide, tvec_wide = self.detWide.estimate_pose(corners_wide[indices[0]], self.MARKER_SIZE)
                    frame_wide = aruco.drawDetectedMarkers(frame_wide, corners_wide, ids_wide)

        if ret_n:
            gray = cv2.cvtColor(frame_narr, cv2.COLOR_BGR2GRAY)
            corners_narr, ids_narr, rejected_narr = self.detNarr.detectMarkers(gray)
            if ids_narr is not None:                                          
                indices = np.where(ids_narr == self.MARKER_ID)[0]
                if len(indices) > 0:
                    rvec_narr, tvec_narr = self.detNarr.estimate_pose(corners_narr[indices[0]], self.MARKER_SIZE)
                    frame_narr = aruco.drawDetectedMarkers(frame_narr, corners_narr, ids_narr)

        # Estimation from both cameras
        if rvec_wide is not None and rvec_narr is not None and self.T_C_WIDE is not None and self.T_C_NARR is not None:
            rvec, tvec = fuse_stereo_aruco_poses(rvec_wide, tvec_wide, self.T_C_WIDE, rvec_narr, tvec_narr, self.T_C_NARR)

        elif rvec_wide is not None:
            rvec, tvec = transform_auco_poses(rvec_wide, tvec_wide, self.T_C_WIDE)

        elif rvec_narr is not None:
            rvec, tvec = transform_auco_poses(rvec_narr, tvec_narr, self.T_C_NARR)
        
        # Safely update the latest pose variables
        with self.lock:
            if rvec is not None and tvec is not None:
                self.latest_tvec = tvec
                self.latest_rvec = rvec
            else:
                self.latest_tvec = None
                self.latest_rvec = None

            self.latest_frame_server = frame_wide
            self.latest_frame_client = frame_narr
        
    def get_latest_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fetch the most recently estimated pose without blocking.
        Returns (tvec, rvec) or None.
        """
        with self.lock:
            if self.latest_tvec is not None and self.latest_rvec is not None:
                return self.latest_tvec, self.latest_rvec
            return None
        

class GazeboStereoCapture:
    def __init__(self, pipeline_wide, pipeline_narr,
        mtx_wide,
        dst_wide,
        mtx_narr,
        dst_narr):
        
        self.cap_wide = cv2.VideoCapture(pipeline_wide, cv2.CAP_GSTREAMER)
        print("[SYSTEM] Capture 'wide' open?", self.cap_wide.isOpened())

        self.cap_narr = cv2.VideoCapture(pipeline_narr, cv2.CAP_GSTREAMER)
        print("[SYSTEM] Capture 'narrow' open?", self.cap_narr.isOpened())

        self.mtx_wide = mtx_wide
        self.dst_wide = dst_wide
        self.mtx_narr = mtx_narr
        self.dst_narr = dst_narr

        self.detWide = ArucoDetector(mtx_wide, dst_wide, cv2.aruco.DICT_4X4_50)
        self.detNarr = ArucoDetector(mtx_wide, dst_wide, cv2.aruco.DICT_4X4_50)

        self.T_C_WIDE = np.eye(4)
        self.T_C_WIDE[0, 3] = 0.09675
        self.T_C_NARR = np.eye(4)
        self.T_C_NARR[0, 3] = -0.09675

        self.MARKER_SIZE = 1
        self.MARKER_ID = 49

        # Threading mechanisms
        self.lock = Lock()
        self.running = False
        self.thread = None
        self.latest_tvec = None
        self.latest_rvec = None
        self.latest_display_frame = None

    def start(self):
        """Starts the background thread for continuous pose estimation."""
        self.running = True
        self.thread = Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("[SYSTEM] Camera processing thread started.")
        return self

    def stop(self):
        """Stops the background thread safely."""
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _update_loop(self):
        """Continuously captures frames and calculates the pose."""
        while self.running:
            ret_w, frame_wide = self.cap_wide.read()
            ret_n, frame_narr = self.cap_narr.read()

            rvec_wide = None
            tvec_wide = None
            rvec_narr = None
            tvec_narr = None
            rvec = None
            tvec = None

            if ret_w:
                gray = cv2.cvtColor(frame_wide, cv2.COLOR_BGR2GRAY)
                corners_wide, ids_wide, rejected_wide = self.detWide.detectMarkers(gray)
                if ids_wide is not None:                                          
                    indices = np.where(ids_wide == self.MARKER_ID)[0]
                    if len(indices) > 0:
                        rvec_wide, tvec_wide = self.detWide.estimate_pose(corners_wide[indices[0]], self.MARKER_SIZE)
                        frame_wide = aruco.drawDetectedMarkers(frame_wide, corners_wide, ids_wide)
                        x, y, w, h = cv2.boundingRect(corners_wide[indices[0]])
                        cv2.rectangle(frame_wide, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if ret_n:
                gray = cv2.cvtColor(frame_narr, cv2.COLOR_BGR2GRAY)
                corners_narr, ids_narr, rejected_narr = self.detNarr.detectMarkers(gray)
                if ids_narr is not None:                                          
                    indices = np.where(ids_narr == self.MARKER_ID)[0]
                    if len(indices) > 0:
                        rvec_narr, tvec_narr = self.detNarr.estimate_pose(corners_narr[indices[0]], self.MARKER_SIZE)
                        frame_narr = aruco.drawDetectedMarkers(frame_narr, corners_narr, ids_narr)
                        x, y, w, h = cv2.boundingRect(corners_narr[indices[0]])
                        cv2.rectangle(frame_narr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimation from both cameras
            if rvec_wide is not None and rvec_narr is not None:
                rvec, tvec = fuse_stereo_aruco_poses(rvec_wide, tvec_wide, self.T_C_WIDE, rvec_narr, tvec_narr, self.T_C_NARR)

            elif rvec_wide is not None:
                rvec, tvec = transform_auco_poses(rvec_wide, tvec_wide, self.T_C_WIDE)

            elif rvec_narr is not None:
                rvec, tvec = transform_auco_poses(rvec_narr, tvec_narr, self.T_C_NARR)
            
            # Safely update the latest pose variables
            with self.lock:
                if rvec is not None and tvec is not None:
                    self.latest_tvec = tvec
                    self.latest_rvec = rvec
                else:
                    self.latest_tvec = None
                    self.latest_rvec = None
                
                if ret_w and ret_n:
                    self.latest_display_frame = joint_display(frame_wide, frame_narr, 0, tvec)

    def get_latest_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fetch the most recently estimated pose without blocking.
        Returns (tvec, rvec) or None.
        """
        with self.lock:
            if self.latest_tvec is not None and self.latest_rvec is not None:
                return self.latest_tvec, self.latest_rvec
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Fetch the most recent display frame for OpenCV imshow."""
        with self.lock:
            if self.latest_display_frame is not None:
                # Return a copy to prevent the main thread from reading 
                # while the background thread is overwriting it
                return self.latest_display_frame.copy() 
            return None

if __name__ == "__main__":

    cap = VideoCapture(0)
    cap.start()

    server = VideoServer(cap)
    server.start()