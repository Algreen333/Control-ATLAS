import cv2
import json
import numpy as np
from datetime import datetime
import os

try:
    from picamera2 import Picamera2
    from libcamera import controls
    HAS_PICAMERA = True
except (ImportError, RuntimeError):
    HAS_PICAMERA = False

import logging
logger = logging.getLogger(__name__)


def save_img_dir(capture, directory:str):
    fname = f"{datetime.now().strftime('%F_%T.%f')[:-3]}.jpg"
    path = os.path.join(directory, fname)
    
    cv2.imwrite(path, capture)

class VideoCapture:
    def __init__(self, capture_source = 0, resolution=(640, 480), format="RGB888", fps=30.0, AeEnable=False, ExposureTime=50000, AnalogueGain=2.0, Brightness=0.1):
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
            self.HAS_PICAMERA = True
            self.capture = Picamera2(capture_source)
            config = self.capture.create_preview_configuration(main={"size": self.resolution, "format": self.format})
            self.capture.configure(config)
            self.capture.set_controls({
                "AeEnable": AeEnable,
                "ExposureTime": ExposureTime,
                "AnalogueGain": AnalogueGain,
                "Brightness": Brightness
            })
            self.capture.start()

        else:
            self.HAS_PICAMERA = False
            self.capture = cv2.VideoCapture(self.capture_source)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self):
        if HAS_PICAMERA:
            new_frame = self.capture.capture_array()
            return (True, new_frame)
        return self.capture.read()

class StereoCapture:
    def __init__(self, idx_server=0, idx_client=1, resolution=(640, 480), format="RGB888", fps=30.0, 
                 cam1_cfg=None, cam2_cfg=None, max_drift_ns=5_000_000):
        self.resolution = resolution
        self.fps = fps
        self.format = format
        self.max_drift_ns = max_drift_ns
        
        self.cam1_cfg = cam1_cfg or {}
        self.cam2_cfg = cam2_cfg or {}
        
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        
        self.latest_frame_server = None
        self.latest_frame_client = None
        self.latest_drift_ms = None
        
        if not HAS_PICAMERA:
            logger.error("Picamera2 is required for hardware synchronized StereoCapture.")
            raise RuntimeError("Picamera2 not found.")

        self.capture_server = self._setup_node(idx_server, "server", self.cam1_cfg)
        self.capture_client = self._setup_node(idx_client, "client", self.cam2_cfg)

    def _setup_node(self, cam_index, sync_mode, cfg):
        picam = Picamera2(cam_index)
        config = picam.create_video_configuration(main={"size": self.resolution, "format": self.format})
        
        config["main"]["queue"] = False
        config["main"]["buffer_count"] = 6
        config["main"]["framerate"] = self.fps
        picam.configure(config)
        
        controls_dict = {
            "NoiseReductionMode": 2,
            "ExposureTime": cfg.get("ExposureTime", 50000),
            "AnalogueGain": cfg.get("AnalogueGain", 2.0),
            "AeEnable": cfg.get("AeEnable", False),
            "Brightness": cfg.get("Brightness", 0.0),
            "Contrast": cfg.get("Contrast", 1.0)
        }
        
        if sync_mode == "server":
            controls_dict["SyncMode"] = controls.rpi.SyncModeEnum.Server
        elif sync_mode == "client":
            controls_dict["SyncMode"] = controls.rpi.SyncModeEnum.Client

        picam.set_controls(controls_dict)
        picam.start()
        return picam
        
    def _apply_rotation(self, img, angle):
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _update_loop(self):
        time.sleep(2) # Camera warmup
        
        while self._running:
            try:
                req_server = self.capture_server.capture_request()
                req_client = self.capture_client.capture_request()
                
                ts_server = req_server.get_metadata().get('SensorTimestamp', 0)
                ts_client = req_client.get_metadata().get('SensorTimestamp', 0)
                
                time_diff_ns = ts_server - ts_client
                drift_ns = abs(time_diff_ns)
                
                with self._lock:
                    self.latest_drift_ms = drift_ns / 1_000_000.0
                
                if drift_ns > self.max_drift_ns:
                    logger.warning(f"Desync detected: {self.latest_drift_ms:.2f}ms. Realigning...")
                    if time_diff_ns < 0:
                        req_server.release()
                        req_server = self.capture_server.capture_request()
                    else:
                        req_client.release()
                        req_client = self.capture_client.capture_request()
                        
                img_server = req_server.make_array("main")
                img_client = req_client.make_array("main")
                
                img_server = self._apply_rotation(img_server, self.cam1_cfg.get("rotation", 0))
                img_client = self._apply_rotation(img_client, self.cam2_cfg.get("rotation", 0))
                
                req_server.release()
                req_client.release()
                
                with self._lock:
                    self.latest_frame_server = img_server.copy()
                    self.latest_frame_client = img_client.copy()
                    
            except RuntimeError as e:
                # DmaSync errors happen if the camera shuts down while a frame is processing
                if not self._running:
                    break
                else:
                    logger.error(f"StereoCapture frame grab error: {e}")
            except Exception as e:
                if self._running:
                    logger.error(f"Unexpected StereoCapture error: {e}")

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("[SYNC-CAP] Sync camera thread started.")
        return self
        
    def read(self):
        with self._lock:
            if self.latest_frame_server is not None:
                serv = (True, self.latest_frame_server.copy())
            else:
                serv = (False, None)
                
            if self.latest_frame_client is not None:
                clnt = (True, self.latest_frame_client.copy())
            else:
                clnt = (False, None)
                
            drft = self.latest_drift_ms
            
        return serv, clnt, drft

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        
        try:
            if self.capture_server:
                self.capture_server.stop()
                self.capture_server.close()
        except Exception as e: logger.error(f"Error closing server cam: {e}")
            
        try:
            if self.capture_client:
                self.capture_client.stop()
                self.capture_client.close()
        except Exception as e: logger.error(f"Error closing client cam: {e}")

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



class GazeboVideoCapture:
    def __init__(self, capture_source = 0, resolution=(640, 480), format="RGB888", fps=30.0):
        """
        :param any capture_source: Source of the capture. Irrelevant when using Picamera2
        :param (int,int) resolution: Resolution of the capture. By default (640,480)
        :param str format: Format of the capture. Only relevant when using Picamera2. By default "RGB888"
        :param float fps: Fps of the capture. By default 30.0
        """

        VIDSRC_PORT_NARR = 5600
        pipeline_narr = (
            f"udpsrc port={VIDSRC_PORT_NARR} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
            "rtph264depay ! "
            "avdec_h264 ! "
            "videoconvert ! "
            "appsink drop=1"
        )
        
        self.resolution = resolution
        self.fps = fps
        self.format = format
        self.capture_source = capture_source
        
        self.capture = cv2.VideoCapture(pipeline_narr, cv2.CAP_GSTREAMER)


    def read(self):
        return self.capture.read()


import time
import threading
from flask import Flask, Response, render_template_string, request
from werkzeug.serving import make_server

class FlaskPreviewServer:
    def __init__(self, host="0.0.0.0", port=5000, max_fps=15, jpeg_quality=55, tuning_enabled=False, pid_controllers=None, config_path=None):
        self.host = host
        self.port = port
        self.frame_delay = 1.0 / max_fps
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        
        self.tuning_enabled = tuning_enabled
        self.pid_controllers = pid_controllers or []
        self.config_path = config_path
        
        self._frame = None
        self._drift_ms = None
        self._lock = threading.Lock()
        self._running = False
        self._server = None
        self._thread = None
        
        self.app = Flask(__name__)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            tuning_html = ""
            if self.tuning_enabled and self.pid_controllers:
                base_kp = self.pid_controllers[0].kp
                base_ki = self.pid_controllers[0].ki
                base_kd = self.pid_controllers[0].kd
                
                tuning_html = f'''
                <div style="position:absolute; top:20px; left:20px; background:rgba(255,255,255,0.9); padding:15px; border-radius:8px; font-family:sans-serif; z-index:1000; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    <h3 style="margin-top:0; margin-bottom:15px;">Live PID Tuning</h3>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:25px; font-weight:bold;">Kp:</label>
                        <input type="range" id="kp_slider" min="0" max="2.0" step="0.01" value="{base_kp}" oninput="syncPID('kp', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="kp_input" min="0" max="2.0" step="0.01" value="{base_kp}" oninput="syncPID('kp', 'input')" style="width:60px; margin-left:10px;">
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:25px; font-weight:bold;">Ki:</label>
                        <input type="range" id="ki_slider" min="0" max="0.2" step="0.001" value="{base_ki}" oninput="syncPID('ki', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="ki_input" min="0" max="0.2" step="0.001" value="{base_ki}" oninput="syncPID('ki', 'input')" style="width:60px; margin-left:10px;">
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:25px; font-weight:bold;">Kd:</label>
                        <input type="range" id="kd_slider" min="0" max="1.0" step="0.01" value="{base_kd}" oninput="syncPID('kd', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="kd_input" min="0" max="1.0" step="0.01" value="{base_kd}" oninput="syncPID('kd', 'input')" style="width:60px; margin-left:10px;">
                    </div>

                    <script>
                        function syncPID(param, source) {{
                            let slider = document.getElementById(param + '_slider');
                            let input = document.getElementById(param + '_input');
                            
                            // If the user cleared the input box, do not update the slider or send the request
                            if (source === 'input' && (input.value === "" || isNaN(parseFloat(input.value)))) {{
                                return;
                            }}
                            
                            // Sync the UI elements bidirectionally
                            if (source === 'slider') {{
                                input.value = slider.value;
                            }} else {{
                                slider.value = input.value;
                            }}
                            
                            let kp = document.getElementById('kp_input').value;
                            let ki = document.getElementById('ki_input').value;
                            let kd = document.getElementById('kd_input').value;
                            
                            // Final safety check: ensure no inputs are empty before sending to the backend
                            if (kp === "" || ki === "" || kd === "" || 
                                isNaN(parseFloat(kp)) || isNaN(parseFloat(ki)) || isNaN(parseFloat(kd))) {{
                                return;
                            }}
                            
                            fetch('/update_pid', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{
                                    kp: parseFloat(kp), 
                                    ki: parseFloat(ki), 
                                    kd: parseFloat(kd)
                                }})
                            }});
                        }}
                    </script>
                </div>
                '''

            return render_template_string(f'''
                <!doctype html>
                <html>
                <head><title>Live Preview</title></head>
                <body style="background:#111;margin:0;display:flex;justify-content:center;align-items:center;height:100vh;position:relative;">
                    <img src="/video_feed" style="max-width:98%;max-height:98%;border-radius:6px;" />
                    {tuning_html}
                </body>
                </html>
            ''')

        @self.app.route('/update_pid', methods=['POST'])
        def update_pid():
            if not self.tuning_enabled:
                return {"status": "forbidden", "message": "Live tuning is disabled in config"}, 403
                
            data = request.json
            
            for pid in self.pid_controllers:
                pid.kp = data.get('kp', pid.kp)
                pid.ki = data.get('ki', pid.ki)
                pid.kd = data.get('kd', pid.kd)
            
            logger.info(f"[PID] Updated in memory: Kp={data.get('kp')}, Ki={data.get('ki')}, Kd={data.get('kd')}")
            
            if self.config_path and os.path.exists(self.config_path):
                try:
                    with self._lock: # Prevent concurrent read/write issues
                        with open(self.config_path, "r") as f:
                            cfg_data = json.load(f)
                            
                        if "pid" not in cfg_data:
                            cfg_data["pid"] = {}
                            
                        cfg_data["pid"]["kp"] = data.get("kp", cfg_data["pid"].get("kp", 0.7))
                        cfg_data["pid"]["ki"] = data.get("ki", cfg_data["pid"].get("ki", 0.02))
                        cfg_data["pid"]["kd"] = data.get("kd", cfg_data["pid"].get("kd", 0.15))
                        
                        with open(self.config_path, "w") as f:
                            json.dump(cfg_data, f, indent=2)
                            
                    logger.info(f"[PID] Saved new values to {self.config_path}")
                except Exception as e:
                    logger.error(f"[PID] Failed to write to config file: {e}")
                    return {"status": "error", "message": str(e)}, 500
                    
            return {"status": "success"}

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    def _generate(self):
        while self._running:
            start_time = time.time()
            frame_to_encode = None
            drift_val = None
            
            with self._lock:
                if self._frame is not None:
                    frame_to_encode = self._frame.copy()
                drift_val = self._drift_ms
            
            if frame_to_encode is not None:
                if drift_val is not None:
                    cv2.putText(frame_to_encode, f"Sync Drift: {drift_val:.2f} ms", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                success, buffer = cv2.imencode('.jpg', frame_to_encode, self.encode_param)
                if success:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            elapsed = time.time() - start_time
            time.sleep(max(0.0, self.frame_delay - elapsed))

    def update_frame(self, frame, drift_ms=None):
        with self._lock:
            self._frame = frame
            self._drift_ms = drift_ms

    def start(self):
        if self._running:
            return
        self._running = True
        self._server = make_server(self.host, self.port, self.app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()