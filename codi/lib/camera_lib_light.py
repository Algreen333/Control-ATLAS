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
    def __init__(self, capture_source=0, resolution=(640, 480), format="RGB888", fps=30.0,
                 AeEnable=False, ExposureTime=50000, AnalogueGain=2.0, Brightness=0.1, rotation=0):
        """
        :param any capture_source: Source of the capture.
        :param (int,int) resolution: Resolution of the capture. By default (640,480)
        :param str format: Format of the capture. Only relevant when using Picamera2. By default "RGB888"
        :param float fps: Fps of the capture. By default 30.0
        """
        self.resolution = resolution
        self.fps = fps
        self.format = format
        self.capture_source = capture_source
        self.exposure_time = ExposureTime
        self.analogue_gain = AnalogueGain
        self.rotation = rotation

        if HAS_PICAMERA:
            self.HAS_PICAMERA = True
            # Ensure we fall back to index 0 if the requested index isn't available when only 1 camera is connected
            available_cams = Picamera2.global_camera_info()
            if not available_cams:
                raise RuntimeError("No cameras detected by Picamera2.")
            cam_idx = capture_source if isinstance(capture_source, int) and capture_source < len(available_cams) else 0

            self.capture = Picamera2(cam_idx)
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

    def _apply_rotation(self, img, angle):
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def read(self):
        if HAS_PICAMERA:
            new_frame = self.capture.capture_array()
            new_frame = self._apply_rotation(new_frame, self.rotation)
            return (True, new_frame)
        ret, frame = self.capture.read()
        if ret and frame is not None:
            frame = self._apply_rotation(frame, self.rotation)
        return ret, frame

    def set_exposure(self, exposure_time: int = None):
        if exposure_time is not None:
            self.exposure_time = int(exposure_time)
        else:
            self.exposure_time = int(self.exposure_time)
        if self.HAS_PICAMERA:
            self.capture.set_controls({
                "AeEnable": False,
                "ExposureTime": self.exposure_time
            })

    def set_analogue_gain(self, analogue_gain: float = None):
        if analogue_gain is not None:
            self.analogue_gain = float(analogue_gain)
        else:
            self.analogue_gain = float(self.analogue_gain)
        if self.HAS_PICAMERA:
            self.capture.set_controls({
                "AeEnable": False,
                "AnalogueGain": self.analogue_gain
            })

    def stop(self):
        if self.HAS_PICAMERA and hasattr(self, "capture") and self.capture is not None:
            try:
                self.capture.stop()
                self.capture.close()
            except Exception as e:
                logger.error(f"Error closing VideoCapture: {e}")
        elif hasattr(self, "capture") and self.capture is not None:
            self.capture.release()

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
        
        self.capture_server = None
        self.capture_client = None

        if not HAS_PICAMERA:
            logger.error("Picamera2 is required for hardware synchronized StereoCapture.")
            raise RuntimeError("Picamera2 not found.")

        available_cams = Picamera2.global_camera_info()
        if len(available_cams) < 2:
            raise RuntimeError(f"StereoCapture requires 2 cameras, but only {len(available_cams)} detected.")

        try:
            self.capture_server = self._setup_node(idx_server, "server", self.cam1_cfg)
            self.capture_client = self._setup_node(idx_client, "client", self.cam2_cfg)
        except Exception as e:
            logger.error(f"Failed to set up stereo camera nodes: {e}")
            self.stop()
            raise

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

    def set_exposure(self, cam_idx: int, exposure_time: int):
        exposure_time = int(exposure_time)
        if cam_idx == 1 and self.capture_server:
            self.cam1_cfg["ExposureTime"] = exposure_time
            self.cam1_cfg["AeEnable"] = False
            self.capture_server.set_controls({"AeEnable": False, "ExposureTime": exposure_time})
        elif cam_idx == 2 and self.capture_client:
            self.cam2_cfg["ExposureTime"] = exposure_time
            self.cam2_cfg["AeEnable"] = False
            self.capture_client.set_controls({"AeEnable": False, "ExposureTime": exposure_time})

    def set_analogue_gain(self, cam_idx: int, analogue_gain: float):
        analogue_gain = float(analogue_gain)
        if cam_idx == 1 and self.capture_server:
            self.cam1_cfg["AnalogueGain"] = analogue_gain
            self.cam1_cfg["AeEnable"] = False
            self.capture_server.set_controls({"AeEnable": False, "AnalogueGain": analogue_gain})
        elif cam_idx == 2 and self.capture_client:
            self.cam2_cfg["AnalogueGain"] = analogue_gain
            self.cam2_cfg["AeEnable"] = False
            self.capture_client.set_controls({"AeEnable": False, "AnalogueGain": analogue_gain})   

    

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
from typing import Optional, Tuple, List

class FlaskPreviewServer:
    def __init__(self, host="0.0.0.0", port=5000, max_fps=15, jpeg_quality=55,
                 tuning_enabled=False, pid_controllers=None,
                 cam_tuning_enabled=False, capture=None, config_path=None,
                 pip_video_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.frame_delay = 1.0 / max_fps
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        
        self.tuning_enabled = tuning_enabled
        self.pid_controllers = pid_controllers or []
        self.cam_tuning_enabled = cam_tuning_enabled
        self.capture = capture
        self.config_path = config_path
        
        self._frame = None
        self._drift_ms = None
        self._filter_mode = "none"
        self._pip_enabled = False
        self._pip_video_path = pip_video_path
        
        self._pip_frame = None
        self._pip_running = False
        self._pip_thread = None
        
        self._lock = threading.Lock()
        self._running = False
        self._server = None
        self._thread = None
        
        if self._pip_video_path and os.path.exists(self._pip_video_path):
            self._start_pip_capture()

        self.app = Flask(__name__)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self._setup_routes()

    def _start_pip_capture(self):
        def pip_loop():
            self._pip_running = True
            while self._pip_running:
                cap = cv2.VideoCapture(self._pip_video_path)
                while cap.isOpened() and self._pip_running:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video back to start
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    with self._lock:
                        self._pip_frame = frame
                    time.sleep(1.0 / 30.0) # Assume 30fps for video loop
                cap.release()
                time.sleep(1)

        self._pip_thread = threading.Thread(target=pip_loop, daemon=True)
        self._pip_thread.start()

    def _apply_preview_filter(self, frame: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none" or frame is None:
            return frame

        h, w = frame.shape[:2]

        if mode == "pixelated":
            scale = 12
            small = cv2.resize(frame, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif mode == "creeper":
            scale = 6
            small = cv2.resize(frame, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(pixelated, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 140)
            
            out = np.zeros_like(frame)
            out[:, :, 1] = cv2.add(gray, edges)
            out[:, :, 0] = gray // 5
            out[:, :, 2] = gray // 5
            return out

        elif mode == "enderman":
            inverted = cv2.bitwise_not(frame)
            b, g, r = cv2.split(inverted)
            g = (g * 0.45).astype(np.uint8)
            b = cv2.add(b, 35)
            r = cv2.add(r, 25)
            return cv2.merge([b, g, r])

        elif mode == "spider":
            small = cv2.resize(frame, (w // 2, h // 2))
            top = np.hstack([small, small])
            bottom = np.hstack([small, small])
            tiled = np.vstack([top, bottom])
            tiled = cv2.resize(tiled, (w, h))
            blended = cv2.addWeighted(frame, 0.55, tiled, 0.45, 0)
            blended[:, :, 0] = (blended[:, :, 0] * 0.6).astype(np.uint8)
            blended[:, :, 1] = (blended[:, :, 1] * 0.6).astype(np.uint8)
            return blended

        elif mode == "nether":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        return frame

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
                    <h3 style="margin-top:0; margin-bottom:15px;">3 useless and 1 very useful sliders</h3>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:85px; font-weight:bold;">enis (rauxa):</label>
                        <input type="range" id="kp_slider" min="0" max="2.0" step="0.01" value="{base_kp}" oninput="syncPID('kp', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="kp_input" min="0" max="2.0" step="0.01" value="{base_kp}" oninput="syncPID('kp', 'input')" style="width:60px; margin-left:10px;">
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:85px; font-weight:bold;">nutil (aura):</label>
                        <input type="range" id="ki_slider" min="0" max="0.2" step="0.001" value="{base_ki}" oninput="syncPID('ki', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="ki_input" min="0" max="0.2" step="0.001" value="{base_ki}" oninput="syncPID('ki', 'input')" style="width:60px; margin-left:10px;">
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:85px; font-weight:bold;">ingus (vibe):</label>
                        <input type="range" id="kd_slider" min="0" max="1.0" step="0.01" value="{base_kd}" oninput="syncPID('kd', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="kd_input" min="0" max="1.0" step="0.01" value="{base_kd}" oninput="syncPID('kd', 'input')" style="width:60px; margin-left:10px;">
                    </div>

                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:85px; font-weight:bold;">useless slider:</label>
                        <input type="range" id="useless" min="0" max="2.0" step="0.01" value="{-1}" style="vertical-align: middle;">
                        <input type="number" id="useless" min="0" max="2.0" step="0.01" value="{-1}" style="width:60px; margin-left:10px;">
                    </div>

                    <script>
                        function syncPID(param, source) {{
                            let slider = document.getElementById(param + '_slider');
                            let input = document.getElementById(param + '_input');
                            
                            if (source === 'input' && (input.value === "" || isNaN(parseFloat(input.value)))) {{
                                return;
                            }}
                            
                            if (source === 'slider') {{
                                input.value = slider.value;
                            }} else {{
                                slider.value = input.value;
                            }}
                            
                            let kp = document.getElementById('kp_input').value;
                            let ki = document.getElementById('ki_input').value;
                            let kd = document.getElementById('kd_input').value;
                            
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

            cam_tuning_html = ""
            if self.cam_tuning_enabled and self.capture is not None:
                is_stereo = isinstance(self.capture, StereoCapture)
                if is_stereo:
                    exp1 = self.capture.cam1_cfg.get("ExposureTime", 700)
                    exp2 = self.capture.cam2_cfg.get("ExposureTime", 700)
                    gain1 = self.capture.cam1_cfg.get("AnalogueGain", 1.0)
                    gain2 = self.capture.cam2_cfg.get("AnalogueGain", 1.0)
                else:
                    exp1 = getattr(self.capture, "exposure_time", 700)
                    exp2 = exp1
                    gain1 = getattr(self.capture, "analogue_gain", 1.0)
                    gain2 = gain1

                cam2_exp_row = ""
                cam2_gain_row = ""
                if is_stereo:
                    cam2_exp_row = f'''
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:65px; font-weight:bold;">Cam 2:</label>
                        <input type="range" id="exp2_slider" min="100" max="50000" step="50" value="{exp2}" oninput="syncCam(2, 'exp', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="exp2_input" min="100" max="50000" step="50" value="{exp2}" oninput="syncCam(2, 'exp', 'input')" style="width:70px; margin-left:10px;">
                    </div>
                    '''
                    cam2_gain_row = f'''
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:65px; font-weight:bold;">Cam 2:</label>
                        <input type="range" id="gain2_slider" min="1.0" max="16.0" step="0.1" value="{gain2}" oninput="syncCam(2, 'gain', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="gain2_input" min="1.0" max="16.0" step="0.1" value="{gain2}" oninput="syncCam(2, 'gain', 'input')" style="width:70px; margin-left:10px;">
                    </div>
                    '''

                cam_tuning_html = f'''
                <div style="position:absolute; top:20px; right:20px; background:rgba(255,255,255,0.9); padding:15px; border-radius:8px; font-family:sans-serif; z-index:1000; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    <h3 style="margin-top:0; margin-bottom:10px;">Live Exposure (µs)</h3>
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:65px; font-weight:bold;">Cam 1:</label>
                        <input type="range" id="exp1_slider" min="100" max="50000" step="50" value="{exp1}" oninput="syncCam(1, 'exp', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="exp1_input" min="100" max="50000" step="50" value="{exp1}" oninput="syncCam(1, 'exp', 'input')" style="width:70px; margin-left:10px;">
                    </div>
                    {cam2_exp_row}

                    <h3 style="margin-top:15px; margin-bottom:10px;">Analogue Gain</h3>
                    <div style="margin-bottom: 8px;">
                        <label style="display:inline-block; width:65px; font-weight:bold;">Cam 1:</label>
                        <input type="range" id="gain1_slider" min="1.0" max="16.0" step="0.1" value="{gain1}" oninput="syncCam(1, 'gain', 'slider')" style="vertical-align: middle;">
                        <input type="number" id="gain1_input" min="1.0" max="16.0" step="0.1" value="{gain1}" oninput="syncCam(1, 'gain', 'input')" style="width:70px; margin-left:10px;">
                    </div>
                    {cam2_gain_row}

                    <script>
                        let camDebounce = {{}};
                        function syncCam(camIdx, param, source) {{
                            let slider = document.getElementById(param + camIdx + '_slider');
                            let input = document.getElementById(param + camIdx + '_input');
                            
                            if (source === 'input' && (input.value === "" || isNaN(parseFloat(input.value)))) {{
                                return;
                            }}
                            
                            if (source === 'slider') {{
                                input.value = slider.value;
                            }} else {{
                                slider.value = input.value;
                            }}
                            
                            let expInput = document.getElementById('exp' + camIdx + '_input');
                            let gainInput = document.getElementById('gain' + camIdx + '_input');
                            let expVal = parseInt(expInput.value);
                            let gainVal = parseFloat(gainInput.value);
                            if (isNaN(expVal) || isNaN(gainVal)) return;

                            clearTimeout(camDebounce[camIdx]);
                            camDebounce[camIdx] = setTimeout(() => {{
                                fetch('/update_exposure', {{
                                    method: 'POST',
                                    headers: {{'Content-Type': 'application/json'}},
                                    body: JSON.stringify({{
                                        cam_idx: camIdx,
                                        exposure: expVal,
                                        analogue_gain: gainVal
                                    }})
                                }});
                            }}, 60);
                        }}
                    </script>
                </div>
                '''

            filter_html = f'''
            <div style="position:absolute; bottom:20px; right:20px; background:rgba(255,255,255,0.9); padding:12px 15px; border-radius:8px; font-family:sans-serif; z-index:1000; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <label for="filter_select" style="font-weight:bold; margin-right:8px;">Super Secret Settings:</label>
                <select id="filter_select" onchange="updateFilter(this.value)" style="padding:5px; border-radius:4px; font-size:14px;">
                    <option value="none" {"selected" if self._filter_mode == "none" else ""}>Normal Vision</option>
                    <option value="creeper" {"selected" if self._filter_mode == "creeper" else ""}>Creeper Vision</option>
                    <option value="enderman" {"selected" if self._filter_mode == "enderman" else ""}>Enderman Vision</option>
                    <option value="spider" {"selected" if self._filter_mode == "spider" else ""}>Spider Vision</option>
                    <option value="pixelated" {"selected" if self._filter_mode == "pixelated" else ""}>Pixelated (8-Bit)</option>
                    <option value="nether" {"selected" if self._filter_mode == "nether" else ""}>Nether Vision</option>
                </select>

                <div style="margin-top: 10px;">
                    <label style="font-weight:bold; cursor:pointer;">
                        <input type="checkbox" id="pip_toggle" {"checked" if self._pip_enabled else ""} onchange="updatePiP(this.checked)" style="vertical-align:middle; margin-right:5px;">
                        Subway Surfers PiP
                    </label>
                </div>

                <script>
                    function updateFilter(mode) {{
                        fetch('/update_filter', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{ filter: mode }})
                        }});
                    }}
                    function updatePiP(enabled) {{
                        fetch('/update_pip', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{ enabled: enabled }})
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
                    {cam_tuning_html}
                    {filter_html}
                </body>
                </html>
            ''')

        @self.app.route('/update_filter', methods=['POST'])
        def update_filter():
            data = request.json or {}
            mode = data.get("filter", "none")
            valid_modes = {"none", "creeper", "enderman", "spider", "pixelated", "nether"}
            if mode in valid_modes:
                with self._lock:
                    self._filter_mode = mode
                logger.info(f"[PREVIEW] Filter changed to: {mode}")
                return {"status": "success", "filter": mode}
            return {"status": "error", "message": "Invalid filter mode"}, 400

        @self.app.route('/update_pip', methods=['POST'])
        def update_pip():
            data = request.json or {}
            enabled = bool(data.get("enabled", False))
            with self._lock:
                self._pip_enabled = enabled
            logger.info(f"[PREVIEW] Subway Surfers PiP toggled: {enabled}")
            return {"status": "success", "enabled": enabled}

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
                    with self._lock:
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

        @self.app.route('/update_exposure', methods=['POST'])
        def update_exposure():
            if not self.cam_tuning_enabled or self.capture is None:
                return {"status": "forbidden", "message": "Live camera tuning is disabled in config"}, 403

            data = request.json
            cam_idx = int(data.get('cam_idx', 1))
            exposure = data.get('exposure')
            analogue_gain = data.get('analogue_gain')

            try:
                if exposure is not None:
                    exposure = int(exposure)
                    if isinstance(self.capture, StereoCapture):
                        self.capture.set_exposure(cam_idx, exposure)
                    elif hasattr(self.capture, "set_exposure"):
                        self.capture.set_exposure(exposure)

                if analogue_gain is not None:
                    analogue_gain = float(analogue_gain)
                    if isinstance(self.capture, StereoCapture):
                        self.capture.set_analogue_gain(cam_idx, analogue_gain)
                    elif hasattr(self.capture, "set_analogue_gain"):
                        self.capture.set_analogue_gain(analogue_gain)

                logger.info(f"[CAM] Updated Cam {cam_idx} in memory: ExposureTime={exposure} us, AnalogueGain={analogue_gain}")

                if self.config_path and os.path.exists(self.config_path):
                    with self._lock:
                        with open(self.config_path, "r") as f:
                            cfg_data = json.load(f)

                        if "camera_config" not in cfg_data:
                            cfg_data["camera_config"] = {}

                        cam_key = f"cam{cam_idx}"
                        if cam_key in cfg_data["camera_config"]:
                            target_cfg = cfg_data["camera_config"][cam_key]
                        else:
                            target_cfg = cfg_data["camera_config"]

                        target_cfg["AeEnable"] = False
                        if exposure is not None:
                            target_cfg["ExposureTime"] = exposure
                        if analogue_gain is not None:
                            target_cfg["AnalogueGain"] = analogue_gain

                        with open(self.config_path, "w") as f:
                            json.dump(cfg_data, f, indent=2)

                    logger.info(f"[CAM] Saved Cam {cam_idx} settings to {self.config_path}")
            except Exception as e:
                logger.error(f"[CAM] Failed to update camera settings: {e}")
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
            filter_mode = "none"
            pip_enabled = False
            pip_frame = None
            
            with self._lock:
                if self._frame is not None:
                    frame_to_encode = self._frame.copy()
                drift_val = self._drift_ms
                filter_mode = self._filter_mode
                pip_enabled = self._pip_enabled
                if self._pip_frame is not None:
                    pip_frame = self._pip_frame.copy()
            
            if frame_to_encode is not None:
                frame_to_encode = self._apply_preview_filter(frame_to_encode, filter_mode)

                # Overlay Subway Surfers PiP in the bottom-right corner if enabled
                if pip_enabled and pip_frame is not None:
                    fh, fw = frame_to_encode.shape[:2]
                    pip_h = fh // 3  # Take 1/3 of the screen height
                    pip_w = int(pip_h * (pip_frame.shape[1] / pip_frame.shape[0]))
                    resized_pip = cv2.resize(pip_frame, (pip_w, pip_h))
                    
                    # Place in bottom-right corner with a small padding (20px)
                    y1, y2 = fh - pip_h - 20, fh - 20
                    x1, x2 = fw - pip_w - 20, fw - 20
                    if y1 >= 0 and x1 >= 0:
                        frame_to_encode[y1:y2, x1:x2] = resized_pip
                        # Draw a neat border around the PiP
                        cv2.rectangle(frame_to_encode, (x1, y1), (x2, y2), (0, 255, 255), 2)

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
        self._pip_running = False
        self._running = False
        if self._server:
            self._server.shutdown()