import logging
import sys
import os
import json
import time
import math
import numpy as np
from typing import List, Tuple, Optional
import cv2

log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("drone_mission.log")
    ]
)
logger = logging.getLogger("MissionCtrl")

# Módulos del sistema de visión nuevo
from camera import Camera
from aruco import Aruco
from visionOrchestrator import VisionOrchestrator, VisionPayload

# Módulos de control y simulación
from lib.status import FlightPhase, FlightState, StateManager
from lib.drone_lib import MavlinkConnection, FakeMavlinkConnection
from lib.pid_controller import PIDController

try:
    from lib.camera_lib_light import VideoCapture, GazeboVideoCapture, FlaskPreviewServer, save_img_dir, StereoCapture
except ImportError:
    VideoCapture = None
    GazeboVideoCapture = None
    FlaskPreviewServer = None
    save_img_dir = None
    StereoCapture = None


class CVProcessing:
    """Procesamiento monocular adaptado a aruco.py y camera.py."""
    def __init__(self, capture, config_path: str, target_id: int = 26, marker_size: float = 0.48, aruco_dict=cv2.aruco.DICT_4X4_50):
        with open(config_path, "r") as f:
            data = json.load(f)

        self.mtx = np.array(data["mtx"], dtype=np.float32)
        self.dist = np.array(data["dist"], dtype=np.float32)
        self.target_id = target_id
        self.marker_size = marker_size

        self.aruco = Aruco(
            marker_size_m=self.marker_size,
            camera_matrix=self.mtx,
            dist_coeffs=self.dist,
            dict_id=aruco_dict
        )
        self.capture = capture

    def detect(self, do_draw: bool = False):
        if hasattr(self.capture, "get_frame"):
            ret, frame = self.capture.get_frame()
        elif hasattr(self.capture, "read"):
            ret, frame = self.capture.read()
        else:
            return None, [], None

        if not ret or frame is None:
            return None, [], None

        disp_frame = frame.copy() if do_draw else frame

        success, x_m, y_m, z_m = self.aruco.process_frame(frame)
        target_detections = []
        if success:
            target_detections.append((x_m, y_m, z_m))

        if do_draw:
            gray = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.aruco.detector.detectMarkers(gray)
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(disp_frame, corners, ids)
            return disp_frame, target_detections, None

        return None, target_detections, None

    def stop(self):
        if hasattr(self.capture, "stop"):
            self.capture.stop()


cvVideoProcessing = CVProcessing


class CVStereoProcessing:
    """Procesamiento estéreo adaptado a VisionOrchestrator, Camera y Aruco."""
    def __init__(
        self,
        capture=None,
        cam1_config_path: str = "./configs/1640x1232-v2.conf",
        cam2_config_path: str = "./configs/1640x1232-v2.conf",
        offsets: dict = None,
        target_id: int = 26,
        marker_size: float = 0.48,
        aruco_dict=cv2.aruco.DICT_4X4_50,
        cam0_idx: int = 0,
        cam1_idx: int = 1
    ):
        self.target_id = target_id
        self.marker_size = marker_size
        self.offsets = offsets or {}

        if isinstance(capture, VisionOrchestrator):
            self.orchestrator = capture
        else:
            with open(cam1_config_path, "r") as f:
                data1 = json.load(f)
            self.mtx1 = np.array(data1["mtx"], dtype=np.float32)
            self.dist1 = np.array(data1["dist"], dtype=np.float32)

            with open(cam2_config_path, "r") as f:
                data2 = json.load(f)
            self.mtx2 = np.array(data2["mtx"], dtype=np.float32)
            self.dist2 = np.array(data2["dist"], dtype=np.float32)

            cam0_offset = (
                float(self.offsets.get("cam1_chassis_x_dist", 0.0)),
                float(self.offsets.get("cam1_chassis_y_dist", 0.0)),
                float(self.offsets.get("cam1_chassis_z_dist", 0.0))
            )
            cam1_offset = (
                float(self.offsets.get("cam2_chassis_x_dist", 0.0)),
                float(self.offsets.get("cam2_chassis_y_dist", 0.0)),
                float(self.offsets.get("cam2_chassis_z_dist", 0.0))
            )

            # Inicialización Orchestrator
            self.orchestrator = VisionOrchestrator(
                cam0_idx=cam0_idx,
                cam1_idx=cam1_idx,
                marker_size_m=self.marker_size,
                cam0_matrix=self.mtx1,
                cam0_dist=self.dist1,
                cam1_matrix=self.mtx2,
                cam1_dist=self.dist2,
                cam0_offset_m=cam0_offset,
                cam1_offset_m=cam1_offset
            )

        if aruco_dict is not None and hasattr(self.orchestrator, "pipe0"):
            self.orchestrator.pipe0 = Aruco(self.marker_size, self.orchestrator.pipe0.camera_matrix, self.orchestrator.pipe0.dist_coeffs, dict_id=aruco_dict)
            self.orchestrator.pipe1 = Aruco(self.marker_size, self.orchestrator.pipe1.camera_matrix, self.orchestrator.pipe1.dist_coeffs, dict_id=aruco_dict)

        self.capture = self.orchestrator

    def detect(self, do_draw: bool = False):
        payload: VisionPayload = self.orchestrator.get_vision_data()

        target_detections = []
        if payload.target_acquired:
            target_detections.append((payload.x_offset_m, payload.y_offset_m, payload.z_distance_m))

        drft = 0.0
        try:
            t0 = getattr(self.orchestrator.cam0, "last_frame_time", 0.0)
            t1 = getattr(self.orchestrator.cam1, "last_frame_time", 0.0)
            if t0 > 0 and t1 > 0:
                drft = abs(t0 - t1) * 1000.0
        except Exception:
            drft = 0.0

        disp_frame = None
        if do_draw:
            h0, frame0 = self.orchestrator.cam0.get_frame()
            h1, frame1 = self.orchestrator.cam1.get_frame()

            f0 = frame0.copy() if (h0 and frame0 is not None) else None
            f1 = frame1.copy() if (h1 and frame1 is not None) else None

            if f0 is not None:
                gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                corners0, ids0, _ = self.orchestrator.pipe0.detector.detectMarkers(gray0)
                if ids0 is not None and len(corners0) > 0:
                    cv2.aruco.drawDetectedMarkers(f0, corners0, ids0)

            if f1 is not None:
                gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                corners1, ids1, _ = self.orchestrator.pipe1.detector.detectMarkers(gray1)
                if ids1 is not None and len(corners1) > 0:
                    cv2.aruco.drawDetectedMarkers(f1, corners1, ids1)

            if f0 is not None and f1 is not None:
                disp_frame = cv2.hconcat([f0, f1])
            elif f0 is not None:
                disp_frame = f0
            elif f1 is not None:
                disp_frame = f1

        return disp_frame, target_detections, drft

    def read(self):
        """Compatibilidad con interfaces de lectura estilo tupla."""
        h0, f0 = self.orchestrator.cam0.get_frame()
        h1, f1 = self.orchestrator.cam1.get_frame()
        t0 = getattr(self.orchestrator.cam0, "last_frame_time", 0.0)
        t1 = getattr(self.orchestrator.cam1, "last_frame_time", 0.0)
        drift = abs(t0 - t1) * 1000.0 if (t0 > 0 and t1 > 0) else 0.0
        return (h0, f0), (h1, f1), drift

    def get_frame(self):
        """Retorna el frame de la cámara principal (cam0)."""
        return self.orchestrator.cam0.get_frame()

    def stop(self):
        """Detiene de forma segura los hilos de captura de ambas cámaras."""
        if self.orchestrator is not None:
            self.orchestrator.stop()


class ZDCMissionController:
    def __init__(self, config_path: str = "mission_config.json"):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.state_manager = StateManager(self.cfg.get("storage", {}).get("state_file", "flight_status.json"))
        self.state = FlightState()

        use_fake = self.cfg["mavlink"].get("fake_connection", False)
        if use_fake:
            self.mav = FakeMavlinkConnection(
                connection_string=self.cfg["mavlink"]["connection"],
                baud=self.cfg["mavlink"]["baud"],
                debug=self.cfg["mavlink"].get("debug", False)
            )
        else:
            self.mav = MavlinkConnection(
                connection_string=self.cfg["mavlink"]["connection"],
                baud=self.cfg["mavlink"]["baud"],
                debug=self.cfg["mavlink"].get("debug", False)
            )

        res = tuple(self.cfg["vision"]["capture_resolution"])
        fps = self.cfg["vision"]["capture_fps"]
        src = self.cfg["vision"]["capture_source"]
        
        cam_cfg = self.cfg.get("camera_config", {})
        cam1_cfg = cam_cfg.get("cam1", cam_cfg) 
        cam2_cfg = cam_cfg.get("cam2", cam_cfg)
        
        cam1_calib = cam1_cfg.get("calibration_file", "./configs/1640x1232-v2.conf")
        cam2_calib = cam2_cfg.get("calibration_file", "./configs/1640x1232-v2.conf")
        
        offsets = self.cfg.get("offsets", {})

        self.target_id = self.cfg["vision"].get("aruco_target_id", 26)
        self.marker_size = self.cfg["vision"].get("marker_size_m", 0.48)
        
        aruco_dict_name = self.cfg["vision"].get("aruco_dict", "DICT_4X4_50")
        aruco_dict_enum = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)

        if self.cfg["mavlink"].get("do_gazebo", False):
            self.capture = GazeboVideoCapture(capture_source=src, resolution=res, fps=fps) if GazeboVideoCapture else None
            self.cvproc = CVProcessing(
                capture=self.capture,
                config_path=cam1_calib,
                target_id=self.target_id,
                marker_size=self.marker_size,
                aruco_dict=aruco_dict_enum
            )
        else:
            try:
                logger.info("Inicializando pipeline estéreo con VisionOrchestrator...")
                self.cvproc = CVStereoProcessing(
                    cam1_config_path=cam1_calib,
                    cam2_config_path=cam2_calib,
                    offsets=offsets,
                    target_id=self.target_id,
                    marker_size=self.marker_size,
                    aruco_dict=aruco_dict_enum,
                    cam0_idx=0,
                    cam1_idx=1
                )
                self.capture = self.cvproc.orchestrator
                logger.info("VisionOrchestrator inicializado exitosamente.")
            except Exception as e:
                logger.warning(f"Fallo al inicializar VisionOrchestrator: {e}. Activando fallback a Camera monocular.")
                
                if hasattr(self, 'cvproc') and hasattr(self.cvproc, 'stop'):
                    try:
                        self.cvproc.stop()
                    except Exception:
                        pass
                
                self.capture = Camera(
                    source=src,
                    resolution=res,
                    fps=fps,
                    AeEnable=cam1_cfg.get("AeEnable", False),
                    ExposureTime=cam1_cfg.get("ExposureTime", 50000),
                    AnalogueGain=cam1_cfg.get("AnalogueGain", 2.0),
                    Brightness=cam1_cfg.get("Brightness", 0.1)
                )
                self.cvproc = CVProcessing(
                    capture=self.capture,
                    config_path=cam1_calib,
                    target_id=self.target_id,
                    marker_size=self.marker_size,
                    aruco_dict=aruco_dict_enum
                )

        self.images_dir = self.cfg["vision"].get("images_dir", "captured_images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.gnss_midpoint = np.array(self.cfg["flight"].get("gnss_midpoint_ned", [0.0, 0.0, 0.0]))

        auto_cfg = self.cfg.get("autonomy", {})
        self.max_horizontal_speed = auto_cfg.get("max_horizontal_speed_ms", 1.0)
        self.geofence_radius = auto_cfg.get("geofence_radius_m", 10.0)
        self.altitude_ceiling = auto_cfg.get("altitude_ceiling_m", 30.0)
        self.min_activation_alt = auto_cfg.get("min_activation_alt_m", 5.0)
        self.max_activation_alt = auto_cfg.get("max_activation_alt_m", 20.0)
        self.climb_target_alt = auto_cfg.get("climb_target_alt_m", 5.0)
        self.blind_land_alt = auto_cfg.get("blind_land_alt_m", 0.6)
        
        self.max_lost_frames = auto_cfg.get("max_lost_frames", 15)
        self.loop_delay = auto_cfg.get("loop_delay_s", 0.05)
        
        self.descend_radius_inner = auto_cfg.get("descend_radius_inner_m", 0.4)
        self.descend_speed_fast = auto_cfg.get("descend_speed_fast_ms", 0.35)
        self.descend_radius_outer = auto_cfg.get("descend_radius_outer_m", 0.8)
        self.descend_speed_slow = auto_cfg.get("descend_speed_slow_ms", 0.20)

        pid_cfg = self.cfg.get("pid", {})
        self.pid_x = PIDController(
            kp=pid_cfg.get("kp", 0.7),
            ki=pid_cfg.get("ki", 0.02),
            kd=pid_cfg.get("kd", 0.15),
            output_min=-1.0,
            output_max=1.0
        )
        self.pid_y = PIDController(
            kp=pid_cfg.get("kp", 0.7),
            ki=pid_cfg.get("ki", 0.02),
            kd=pid_cfg.get("kd", 0.15),
            output_min=-1.0,
            output_max=1.0
        )

        self.preview = None
        if FlaskPreviewServer:
            self.preview = FlaskPreviewServer(
                host="0.0.0.0",
                port=self.cfg["vision"].get("preview_port", 5000),
                max_fps=15,
                jpeg_quality=55,
                tuning_enabled=pid_cfg.get("live_tuning_enabled", False),
                pid_controllers=[self.pid_x, self.pid_y],
                config_path=config_path
            )
            self.preview.start()

    def on_boot(self):
        logger.info("==========================================")
        logger.info("       ZDC 2026 AUTONOMOUS CONTROLLER      ")
        logger.info("==========================================")
        self.state = self.state_manager.load_state()

        if self.state.current_phase == FlightPhase.MISSION_COMPLETED:
            self.state_manager.clear()
            self.state = FlightState()

        self.run_mission_loop()

    def _check_geofence_and_override(self, allow_land=False) -> bool:
        if not (self.mav.isGuided() or (allow_land and (self.mav.mav.flightmode in ["LANDING", "LAND"]))):
            logger.warning("[SAFETY] Manual pilot override detected. Aborting autonomous control.")
            return False

        pos = self.mav.get_local_position()
        if pos:
            horizontal_dist = np.linalg.norm(np.array(pos[:2]) - self.gnss_midpoint[:2])
            alt = -pos[2]

            if horizontal_dist > self.geofence_radius:
                logger.error(f"[SAFETY] Geofence radius breached ({horizontal_dist:.2f} m > {self.geofence_radius} m). Triggering RTH.")
                self.mav.mav.set_mode("RTL")
                return False

            if alt > self.altitude_ceiling:
                logger.error(f"[SAFETY] Altitude ceiling breached ({alt:.2f} m > {self.altitude_ceiling} m). Triggering RTH.")
                self.mav.mav.set_mode("RTL")
                return False

        return True

    def run_mission_loop(self):
        while self.state.current_phase != FlightPhase.MISSION_COMPLETED:
            phase = self.state.current_phase

            if phase == FlightPhase.INIT or phase == FlightPhase.WAITING_GUIDED:
                self._wait_for_autonomous_activation()

            elif phase == FlightPhase.SEARCHING:
                success = self._execute_search_phase()
                if success:
                    self.state.current_phase = FlightPhase.ALIGNING
                    self.state_manager.save_state(self.state)

            elif phase == FlightPhase.ALIGNING:
                success = self._execute_visual_servoing_landing()
                if success:
                    self.state.current_phase = FlightPhase.LANDING
                    self.state_manager.save_state(self.state)

            elif phase == FlightPhase.LANDING:
                success = self._execute_touchdown_and_ascent()
                if success:
                    self.state.current_phase = FlightPhase.MISSION_COMPLETED
                    self.state_manager.save_state(self.state)
                else:
                    break

        self.mav.move_velocity_body(0.0, 0.0, 0.0)
        self.state_manager.clear()
        if hasattr(self.cvproc, "stop"):
            self.cvproc.stop()
        logger.info("[ZDC] Autonomous sequence finished.")

    def _wait_for_autonomous_activation(self):
        logger.info("[HANDOVER] Waiting for manual fly-in and GUIDED activation in autonomous handover zone...")
        while True:
            if self.mav.isGuided():
                pos = self.mav.get_local_position()
                curr_alt = -pos[2] if pos else 0.0
                if self.min_activation_alt <= curr_alt <= self.max_activation_alt:
                    logger.info(f"[HANDOVER] Autonomous mode engaged at {curr_alt:.2f} m AGL.")
                    self.state.current_phase = FlightPhase.SEARCHING
                    self.state_manager.save_state(self.state)
                    break
                else:
                    logger.warning(f"[HANDOVER] Switched to GUIDED outside altitude window (5-20m): {curr_alt:.2f} m")
            time.sleep(self.loop_delay)

    def _execute_search_phase(self) -> bool:
        logger.info("[SEARCH] Looking for marker...")
        self.pid_x.reset()
        self.pid_y.reset()
        last_detection_time = time.time()

        while True:
            if not self._check_geofence_and_override():
                return False

            frame, targets, drift = self.cvproc.detect(do_draw=True)
            if frame is not None and self.preview:
                self.preview.update_frame(frame, drift_ms=drift)

            if len(targets) > 0:
                logger.info("[SEARCH] Marker acquired. Switching to ALIGNING.")
                return True

            self.mav.move_velocity_body(0.0, 0.0, 0.0)

            if time.time() - last_detection_time > self.cfg.get("autonomy", {}).get("marker_search_timeout", 15.0):
                logger.warning("[SEARCH] Timeout searching for marker. Hovering and aborting.")
                self.mav.move_velocity_body(0.0, 0.0, 0.0)
                self.mav.mav.set_mode("RTL")
                return False

            time.sleep(self.loop_delay)

    def _execute_visual_servoing_landing(self) -> bool:
        logger.info("[ALIGN] Executing relative visual descent...")
        self.pid_x.reset()
        self.pid_y.reset()

        lost_frames = 0

        while True:
            if not self._check_geofence_and_override():
                return False

            frame, targets, drift = self.cvproc.detect(do_draw=True)
            if frame is not None and self.preview:
                self.preview.update_frame(frame, drift_ms=drift)

            if len(targets) == 0:
                lost_frames += 1
                self.mav.move_velocity_body(0.0, 0.0, 0.0)
                if lost_frames > self.max_lost_frames:
                    logger.warning("[ALIGN] Marker lost during visual servoing. Hovering.")
                    pos = self.mav.get_local_position()
                    if pos and -pos[2] <= self.blind_land_alt:
                        logger.info(f"[ALIGN] Close to ground ({-pos[2]:.2f} m <= {self.blind_land_alt} m). Switching to physical touchdown (LAND mode).")
                        return True
                    else:
                        return False
                time.sleep(self.loop_delay)
                continue

            lost_frames = 0
            cam_x, cam_y, cam_z = targets[0]

            error_fwd = -cam_y
            error_right = cam_x

            v_fwd = self.pid_x.update(error_fwd)
            v_right = self.pid_y.update(error_right)

            current_horizontal_speed = math.hypot(v_fwd, v_right)
            if current_horizontal_speed > self.max_horizontal_speed:
                scale = self.max_horizontal_speed / current_horizontal_speed
                v_fwd *= scale
                v_right *= scale

            horizontal_error = math.hypot(error_fwd, error_right)

            if horizontal_error < self.descend_radius_inner:
                v_down = self.descend_speed_fast
            elif horizontal_error < self.descend_radius_outer:
                v_down = self.descend_speed_slow
            else:
                v_down = 0.0

            pos = self.mav.get_local_position()
            curr_alt = -pos[2] if pos else cam_z

            logger.info(f"[ALIGN] Current height: {curr_alt:.2f} m, Distance to target: {horizontal_error:.2f} m")

            if curr_alt <= self.blind_land_alt:
                logger.info(f"[ALIGN] Close to ground ({curr_alt:.2f} m <= {self.blind_land_alt} m). Switching to physical touchdown (LAND mode).")
                return True

            self.mav.move_velocity_body(float(v_fwd), float(v_right), float(v_down))

    def _execute_touchdown_and_ascent(self) -> bool:
        logger.info("[TOUCHDOWN] Final descent to contact...")
        self.mav.switch_to_land()

        touchdown_detected = False
        start_wait = time.time()
        while time.time() - start_wait < 15.0:
            if not self._check_geofence_and_override(allow_land=True):
                return False

            pos = self.mav.get_local_position()
            curr_alt = -pos[2] if pos else 0.0

            if not self.mav.is_airborne(alt_threshold_m=0.15) or curr_alt <= 0.12:
                touchdown_detected = True
                break
            time.sleep(self.loop_delay)

        if not touchdown_detected:
            logger.warning("[TOUCHDOWN] Ground contact not confirmed within timeout.")
            return False

        logger.info("[TOUCHDOWN] Touchdown confirmed. Capturing mandatory touchdown evidence...")
        ret, td_frame = False, None
        
        if hasattr(self.cvproc, "orchestrator") and self.cvproc.orchestrator is not None:
            ret, td_frame = self.cvproc.orchestrator.cam0.get_frame()
        elif hasattr(self.cvproc, "get_frame"):
            ret, td_frame = self.cvproc.get_frame()
        elif hasattr(self.capture, "get_frame"):
            ret, td_frame = self.capture.get_frame()
        elif hasattr(self.capture, "read"):
            out = self.capture.read()
            if isinstance(out, tuple) and len(out) == 3:
                ret, td_frame = out[0]
            elif isinstance(out, tuple) and len(out) == 2:
                ret, td_frame = out

        if ret and td_frame is not None:
            td_path = os.path.join(self.images_dir, "touchdown.jpg")
            cv2.imwrite(td_path, td_frame)
            if save_img_dir:
                save_img_dir(td_frame, self.images_dir)
            logger.info(f"[TOUCHDOWN] Evidence saved to {td_path}")

        logger.info("[TOUCHDOWN] Holding stable on ground for >= 1 second...")
        time.sleep(1.2)

        logger.info(f"[ASCENT] Climbing back to {self.climb_target_alt} m AGL directly above marker...")
        if hasattr(self.mav, 'arm_and_takeoff'):
            self.mav.arm_and_takeoff(self.climb_target_alt)
        else:
            self.mav.setGuided()
            time.sleep(0.2)
            self.mav.takeoff(self.climb_target_alt)

        climb_start = time.time()
        while time.time() - climb_start < 10.0:
            pos = self.mav.get_local_position()
            curr_alt = -pos[2] if pos else 0.0
            if curr_alt >= self.climb_target_alt * 0.95:
                logger.info(f"[ASCENT] Reached hover altitude: {curr_alt:.2f} m AGL.")
                break
            time.sleep(0.1)

        self.mav.move_velocity_body(0.0, 0.0, 0.0)
        logger.info("[HANDBACK] Autonomous sequence completed. Handing control back to pilot.")
        return True


if __name__ == "__main__":
    config_file = os.getenv("MISSION_CONFIG_FILE", "mission_config.json")
    controller = ZDCMissionController(config_path=config_file)
    controller.on_boot()