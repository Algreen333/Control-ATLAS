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

from lib.status import FlightPhase, FlightState, StateManager
from lib.drone_lib import MavlinkConnection
from lib.camera_lib_light import VideoCapture, GazeboVideoCapture, FlaskPreviewServer, save_img_dir
from lib.aruco_lib import ArucoDetector
from lib.pid_controller import PIDController


class CVProcessing:
    def __init__(self, capture, config_path: str, target_id: int = 26, marker_size: float = 0.48, aruco_dict=cv2.aruco.DICT_4X4_50):
        with open(config_path, "r") as f:
            data = json.load(f)

        self.mtx = np.array(data["mtx"], dtype=np.float32)
        self.dist = np.array(data["dist"], dtype=np.float32)
        self.target_id = target_id
        self.marker_size = marker_size

        self.detector = ArucoDetector(self.mtx, self.dist, dict=aruco_dict)
        self.capture = capture

    def detect(self, do_draw: bool = False):
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        target_detections = []
        if ids is not None:
            flat_ids = np.ravel(ids)
            for i in range(len(flat_ids)):
                if flat_ids[i] == self.target_id:
                    rvec, tvec = self.detector.estimate_pose(corners[i], self.marker_size)
                    x_cam = float(tvec[0][0])
                    y_cam = float(tvec[1][0])
                    z_cam = float(tvec[2][0])
                    target_detections.append((x_cam, y_cam, z_cam))

            if do_draw:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        return frame, target_detections


class ZDCMissionController:
    def __init__(self, config_path: str = "mission_config.json"):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.state_manager = StateManager(self.cfg.get("storage", {}).get("state_file", "flight_status.json"))
        self.state = FlightState()

        self.mav = MavlinkConnection(
            connection_string=self.cfg["mavlink"]["connection"],
            baud=self.cfg["mavlink"]["baud"],
            debug=self.cfg["mavlink"].get("debug", False)
        )

        res = tuple(self.cfg["vision"]["capture_resolution"])
        fps = self.cfg["vision"]["capture_fps"]
        src = self.cfg["vision"]["capture_source"]

        if self.cfg["mavlink"].get("do_gazebo", False):
            self.capture = GazeboVideoCapture(capture_source=src, resolution=res, fps=fps)
        else:
            cam_cfg = self.cfg.get("camera_config", {})
            self.capture = VideoCapture(
                capture_source=src,
                resolution=res,
                fps=fps,
                AeEnable=cam_cfg.get("AeEnable", False),
                ExposureTime=cam_cfg.get("ExposureTime", 50000),
                AnalogueGain=cam_cfg.get("AnalogueGain", 2.0),
                Brightness=cam_cfg.get("Brightness", 0.1)
            )

        self.target_id = self.cfg["vision"].get("aruco_target_id", 26)
        self.marker_size = self.cfg["vision"].get("marker_size_m", 0.48)
        self.cvproc = CVProcessing(
            capture=self.capture,
            config_path=self.cfg["vision"]["camera_config_path"],
            target_id=self.target_id,
            marker_size=self.marker_size,
            aruco_dict=cv2.aruco.DICT_4X4_50
        )

        self.images_dir = self.cfg["vision"].get("images_dir", "captured_images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.gnss_midpoint = np.array(self.cfg["flight"].get("gnss_midpoint_ned", [0.0, 0.0, 0.0]))

        # --- Extracted Autonomy Config Variables ---
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
        # -------------------------------------------

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

            frame, targets = self.cvproc.detect(do_draw=True)
            if frame is not None:
                self.preview.update_frame(frame)

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

            frame, targets = self.cvproc.detect(do_draw=True)
            if frame is not None:
                self.preview.update_frame(frame)

            if len(targets) == 0:
                lost_frames += 1
                self.mav.move_velocity_body(0.0, 0.0, 0.0)
                if lost_frames > self.max_lost_frames:
                    logger.warning("[ALIGN] Marker lost during visual servoing. Hovering.")
                    pos = self.mav.get_local_position()
                    if pos and -pos[2] <= self.blind_land_alt:
                        logger.info(f"[ALIGN] Close to ground ({-pos[2]:.2f} m <= {self.blind_land_alt} m). Switching to physical touchdown (LAND mode).")
                        return True
                    else: return False
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
        ret, td_frame = self.capture.read()
        if ret and td_frame is not None:
            td_path = os.path.join(self.images_dir, "touchdown.jpg")
            cv2.imwrite(td_path, td_frame)
            save_img_dir(td_frame, self.images_dir)
            logger.info(f"[TOUCHDOWN] Evidence saved to {td_path}")

        logger.info("[TOUCHDOWN] Holding stable on ground for >= 1 second...")
        time.sleep(1.2)

        logger.info(f"[ASCENT] Climbing back to {self.climb_target_alt} m AGL directly above marker...")
        self.mav.arm_and_takeoff(self.climb_target_alt)

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
    config_file = os.getenv("MISSION_CONFIG_FILE", ".mission_mac.json")
    controller = ZDCMissionController(config_path=config_file)
    controller.on_boot()