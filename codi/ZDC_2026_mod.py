import logging
import sys
import os
import json
import time
import math
from typing import Tuple, Optional
import cv2
import numpy as np

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
from lib.drone_lib import MavlinkConnection, FakeMavlinkConnection
from lib.camera_lib_light import FlaskPreviewServer, save_img_dir
from lib.pid_controller import PIDController

# Support root or package imports cleanly
try:
    from ZDC_vision_libs.visionOrchestrator import VisionOrchestrator, VisionPayload
except ImportError:
    from visionOrchestrator import VisionOrchestrator, VisionPayload


class ZDCMissionController:
    def __init__(self, config_path: str = "mission_config.json"):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.state_manager = StateManager(self.cfg.get("storage", {}).get("state_file", "flight_status.json"))
        self.state = FlightState()

        use_fake = self.cfg["mavlink"].get("fake_connection", False)
        conn_cls = FakeMavlinkConnection if use_fake else MavlinkConnection
        self.mav = conn_cls(
            connection_string=self.cfg["mavlink"]["connection"],
            baud=self.cfg["mavlink"]["baud"],
            debug=self.cfg["mavlink"].get("debug", False)
        )

        do_gazebo = bool(self.cfg["mavlink"].get("do_gazebo", False))

        # Vision and Target Geometry
        self.marker_size = float(self.cfg["vision"].get("marker_size_m", 0.48))
        self.target_id = int(self.cfg["vision"].get("aruco_target_id", 26))
        
        dict_name = self.cfg["vision"].get("aruco_dict", "DICT_4X4_50")
        aruco_dict_enum = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)

        cam_cfg = self.cfg.get("camera_config", {})
        cam1_cfg = cam_cfg.get("cam1", cam_cfg)
        cam2_cfg = cam_cfg.get("cam2", cam_cfg)

        cam1_calib = cam1_cfg.get("calibration_file", "./configs/1640x1232-v2.conf")
        cam2_calib = cam2_cfg.get("calibration_file", "./configs/1640x1232-v2.conf")

        mtx1, dist1 = self._load_calibration(cam1_calib)
        mtx2, dist2 = self._load_calibration(cam2_calib)

        offsets = self.cfg.get("offsets", {})
        if do_gazebo:
            # Zero out chassis offsets in Gazebo SITL if using centered sim camera
            cam0_offset = (0.0, 0.0, 0.0)
            cam1_offset = (0.0, 0.0, 0.0)
            idx_cam0 = self.cfg["vision"].get("capture_source", 0)
            idx_cam1 = 1
        else:
            cam0_offset = (
                float(offsets.get("cam1_chassis_x_dist", 0.0)),
                float(offsets.get("cam1_chassis_y_dist", 0.0)),
                float(offsets.get("cam1_chassis_z_dist", 0.0))
            )
            cam1_offset = (
                float(offsets.get("cam2_chassis_x_dist", 0.0)),
                float(offsets.get("cam2_chassis_y_dist", 0.0)),
                float(offsets.get("cam2_chassis_z_dist", 0.0))
            )
            idx_cam0 = cam1_cfg.get("source", 0)
            idx_cam1 = cam2_cfg.get("source", 1)

        resolution = tuple(self.cfg["vision"].get("capture_resolution", (640, 480)))
        fps = int(self.cfg["vision"].get("capture_fps", 30))

        # Instantiate Threaded Multi-Camera Pipeline with complete config
        logger.info(f"[INIT] Instantiating VisionOrchestrator (Gazebo SITL: {do_gazebo})...")
        self.orchestrator = VisionOrchestrator(
            cam0_idx=idx_cam0,
            cam1_idx=idx_cam1,
            marker_size_m=self.marker_size,
            cam0_matrix=mtx1,
            cam0_dist=dist1,
            cam1_matrix=mtx2,
            cam1_dist=dist2,
            target_id=self.target_id,
            aruco_dict_id=aruco_dict_enum,
            cam0_offset_m=cam0_offset,
            cam1_offset_m=cam1_offset,
            resolution=resolution,
            fps=fps,
            cam0_settings=cam1_cfg,
            cam1_settings=cam2_cfg,
            do_gazebo=do_gazebo
        )

        self.images_dir = self.cfg["vision"].get("images_dir", "captured_images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.gnss_midpoint = np.array(self.cfg["flight"].get("gnss_midpoint_ned", [0.0, 0.0, 0.0]), dtype=np.float32)

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

    @staticmethod
    def _load_calibration(config_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(config_path, "r") as f:
            data = json.load(f)
        mtx = np.array(data["mtx"], dtype=np.float32)
        dist = np.array(data["dist"], dtype=np.float32)
        return mtx, dist

    def on_boot(self):
        logger.info("==========================================")
        logger.info("       ZDC 2026 AUTONOMOUS CONTROLLER      ")
        logger.info("==========================================")
        self.state = self.state_manager.load_state()

        if self.state.current_phase == FlightPhase.MISSION_COMPLETED:
            self.state_manager.clear()
            self.state = FlightState()

        try:
            self.run_mission_loop()
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("[CLEANUP] Stopping Vision Orchestrator and hardware streams...")
        self.orchestrator.stop()

    def _check_geofence_and_override(self, allow_land: bool = False) -> bool:
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

            if phase in (FlightPhase.INIT, FlightPhase.WAITING_GUIDED):
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
        logger.info("[HANDOVER] Waiting for manual fly-in and GUIDED activation...")
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
                    logger.warning(f"[HANDOVER] Switched to GUIDED outside altitude window: {curr_alt:.2f} m")
            time.sleep(self.loop_delay)

    def _execute_search_phase(self) -> bool:
        logger.info("[SEARCH] Looking for marker...")
        self.pid_x.reset()
        self.pid_y.reset()
        last_detection_time = time.time()

        while True:
            if not self._check_geofence_and_override():
                return False

            payload, preview_frame = self.orchestrator.get_vision_data(do_draw=True)
            if preview_frame is not None:
                self.preview.update_frame(preview_frame, drift_ms=payload.drift_ms)

            if payload.target_acquired:
                logger.info(f"[SEARCH] Marker acquired ({payload.active_cameras} cam active). Switching to ALIGN.")
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

            payload, preview_frame = self.orchestrator.get_vision_data(do_draw=True)
            if preview_frame is not None:
                self.preview.update_frame(preview_frame, drift_ms=payload.drift_ms)

            if not payload.target_acquired:
                lost_frames += 1
                self.mav.move_velocity_body(0.0, 0.0, 0.0)
                if lost_frames > self.max_lost_frames:
                    logger.warning("[ALIGN] Marker lost during visual servoing.")
                    pos = self.mav.get_local_position()
                    if pos and -pos[2] <= self.blind_land_alt:
                        logger.info(f"[ALIGN] Close to ground ({-pos[2]:.2f} m). Engaging LAND mode.")
                        return True
                    return False
                time.sleep(self.loop_delay)
                continue

            lost_frames = 0

            # Forward is -y_cam, Right is +x_cam
            error_fwd = -payload.y_offset_m
            error_right = payload.x_offset_m

            v_fwd = self.pid_x.update(error_fwd)
            v_right = self.pid_y.update(error_right)

            speed = math.hypot(v_fwd, v_right)
            if speed > self.max_horizontal_speed:
                scale = self.max_horizontal_speed / speed
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
            curr_alt = -pos[2] if pos else payload.z_distance_m

            drift_str = f" | Drift: {payload.drift_ms:.1f}ms" if payload.drift_ms is not None else ""
            logger.info(
                f"[ALIGN] Alt: {curr_alt:.2f} m | Horiz Err: {horizontal_error:.2f} m | "
                f"Z-Dist: {payload.z_distance_m:.2f} m | Cams: {payload.active_cameras}{drift_str}"
            )

            if curr_alt <= self.blind_land_alt:
                logger.info(f"[ALIGN] Close to ground ({curr_alt:.2f} m). Engaging LAND mode.")
                return True

            self.mav.move_velocity_body(float(v_fwd), float(v_right), float(v_down))
            time.sleep(self.loop_delay)

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
        td_frame = self.orchestrator.get_evidence_frame()
        if td_frame is not None:
            td_path = os.path.join(self.images_dir, "touchdown.jpg")
            cv2.imwrite(td_path, td_frame)
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