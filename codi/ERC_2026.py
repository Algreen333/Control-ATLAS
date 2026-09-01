import logging
import sys
import os
import json
import time
import math
import numpy as np
from typing import List, Tuple

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
import cv2


class CVProcessing:
    def __init__(self, capture, config_path: str, home_id: int, land_id: int, aruco_dict=cv2.aruco.DICT_ARUCO_ORIGINAL):
        with open(config_path, "r") as f:
            data = json.load(f)

        self.mtx = np.array(data["mtx"])
        self.dist = np.array(data["dist"])
        self.aruco_dict = aruco_dict
        self.home_id = home_id
        self.land_id = land_id

        self.detector = ArucoDetector(self.mtx, self.dist, self.aruco_dict)
        self.capture = capture

    def detect(self, do_draw=False):
        ret, frame = self.capture.read()
        if not ret:
            logger.warning("Capture read returned error")
            return None, [], []

        prediction = self.detector.full_prediction(frame, do_draw)
        if prediction is None:
            return frame, [], []
            
        frame, detections = prediction
        home_detections = []
        land_detections = []

        for det in detections:
            pos, rot, marker_id = det
            if marker_id == self.home_id:
                home_detections.append(pos)
            elif marker_id == self.land_id:
                land_detections.append(pos)
            logger.info(f"Detected ArUCO {marker_id} with offset {pos}.")

        return frame, home_detections, land_detections


class ERCMissionController:
    def __init__(self, config_path: str = "mission_config.json"):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        # Storage & State
        self.state_manager = StateManager(self.cfg["storage"]["state_file"])
        self.state = FlightState()
        self.state.search_altitude_m = self.cfg["flight"]["search_altitude_m"]

        # Hardware & Communication
        self.mav = MavlinkConnection(
            connection_string=self.cfg["mavlink"]["connection"],
            baud=self.cfg["mavlink"]["baud"],
            debug=self.cfg["mavlink"]["debug"]
        )

        # Video Input & Processing
        res = tuple(self.cfg["vision"]["capture_resolution"])
        fps = self.cfg["vision"]["capture_fps"]
        src = self.cfg["vision"]["capture_source"]

        if self.cfg["mavlink"]["do_gazebo"]:
            self.capture = GazeboVideoCapture(capture_source=src, resolution=res, fps=fps)
        else:
            self.capture = VideoCapture(capture_source=src, resolution=res, fps=fps)

        self.cvproc = CVProcessing(
            capture=self.capture,
            config_path=self.cfg["vision"]["camera_config_path"],
            home_id=self.cfg["vision"]["aruco_home_id"],
            land_id=self.cfg["vision"]["aruco_land_id"]
        )

        # Paths and Directories
        self.search_path: List[Tuple[float, float, float]] = [
            tuple(wp) for wp in self.cfg["flight"]["search_path"]
        ]
        self.images_dir = self.cfg["vision"]["images_dir"]
        os.makedirs(self.images_dir, exist_ok=True)
        self.last_image_time = -1

        # Web Preview Server
        self.preview = FlaskPreviewServer(
            host="0.0.0.0", 
            port=self.cfg["vision"]["preview_port"], 
            max_fps=15, 
            jpeg_quality=55
        )
        self.preview.start()

    def on_boot(self):
        logger.info("==========================================")
        logger.info("       ERC DRONE MISSION BOOTLOADER       ")
        logger.info("==========================================")

        self.state = self.state_manager.load_state()
        airborne = self.mav.is_airborne(alt_threshold_m=0.4)

        if airborne:
            logger.warning("REBOOT DETECTED IN FLIGHT! Resuming active leg...")
            self._handle_airborne_recovery()
        else:
            logger.info("Drone is grounded. Normal startup sequence.")
            self._handle_grounded_boot()

        self.run_mission_loop()

    def _handle_airborne_recovery(self):
        logger.info("Switching FC back to GUIDED mode to retake control...")
        self.mav.setGuided()
        time.sleep(1)

        pos = self.mav.get_local_position()
        if pos:
            self.mav.send_target_ned(pos[0], pos[1], pos[2], speed_ms=self.cfg["flight"]["search_speed_ms"])
            logger.info(f"Hover hold at NED: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        logger.info(f"Resuming phase: {self.state.current_phase.value}")

    def _handle_grounded_boot(self):
        phase = self.state.current_phase

        if phase == FlightPhase.INIT:
            logger.info("Ready for launch. Waiting for launch signal...")
        elif phase == FlightPhase.WAITING_MANUAL_RESET:
            logger.info("Awaiting manual reset...")
        elif phase in [FlightPhase.SEARCHING, FlightPhase.ALIGNING, FlightPhase.LANDING]:
            logger.warning(f"Drone grounded but state was {phase.value}. Resetting phase to INIT...")
            self.state.current_phase = FlightPhase.INIT
            self.state.landing_coords.clear()
            self.state_manager.save_state(self.state)
        elif phase == FlightPhase.MISSION_COMPLETED:
            logger.info("Mission was already completed. Resetting state.")
            self.state_manager.clear()
            self.state = FlightState()

    def _perform_takeoff_sequence(self):
        """Executes arming and mode selection based on config flags[cite: 1]."""
        # GUIDED Mode Handling
        if self.cfg["flight"].get("auto_guided", True):
            logger.info("[MODE] Auto-guided enabled. Switching to GUIDED...")
            self.mav.setGuided()
        else:
            logger.info("[MODE] Auto-guided disabled. Waiting for external GUIDED switch...")
            self.mav.waitGuided()

        # ARMING Handling
        if self.cfg["flight"].get("auto_arm", True):
            logger.info("[ARM] Auto-arm enabled. Arming motors...")
            self.mav.arm()
        else:
            logger.info("[ARM] Auto-arm disabled. Waiting for external ARM command...")
            self.mav.waitArmed()

        # Takeoff Command
        self.mav.set_attitude_speed(self.cfg["flight"]["search_speed_ms"])
        self.mav.takeoff(self.state.search_altitude_m)
        time.sleep(2)
        self.mav.set_attitude_speed(self.cfg["flight"]["search_speed_ms"])

    def run_mission_loop(self):
        while self.state.current_phase != FlightPhase.MISSION_COMPLETED:
            phase = self.state.current_phase

            if phase == FlightPhase.INIT:
                logger.info("--- Starting Mission Takeoff ---")
                self._perform_takeoff_sequence()
                self.state.current_phase = FlightPhase.SEARCHING
                self.state_manager.save_state(self.state)

            elif phase == FlightPhase.SEARCHING:
                if self._execute_search_step():
                    self.state.current_phase = FlightPhase.ALIGNING
                    self.state_manager.save_state(self.state)

            elif phase == FlightPhase.ALIGNING:
                if self._execute_align_step():
                    logger.info("Approach finalised, switching to LAND...")
                    self.state.current_phase = FlightPhase.LANDING
                    self.state_manager.save_state(self.state)

            elif phase == FlightPhase.LANDING:
                self.mav.switch_to_land()
                while self.mav.is_airborne(alt_threshold_m=0.2):
                    time.sleep(1)
                self.state.current_phase = FlightPhase.MISSION_COMPLETED

        self.state_manager.clear()

    def _execute_search_step(self):
        pos = self.mav.get_local_position()
        if pos is None:
            logger.warning("Could not retrieve local position!")
            return False
        pos = np.array(pos)

        att_msg = self.mav.mav.messages.get('ATTITUDE')
        yaw = att_msg.yaw if att_msg else 0.0

        # Frame Processing
        res = self.cvproc.detect()
        try:
            frame, homes, lands = res
            self.preview.update_frame(frame)
        except Exception as e:
            logger.error(e)
            return False

        if time.time() > self.last_image_time + self.cfg["vision"]["images_delay"] or self.last_image_time == -1:
            save_img_dir(frame, self.images_dir)
            self.last_image_time = time.time()

        if len(lands) > 0:
            meanlands = np.mean(np.array(lands), axis=0)
            logger.info(f"Landing ArUco detected at camera offset: {meanlands}")

            cam_x, cam_y, cam_z = meanlands
            body_forward = -cam_y
            body_right = cam_x

            north_offset = (body_forward * math.cos(yaw)) - (body_right * math.sin(yaw))
            east_offset = (body_forward * math.sin(yaw)) + (body_right * math.cos(yaw))

            target_n = pos[0] + north_offset
            target_e = pos[1] + east_offset
            target_d = pos[2]

            self.register_landing_target((target_n, target_e, target_d))

        speed = self.cfg["flight"]["search_speed_ms"]
        max_dist = self.cfg["flight"]["search_max_dist_to_wp"]

        if self.state.search_wp_idx == -1:
            logger.info("Moving to first waypoint")
            if self.state.search_wp_idx < len(self.search_path) - 1:
                self.state.search_wp_idx += 1
                wp = self.search_path[self.state.search_wp_idx]
                self.mav.send_target_ned(wp[0], wp[1], -self.state.search_altitude_m, speed_ms=speed)
                self.state_manager.save_state(self.state)
                return False
            return True

        target_pos = np.array(self.search_path[self.state.search_wp_idx])
        distance = np.linalg.norm(target_pos[:2] - pos[:2])

        if distance < max_dist:
            logger.info(f"Reached search waypoint {self.state.search_wp_idx}")
            if self.state.search_wp_idx < len(self.search_path) - 1:
                self.state.search_wp_idx += 1
                wp = self.search_path[self.state.search_wp_idx]
                self.mav.send_target_ned(wp[0], wp[1], -self.state.search_altitude_m, speed_ms=speed)
                self.state_manager.save_state(self.state)
                return False
            return True
        else:
            wp = self.search_path[self.state.search_wp_idx]
            self.mav.send_target_ned(wp[0], wp[1], -self.state.search_altitude_m, speed_ms=speed)
            return False

    def register_landing_target(self, coords):
        self.state.landing_coords.append(coords)
        if len(self.state.landing_coords) > self.cfg["alignment"]["window_size"]:
            self.state.landing_coords.pop(0)
        self.state_manager.save_state(self.state)

    def _execute_align_step(self):
        self._execute_search_step()
        
        if len(self.state.landing_coords) == 0:
            lnd_crds = (0.0, 0.0, 0.0)
        else:
            coords = np.array(self.state.landing_coords)
            median = np.median(coords, axis=0)
            mad = np.median(np.abs(coords - median), axis=0)
            mad = np.where(mad == 0, 1e-6, mad)

            deviations = np.abs(coords - median) / mad
            clean_coords = coords[np.all(deviations < 2.5, axis=1)]
            lnd_crds = np.mean(clean_coords, axis=0) if len(clean_coords) > 0 else median

        logger.info(f"Centering over landing target: ({lnd_crds[0]:.2f}, {lnd_crds[1]:.2f})")

        speed = self.cfg["flight"]["search_speed_ms"]
        for _ in range(self.cfg["alignment"]["align_iters"]):
            self.mav.send_target_ned(lnd_crds[0], lnd_crds[1], -self.state.search_altitude_m, speed_ms=speed)
            time.sleep(self.cfg["alignment"]["align_delay"])

        _, _, lands = self.cvproc.detect()
        if len(lands) == 0:
            logger.warning("Target lost during alignment!")
            return False

        dist = np.linalg.norm(np.mean(np.array(lands), axis=0)[:2])
        logger.info(f"Aligning distance: {dist:.3f}m")

        return dist < self.cfg["alignment"]["align_thrsh_dist"]


if __name__ == "__main__":
    config_file = os.getenv("MISSION_CONFIG_FILE", "mission_config.json")
    controller = ERCMissionController(config_path=config_file)
    controller.on_boot()