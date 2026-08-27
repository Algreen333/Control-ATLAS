# Init logger
import logging, sys, os

log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),                     # Log to console
        logging.FileHandler("drone_mission.log")      # Log to file for post-crash analysis
    ]
)

logger = logging.getLogger("MissionCtrl")

from lib.status import *
from lib.drone_lib import *
from lib.camera_lib_light import *
from lib.aruco_lib import *

import time
import math
import numpy as np

# PARAMS (to be changed)
SEARCH_SQUARE_SIDE = 2
SEARCH_ALT = -2
SEARCH_MAX_DIST_TO_WP = 0.25
ARUCO_HOME_ID = 101
ARUCO_LAND_ID = 102
ALIGN_ITERS = 6
ALIGN_DELAY = 0.5
ALIGN_THRSH_DIST = 0.75


class CVProcessing:
    def __init__(self, capture: VideoCapture, config_path: str, aruco_dict=cv2.aruco.DICT_ARUCO_ORIGINAL):
        with open(config_path, "r") as f:
            data = json.load(f)

        self.mtx = np.array(data["mtx"])
        self.dist = np.array(data["dist"])
        self.aruco_dict = aruco_dict

        self.detector = ArucoDetector(self.mtx, self.dist, self.aruco_dict)

        self.capture = capture

    def detect(self, do_draw = False):
        ret, frame = self.capture.read()
        if not ret:
            logger.warning("Capture read returned error")
            return None

        prediction = self.detector.full_prediction(frame, do_draw) # pass 'frame' here too!
        if prediction is None:
            return frame, [], []
            
        frame, detections = prediction
        
        frame, detections = self.detector.full_prediction(do_draw)

        home_detections = []
        land_detections = []

        for det in detections:
            pos, rot, id = det
            if id == ARUCO_HOME_ID: home_detections.append(pos)
            elif id == ARUCO_LAND_ID: land_detections.append(pos)

            logger.info(f"Detected ArUCO {id} with offset {pos}.")

        return frame, home_detections, land_detections


class ERCMissionController:
    """Manages full mission execution and mid-path recovery."""

    def __init__(self, 
                state_file: str = "flight_checkpoint.json", 
                capture_source=0, capture_resolution: tuple[int,int] = (1640, 1232), capture_fps: float = 30.0,
                capture_config="./configs/1640x1232-v2.conf", aruco_dict=cv2.aruco.DICT_ARUCO_ORIGINAL,
                mavlink_con: str = "/dev/ttyAMA0", mavlink_baud: int = 57600,
                debug: bool = False, doGazebo: bool = False
                ):

        # Initialise classes
        self.state_manager = StateManager(state_file)
        self.mav = MavlinkConnection(mavlink_con, mavlink_baud, debug)
        if doGazebo: self.capture = GazeboVideoCapture(capture_source=capture_source, resolution=capture_resolution, fps=capture_fps)
        else: self.capture = VideoCapture(capture_source=capture_source, resolution=capture_resolution, fps=capture_fps)

        self.state = FlightState()
        self.cvproc = CVProcessing(self.capture, capture_config, aruco_dict)

        # Generate flight plan
        self.search_path: List[Tuple[float, float, float]] = [
            (SEARCH_SQUARE_SIDE,  -SEARCH_SQUARE_SIDE, SEARCH_ALT),
            (SEARCH_SQUARE_SIDE,  SEARCH_SQUARE_SIDE,  SEARCH_ALT),
            (-SEARCH_SQUARE_SIDE, SEARCH_SQUARE_SIDE,  SEARCH_ALT),
            (-SEARCH_SQUARE_SIDE, -SEARCH_SQUARE_SIDE, SEARCH_ALT)
            #,(0, 0, SEARCH_ALT) # RTH if not found
        ]

    def on_boot(self):
        """Bootloader entry point executed by systemd service."""
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
        """Regain control of the hovering drone and resume the active phase."""
        logger.info("Switching FC back to GUIDED mode to retake control...")
        self.mav.setGuided()
        time.sleep(1)

        pos = self.mav.get_local_position()
        if pos:
            # Hold current spot while confirming state
            self.mav.send_target_ned(pos[0], pos[1], pos[2])
            logger.info(f"Hover position hold commanded at NED: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        logger.info(f"Resuming phase: {self.state.current_phase.value}")

    def _handle_grounded_boot(self):
        if self.state.current_phase == FlightPhase.INIT:
            logger.info("Ready for launch. Waiting for launch signal...")
        elif self.state.current_phase == FlightPhase.WAITING_MANUAL_RESET:
            logger.info(f"Awaiting manual reset...")

    def run_mission_loop(self):
        while self.state.current_phase != FlightPhase.MISSION_COMPLETED:
            phase = self.state.current_phase

            if phase == FlightPhase.INIT:
                # Step 1: Liftoff
                logger.info(f"--- Starting Mission ---")
                self.mav.arm_and_takeoff(self.state.search_altitude_m)
                time.sleep(5)
                self.state.current_phase = FlightPhase.SEARCHING
                self.state_manager.save_state(self.state)

            elif phase == FlightPhase.SEARCHING:
                result = self._execute_search_step()
                if result == True: 
                    self.state.current_phase = FlightPhase.ALIGNING
                    self.state_manager.save_state(self.state)

            elif phase == FlightPhase.ALIGNING:
                result = self._execute_align_step()
                if result == True:
                    logger.info("Approach finalised, landing...")
                    self.state.current_phase = FlightPhase.LANDING
                    self.state_manager.save_state(self.state)

            elif phase == FlightPhase.LANDING:
                self.mav.switch_to_land()
                while self.mav.is_airborne(alt_threshold_m=0.2):
                    time.sleep(1)
                self.state.current_phase = FlightPhase.MISSION_COMPLETED

    def _execute_search_step(self):
        pos = np.array(self.mav.get_local_position())
        if pos is None:
            logger.warning("Could not retrieve local position!")
            return False
        logger.debug(f"Mavlink current pos: {pos}")

        att_msg = self.mav.mav.recv_match(type='ATTITUDE', blocking=False)
        yaw = att_msg.yaw if att_msg else 0.0

        # Image search
        frame, homes, lands = self.cvproc.detect()
        meanhomes = None
        meanlands = None

        if len(homes) > 0:
            homes = np.array(homes)
            meanhomes = np.mean(homes, axis=0)
            logger.info(f"Home ArUco (101) detected at camera offset: {meanhomes}")

        if len(lands) > 0:
            lands = np.array(lands)
            meanlands = np.mean(lands, axis=0)
            logger.info(f"Landing ArUco (102) detected at camera offset: {meanlands}")

            cam_x, cam_y, cam_z = meanlands
            # Map OpenCV camera frame to drone Body frame
            body_forward = -cam_y  # Up in the image is forward for the drone
            body_right = cam_x     # Right in the image is right for the drone

            # Apply 2D rotation matrix based on drone's Yaw (heading)
            north_offset = (body_forward * math.cos(yaw)) - (body_right * math.sin(yaw))
            east_offset = (body_forward * math.sin(yaw)) + (body_right * math.cos(yaw))

            # Add offsets to drone's current NED position
            target_n = pos[0] + north_offset
            target_e = pos[1] + east_offset
            target_d = pos[2] # Keep the drone's current Z altitude as the reference

            logger.info(f"Calculated Landing Pad NED: N={target_n:.2f}, E={target_e:.2f}")

            self.register_landing_target((target_n, target_e, target_d))


        target_pos = np.array(self.search_path[self.state.search_wp_idx])
        distance = np.linalg.norm(target_pos[:2] - pos[:2]) # (ignoring Z/altitude)

        if distance < SEARCH_MAX_DIST_TO_WP:
            logger.info(f"Reached search waypoint {self.state.search_wp_idx}")
            if self.state.search_wp_idx < len(self.search_path) - 1:
                self.state.search_wp_idx += 1
                self.state_manager.save_state(self.state)
                return False
            else: return True

    def register_landing_target(self, coords):
        self.state.landing_coords.append(coords)
        self.state_manager.save_state(self.state)

    def _execute_align_step(self):
        self._execute_search_step() # Keep looking for the landing target for better position estimate
        
        if len(self.state.landing_coords) == 0:
            lnd_crds = (0.0, 0.0, 0.0)
        else:
            coords = np.array(self.state.landing_coords)
            lnd_crds = np.mean(coords, axis=0)

        logger.info(f"Centering over landing target: ({lnd_crds[0]:.2f}, {lnd_crds[1]:.2f})")

        for _ in range(ALIGN_ITERS):
            self.mav.send_target_ned(lnd_crds[0], lnd_crds[1], -self.state.search_altitude_m)
            time.sleep(ALIGN_DELAY)

        _, _, lands = self.cvproc.detect()
        if len(lands) == 0:
            logger.warning("Target lost during alignment!")
            return False
        lands = np.array(lands)
        meanlands = np.mean(lands, axis=0)
        dist = np.linalg.norm(meanlands[:2])

        logger.info(f"Aligning complete. Distance: {dist}m")

        if dist < ALIGN_THRSH_DIST: return True
        return False


if __name__ == "__main__":
    SERIAL_PORT = os.getenv("MAVLINK_PORT", "/dev/ttyAMA0")
    BAUD_RATE = int(os.getenv("MAVLINK_BAUD", "57600"))
    STATE_FILE = os.getenv("FLIGHT_CHKP_FILE", "flight_checkpoint.json")

    controller = ERCMissionController(state_file=STATE_FILE, mavlink_con=SERIAL_PORT,mavlink_baud=BAUD_RATE, debug=False, doGazebo=True)
    controller.on_boot()