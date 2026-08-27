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

import numpy as np

# PARAMS (to be changed)
SEARCH_SQUARE_SIDE = 2
SEARCH_ALT = -2
SEARCH_MAX_DIST_TO_WP = 0.25
ARUCO_HOME_ID = 101
ARUCO_LAND_ID = 102


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
        if ret:
            logger.warning("Capture read returned error")
            return None
        
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
                debug: bool = False
                ):

        # Initialise classes
        self.state_manager = StateManager(state_file)
        self.mav = MavlinkConnection(mavlink_con, mavlink_baud, debug)
        self.capture = VideoCapture(capture_source=capture_source, capture_resolution=capture_resolution, fps=capture_fps)
        self.state = FlightState()
        self.cvproc = CVProcessing(self.capture, capture_config, aruco_dict)

        # Generate flight plan
        self.search_path: List[Tuple[float, float, float]] = [
            (SEARCH_SQUARE_SIDE,  -SEARCH_SQUARE_SIDE, SEARCH_ALT),
            (SEARCH_SQUARE_SIDE,  SEARCH_SQUARE_SIDE,  SEARCH_ALT),
            (-SEARCH_SQUARE_SIDE, SEARCH_SQUARE_SIDE,  SEARCH_ALT),
            (-SEARCH_SQUARE_SIDE, -SEARCH_SQUARE_SIDE, SEARCH_ALT),

            (0, 0, SEARCH_ALT) # RTH if not found
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

    def _execute_search_step(self):
        pos = np.array(self.mav.get_local_position())

        # Image search
        frame, homes, lands = self.cvproc.detect()

        if len(homes) > 0:
            homes = np.array(homes)
            meanhomes = np.mean(homes, axis=0)
            logger.info(f"Mavlink current pos: {pos} vs aruco predicted")

        target_pos = np.array(self.search_path[self.state.search_wp_idx])
        distance = np.linalg.norm(target_pos-pos)


