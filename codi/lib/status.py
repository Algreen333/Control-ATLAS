import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from pymavlink import mavutil


logger = logging.getLogger(__name__)


class FlightPhase(str, Enum):
    INIT = "INIT"
    LIFT_OFF = "LIFT_OFF"
    SEARCHING = "SEARCHING"
    ALIGNING = "ALIGNING"
    LANDING = "LANDING"
    WAITING_MANUAL_RESET = "WAITING_MANUAL_RESET"
    MISSION_COMPLETED = "MISSION_COMPLETED"

@dataclass
class FlightState:
    current_phase: FlightPhase = FlightPhase.INIT

    search_altitude_m: float = 2.0
    search_wp_idx: int = 0

    liftoff_coords: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    landing_coords: List[Tuple[float, float, float]] = []

    last_update_time: float = time.time()


class StateManager:
    """Handles logging of state JSON to survive power cuts."""

    def __init__(self, filepath: str = "flight_status.json"):
        self.filepath = os.path.abspath(filepath)
        self.temp_filepath = self.filepath + ".tmp"

    def load_state(self) -> FlightState:
        if not os.path.exists(self.filepath):
            logger.info("No checkpoint found. Initializing fresh state.")
            return FlightState()

        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)

            phase = FlightPhase(data.get("current_phase", FlightPhase.INIT))

            state = FlightState(
                current_phase=phase,
                custom_platform_mode=data.get("custom_platform_mode", False),
                search_waypoint_idx=data.get("search_waypoint_idx", 0),
                search_altitude_m=data.get("search_altitude_m", 2.0),
                liftoff_coords=tuple(data.get("liftoff_coords", (0.0, 0.0, 0.0))),
                landing_coords=list(data.get("landing_coords", []))
            )

            logger.info(
                f"Checkpoint loaded: Phase={state.current_phase}, "
                f"Attempt={state.mission_attempt}, SearchWP={state.search_waypoint_idx}"
            )
            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint ({e}). Falling back to fresh state.")
            return FlightState()

    def save_state(self, state: FlightState):
        """Atomic write using temporary file + rename to prevent file corruption."""
        try:
            raw_dict = asdict(state)
            raw_dict["current_phase"] = state.current_phase.value

            with open(self.temp_filepath, "w") as f:
                json.dump(raw_dict, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(self.temp_filepath, self.filepath)
            logger.debug(f"Checkpoint saved: Phase={state.current_phase.value}, WP={state.search_waypoint_idx}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def clear(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            logger.info("Checkpoint file cleared.")