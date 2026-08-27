import os
os.environ["MAVLINK20"] = "1" # MUST be set before importing mavutil

from pymavlink import mavutil
import time
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


# Coses a mirar:
# https://mavlink.io/en/messages/common.html#MAV_CMD_REQUEST_MESSAGE -> request_message
# https://mavlink.io/en/messages/common.html#COMMAND_INT -> COMMAND_INT
# https://mavlink.io/en/messages/common.html#ORBIT_YAW_BEHAVIOUR -> Behaviour 
# https://mavlink.io/en/messages/common.html#MAV_FRAME -> Reference frame



class MavlinkConnection:
    def __init__(self, connection_string: str, baud: int = 57600, debug=False):
        """
        Open a MAVLink connection, wait for the first heartbeat, then request
        all telemetry streams needed by the landing sequence.

        Parameters
        ----------
        connection_string : pymavlink URI, e.g. "udpin:0.0.0.0:14550" or "/dev/ttyUSB0"
        baud              : serial baud rate (ignored for UDP/TCP connections)
        debug             : if set, action commands will not be executed
        """
        logger.info(f"[MAV] Connecting to {connection_string} ...")
        self.mav = mavutil.mavlink_connection(connection_string, baud=baud)
        self.mav.wait_heartbeat()
        logger.info(f"[MAV] Heartbeat from system {self.mav.target_system}, "
            f"component {self.mav.target_component}")
        self.request_data_streams(rate_hz=4.0)

        self.debug = debug


    # ---------------------------------------------------------------------------
    # Telemetry stream requests
    # ---------------------------------------------------------------------------

    def request_message_interval(self, msg_id: int, hz: float) -> None:
        """
        Request a specific MAVLink message at a given rate using
        MAV_CMD_SET_MESSAGE_INTERVAL (ArduPilot 4.x+).

        interval_us =  0 → restore default rate
        interval_us = -1 → disable message
        """
        interval_us = int(1e6 / hz)
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,            # confirmation
            msg_id,       # param1: message ID
            interval_us,  # param2: interval (µs)
            0, 0, 0, 0,   # params 3-6 unused
            0,            # param7 unused
        )

    def request_data_streams(self, rate_hz: float = 4.0) -> None:
        """
        Ask ArduPilot to start streaming the telemetry groups needed for
        precision landing.

        ArduPilot sends nothing but heartbeats by default.  This function uses
        both the legacy REQUEST_DATA_STREAM groups (all firmware versions) and
        per-message MAV_CMD_SET_MESSAGE_INTERVAL overrides (firmware 4.x+).

        Streams enabled
        ---------------
        MAV_DATA_STREAM_POSITION → GLOBAL_POSITION_INT (altitude, landing check)
        MAV_DATA_STREAM_EXTRA1   → ATTITUDE            (heading fallback)
        MAV_DATA_STREAM_EXTRA2   → VFR_HUD             (heading, preferred)
        """
        # Legacy group requests (compatible with all ArduPilot versions)
        for stream_id in (
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
        ):
            self.mav.mav.request_data_stream_send(
                self.mav.target_system,
                self.mav.target_component,
                stream_id,
                int(rate_hz),
                1,   # 1 = start, 0 = stop
            )

        # Per-message overrides (take precedence on 4.x+)
        for msg_id, hz in {
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: rate_hz,
            mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT:           1.0,
        }.items():
            self.request_message_interval(msg_id, hz)

        logger.info(f"[MAV] Telemetry streams requested @ {rate_hz} Hz. "
            "Waiting for GLOBAL_POSITION_INT ...")

        msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg is None:
            logger.info("[MAV] WARNING: no GLOBAL_POSITION_INT after 5 s. "
                "Check SR*_POSITION params on the flight controller.")
        else:
            logger.info(f"[MAV] Position streaming OK. Alt={msg.relative_alt / 1000:.1f} m")
    
    def is_armed(self) -> bool | None:
        """
        Returns True or False if armed or not armed, if no heartbeat returns None
        """

        hb = self.mav.recv_match(type="HEARTBEAT", blocking=False)
        if hb and hb.type == 2:
            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            return armed
        
        else: return None


    # ---------------------------------------------------------------------------
    # Status commands
    # ---------------------------------------------------------------------------

    def get_local_position(self) -> Optional[Tuple[float, float, float]]:
        """Returns (x, y, z) in NED frame (meters) where Z is negative upward."""
        msg = self.mav.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=2.0)
        if msg:
            return (msg.x, msg.y, msg.z)
        return None
    
    def is_airborne(self, alt_threshold_m: float = 0.5) -> bool:
        """Determines if the drone is in flight by checking arm status and altitude."""
        msg_hb = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2.0)
        is_armed = False
        if msg_hb:
            is_armed = bool(msg_hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

        pos = self.get_local_position()
        curr_alt = -pos[2] if pos else 0.0

        logger.info(f"Telemetry Status: Armed={is_armed}, Estimated Alt={curr_alt:.2f}m")
        return is_armed and (curr_alt > alt_threshold_m)

    
    # ---------------------------------------------------------------------------
    # Flight control commands
    # ---------------------------------------------------------------------------

    def arm(self) -> None:
        logger.info("[MAV] Arming ...")
        self.mav.arducopter_arm()
        self.mav.motors_armed_wait()
        logger.info("[MAV] Armed")
    
    def waitArmed(self) -> None:
        logger.info("[MAV] Waiting for Arm ...")
        self.mav.motors_armed_wait()
        logger.info("[MAV] Armed")

    def setGuided(self) -> None:
        logger.info("[MAV] Switching to GUIDED ...")

        self.mav.set_mode("GUIDED")
        # Wait for guided
        msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        while (msg is None or msg.type != 2 or not(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED) or not (msg.custom_mode & 4)):
            if (msg is not None and msg.type == 2. and not(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED) or not (msg.custom_mode & 4)):
                logger.info("[MAV] Retrying to switch to GUIDED ...")
                self.mav.set_mode("GUIDED")
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)

        logger.info("[MAV] GUIDED mode set")

    def waitGuided(self) -> None:
        logger.info("[MAV] Waiting for GUIDED mode ...")

        msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        while (msg is None or msg.type != 2 or not(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED) or not (msg.custom_mode & 4)):
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        
        logger.info("[MAV] GUIDED mode set")
    
    def setStabilize(self) -> None:
        logger.info("[MAV] Switching to STABILIZE ...")

        self.mav.set_mode("STABILIZE")
        # Wait for guided
        msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        while (msg is None or msg.type != 2 or not(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED)):
            if (msg is not None and msg.type == 2. and not(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED)):
                logger.info("[MAV] Retrying to switch to STABILIZE ...")
                self.mav.set_mode("STABILIZE")
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)

        logger.info("[MAV] STABILIZE mode set")

    def waitStabilize(self) -> None:
        logger.info("[MAV] Waiting for STABILIZE mode ...")

        msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        while (msg is None or msg.type != 2 or not(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED) or not (msg.custom_mode & 4)):
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        
        logger.info("[MAV] STABILIZE mode set")

    def takeoff(self, altitude_m: float):
        if self.debug: 
            logger.info(f"<DEBUG> [MAV] Taking off to {altitude_m} m ...")
            return
        
        logger.info(f"[MAV] Taking off to {altitude_m} m ...")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,            # confirmation
            0, 0, 0, 0,   # params 1-4 (unused for copter)
            0, 0,         # lat, lon (unused)
            altitude_m,
        )
        while True:
            msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
            if msg and msg.relative_alt / 1000.0 >= altitude_m * 0.92:
                logger.info(f"[MAV] Reached {msg.relative_alt / 1000:.1f} m.")
                break

    def arm_and_takeoff(self, altitude_m: float) -> None:
        """Switch to GUIDED, arm, and climb to altitude_m metres."""
        if self.debug: 
            logger.info(f"<DEBUG> [MAV] Arm and takeoff")
            return

        logger.info("[MAV] Switching to GUIDED ...")
        self.mav.set_mode("GUIDED")
        time.sleep(1)

        logger.info("[MAV] Arming ...")
        self.mav.arducopter_arm()
        self.mav.motors_armed_wait()
        logger.info("[MAV] Armed.")

        logger.info(f"[MAV] Taking off to {altitude_m} m ...")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,            # confirmation
            0, 0, 0, 0,   # params 1-4 (unused for copter)
            0, 0,         # lat, lon (unused)
            altitude_m,
        )
        while True:
            msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
            if msg and msg.relative_alt / 1000.0 >= altitude_m * 0.92:
                logger.info(f"[MAV] Reached {msg.relative_alt / 1000:.1f} m.")
                break

    def switch_to_land(self) -> None:
        """Command LAND mode (activates ArduPilot's PLND controller)."""
        logger.info("[MAV] Switching to LAND mode – PLND active.")
        self.mav.set_mode("LAND")

    def wait_for_disarm(self, timeout_s: float = 120.0) -> bool:
        """
        Block until the vehicle disarms (has landed) or timeout_s elapses.

        Returns True if disarmed, False on timeout.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
            if msg:
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if not armed:
                    logger.info("[MAV] Disarmed – landing complete.")
                    return True
        logger.info("[MAV] Timeout waiting for disarm.")
        return False


    # ---------------------------------------------------------------------------
    # Flight movement commands
    # ---------------------------------------------------------------------------

    def move_relative(self, vertical:int|float, horizontal:int|float, updown:int|float):
        """
        Moves the drone (vertical, horizontal, updown) meters relative to the position and direction of the drone.
         
        :param int|float vertical: Positive is forward
        :param int|float horizontal: Positive is right
        :param int|float updown: Positive is down
        """

        if self.debug: 
            logger.info(f"<DEBUG> [MAV] Moving ({vertical, horizontal, updown}) m ...")
            return

        self.mav.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            0, 
            self.mav.target_system, 
            self.mav.target_component, 

            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 

            0b011111111000,  # YAW RATE = 0 (bit 11)
            vertical, 
            horizontal, 
            updown, 

            0, 0, 0, 0, 0, 0, 0, 0))
    
    def move_relative_w_speed(self, vertical:int|float, horizontal:int|float, updown:int|float, speedv:float, speedh:float, speedz:float=0):
        """
        Moves the drone (vertical, horizontal, updown) meters relative to the position and direction of the drone.
         
        :param int|float vertical: Positive is forward
        :param int|float horizontal: Positive is right
        :param int|float updown: Positive is down
        """

        if self.debug: 
            logger.info(f"<DEBUG> [MAV] Moving ({vertical}, {horizontal}, {updown}) m at ({speedv}, {speedh}, {speedz}) m/s ...")
            return

        self.mav.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            0, 
            self.mav.target_system, 
            self.mav.target_component, 

            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 

            0b011111000000,  # YAW RATE = 0 (bit 11)
            vertical, 
            horizontal, 
            updown, 

            speedv,
            speedh,
            speedz,

            0, 0, 0, 0, 0))
    
    def move_velocity_body(self, v_fwd: float, v_right: float, v_down: float):
        """
        Commands the drone to move at specific velocities relative to its body.
        
        :param float v_fwd: Forward velocity in m/s
        :param float v_right: Rightward velocity in m/s
        :param float v_down: Downward velocity in m/s
        """

        if self.debug: 
            logger.info(f"<DEBUG> [MAV] Moving ({v_fwd}, {v_right}, {v_down}) m/s ...")
            return

        self.mav.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            0, 
            self.mav.target_system, 
            self.mav.target_component, 
            
            # Frame 8: Body-relative (X=Forward, Y=Right, Z=Down)
            mavutil.mavlink.MAV_FRAME_BODY_NED, 

            # Mask: Use Velocity. Ignore Pos, Accel, Force, Yaw, and Yaw Rate.
            0b011111000111, 
            
            0, 0, 0,                # Position targets (ignored)
            v_fwd, v_right, v_down, # Velocity targets (m/s)
            0, 0, 0,                # Acceleration targets (ignored)
            0, 0))                  # Yaw and Yaw rate (ignored)

    def send_target_ned(self, x: float, y: float, z: float, speed_ms: float = None, face_dest: bool = False):
        """
        Sends position target in local NED coordinates (z is negative upward).
        
        :param float x: North target in meters
        :param float y: East target in meters
        :param float z: Down target in meters (negative for altitude)
        :param float speed_ms: Optional speed limit in m/s (feed-forward velocity)
        :param bool face_dest: If True, yaws to face the target. If False, holds current heading.
        """
        import math
        
        # Base mask: ignore velocity, accel, force, yaw, and yaw_rate (0x0FF8 -> 4088)
        # Bits: 0,1,2 (Pos) are 0. Bits 3-11 are 1 (Ignored).
        type_mask = 0b0000111111111000 
        
        vx, vy, vz = 0.0, 0.0, 0.0
        yaw = 0.0
        
        # Use non-blocking reads to get current state without stuttering the mission loop
        pos_msg = self.mav.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        att_msg = self.mav.recv_match(type="ATTITUDE", blocking=False)
        
        # Handle Speed (Velocity Feed-Forward)
        if speed_ms is not None and pos_msg:
            dx = x - pos_msg.x
            dy = y - pos_msg.y
            dz = z - pos_msg.z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            if dist > 0.05: # Avoid division by zero
                # Calculate velocity vector pointing exactly to the target
                vx = (dx / dist) * speed_ms
                vy = (dy / dist) * speed_ms
                vz = (dz / dist) * speed_ms
                
                # Enable velocity bits by clearing bits 3, 4, and 5
                type_mask &= ~0b0000000000111000

        # Handle Yaw (Heading)
        if face_dest and pos_msg:
            dx = x - pos_msg.x
            dy = y - pos_msg.y
            
            # Calculate angle in radians towards the target
            yaw = math.atan2(dy, dx)
            
            # Enable yaw bit by clearing bit 10
            type_mask &= ~0b0000010000000000
            
        elif not face_dest and att_msg:
            # Explicitly force the drone to hold its current yaw.
            # (Prevents ArduPilot's WP_YAW_BEHAVIOR from taking over and auto-rotating)
            yaw = att_msg.yaw
            
            # Enable yaw bit by clearing bit 10
            type_mask &= ~0b0000010000000000

        # Send the command
        self.mav.mav.set_position_target_local_ned_send(
            0,  
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,
            vx, vy, vz,  # Velocities 
            0, 0, 0,     # Accelerations
            yaw, 0       # Yaw, Yaw_rate
        )
        

    # ---------------------------------------------------------------------------
    # PRLND commands
    # ---------------------------------------------------------------------------

    def send_landing_target_pos_quat(self, x_m: float, y_m: float, z_m: float, q_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)):
        """
        Sends a LANDING_TARGET message using the MAVLink 2 "Target as Position/Quaternion" fields.
        """
        import math
        
        #time_usec = int(time.time() * 1e6)
        time_usec = 0 
        
        distance_m = math.sqrt(x_m**2 + y_m**2 + z_m**2)
        
        # If these are 0.0, it might ignore the X/Y offsets.
        angle_x = math.atan2(y_m, z_m)
        angle_y = math.atan2(x_m, z_m)
        
        self.mav.mav.landing_target_send(
            time_usec,                            
            0,                                    
            mavutil.mavlink.MAV_FRAME_BODY_FRD,   
            angle_x, angle_y,
            distance_m,                           
            1.0, 1.0,                             
            x_m,                                  
            y_m,                                  
            z_m,                                  
            q_wxyz,                               
            mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,                                    
            1                                     
        )

    def send_landing_target_angles(self, fwd_m: float, right_m: float, down_m: float):
        """
        Sends a LANDING_TARGET message using angular offsets (Target Relative to Captured Image).
        Instead of defining an exact coordinate, this tells the flight controller the 
        angle to the target from the downward-facing camera's perspective.
        
        :param float fwd_m: Forward distance to target in meters
        :param float right_m: Right distance to target in meters
        :param float down_m: Downward distance to target (altitude) in meters
        """
        import math
        
        # Hardcode time_usec to 0 to bypass companion computer clock sync issues
        time_usec = 0 
        
        # Calculate the absolute line-of-sight distance to the target
        distance_m = math.sqrt(fwd_m**2 + right_m**2 + down_m**2)
        
        # Calculate the angular offsets (in radians)
        # atan2(offset, height) gives us the exact angle from the center of the camera lens
        angle_x = float(math.atan2(right_m, fwd_m))
        angle_y = float(math.atan2(right_m, down_m))
        
        self.mav.mav.landing_target_send(
            time_usec,                            
            0,                                    # target_num (0 = single target)
            mavutil.mavlink.MAV_FRAME_BODY_NED,   # Frame of reference
            
            # The core of the "Relative to Image" method:
            angle_x,                              # X-axis angle (Forward/Back)
            angle_y,                              # Y-axis angle (Left/Right)
            distance_m,                           # Absolute distance
            
            0.0, 0.0,                             # size_x, size_y (unused)
            
            # Since we are using angles, we zero out the position and quaternion fields
            0.0, 0.0, 0.0,                        # x, y, z
            (1.0, 0.0, 0.0, 0.0),                 # q_wxyz
            2,                                    # type: 2 = MAV_LANDING_TARGET_TYPE_VISION_FIDUCIAL
            1                                   
        )

    def send_landing_target(
        self,
        angle_x: float,
        angle_y: float,
        distance: float,
        size_x: float = 1.0,
        size_y: float = 1.0,
    ) -> None:
        """
        Emit a LANDING_TARGET MAVLink message.

        Parameters
        ----------
        angle_x  : lateral angle to target (rad, + = right)
        angle_y  : longitudinal angle to target (rad, + = forward-down)
        distance : slant range to target (m)
        size_x   : angular width of target (rad)  – optional
        size_y   : angular height of target (rad) – optional
        """
        self.mav.mav.landing_target_send(
            int(time.monotonic() * 1e6),   # time_usec
            0,                              # target_num
            mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame (body forward-right-down)
            angle_x,
            angle_y,
            distance,
            size_x,
            size_y,
            mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,
            1
        )
    
    def abort(self):
        self.move_velocity_body(0,0,0)
        self.switch_to_land()

    def send_odometry_message(self, x, y, z, q_wxyz=[1,0,0,0], vx=0, vy=0, vz=0, rollspeed=0, pitchspeed=0, yawspeed=0):
        """
        Send a VISION_POSITION_ESTIMATE MAVLink message.

        Parameters
        ----------
        """
        
        self.mav.mav.odometry_send(
            int(time.monotonic() * 1e6),
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            x, y, z,
            q_wxyz, 
            vx, vy, vz,
            rollspeed, pitchspeed, yawspeed, 
            [0.0]*21,
            [0.0]*21
            )
    
    
    def actuate_servo(self, servo_number=8, pwm_value=1900):
        self.mav.mav.command_long_send(
        self.mav.target_system,    # Target system ID
        self.mav.target_component, # Target component ID
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, # Command ID
        0,                       # Confirmation
        servo_number,            # Parameter 1: Servo instance (1 for channel 1, 2 for channel 2, etc.)
        pwm_value,               # Parameter 2: PWM value (e.g., 1500)
        0, 0, 0, 0, 0            # Unused parameters
    )
        
    def open_gripper(self):
        self.actuate_servo(8, 1900)
    
    def close_gripper(self):
        self.actuate_servo(8, 1100)