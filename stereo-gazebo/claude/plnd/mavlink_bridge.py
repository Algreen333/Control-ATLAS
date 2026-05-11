"""
plnd.mavlink_bridge
===================
All MAVLink I/O: connection management, telemetry stream requests,
outbound command helpers (arm, takeoff, mode switches, LANDING_TARGET,
velocity setpoints), and inbound state reads (heading, disarm detection).
"""

import math
import time
import numpy as np

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(connection_string: str, baud: int = 57600) -> mavutil.mavfile:
    """
    Open a MAVLink connection, wait for the first heartbeat, then request
    all telemetry streams needed by the landing sequence.

    Parameters
    ----------
    connection_string : pymavlink URI, e.g. "udpin:0.0.0.0:14550" or "/dev/ttyUSB0"
    baud              : serial baud rate (ignored for UDP/TCP connections)
    """
    print(f"[MAV] Connecting to {connection_string} ...")
    mav = mavutil.mavlink_connection(connection_string, baud=baud)
    mav.wait_heartbeat()
    print(f"[MAV] Heartbeat from system {mav.target_system}, "
          f"component {mav.target_component}")
    request_data_streams(mav, rate_hz=4.0)
    return mav


# ---------------------------------------------------------------------------
# Telemetry stream requests
# ---------------------------------------------------------------------------

def request_message_interval(mav: mavutil.mavfile, msg_id: int, hz: float) -> None:
    """
    Request a specific MAVLink message at a given rate using
    MAV_CMD_SET_MESSAGE_INTERVAL (ArduPilot 4.x+).

    interval_us =  0 → restore default rate
    interval_us = -1 → disable message
    """
    interval_us = int(1e6 / hz)
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,            # confirmation
        msg_id,       # param1: message ID
        interval_us,  # param2: interval (µs)
        0, 0, 0, 0,   # params 3-6 unused
        0,            # param7 unused
    )


def request_data_streams(mav: mavutil.mavfile, rate_hz: float = 4.0) -> None:
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
        mav.mav.request_data_stream_send(
            mav.target_system,
            mav.target_component,
            stream_id,
            int(rate_hz),
            1,   # 1 = start, 0 = stop
        )

    # Per-message overrides (take precedence on 4.x+)
    for msg_id, hz in {
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: rate_hz,
        mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT:           1.0,
    }.items():
        request_message_interval(mav, msg_id, hz)

    print(f"[MAV] Telemetry streams requested @ {rate_hz} Hz. "
          "Waiting for GLOBAL_POSITION_INT ...")

    msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
    if msg is None:
        print("[MAV] WARNING: no GLOBAL_POSITION_INT after 5 s. "
              "Check SR*_POSITION params on the flight controller.")
    else:
        print(f"[MAV] Position streaming OK. Alt={msg.relative_alt / 1000:.1f} m")


# ---------------------------------------------------------------------------
# Flight commands
# ---------------------------------------------------------------------------

def arm_and_takeoff(mav: mavutil.mavfile, altitude_m: float) -> None:
    """Switch to GUIDED, arm, and climb to altitude_m metres."""
    print("[MAV] Switching to GUIDED ...")
    mav.set_mode("GUIDED")
    time.sleep(1)

    print("[MAV] Arming ...")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    print("[MAV] Armed.")

    print(f"[MAV] Taking off to {altitude_m} m ...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,            # confirmation
        0, 0, 0, 0,   # params 1-4 (unused for copter)
        0, 0,         # lat, lon (unused)
        altitude_m,
    )
    while True:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if msg and msg.relative_alt / 1000.0 >= altitude_m * 0.92:
            print(f"[MAV] Reached {msg.relative_alt / 1000:.1f} m.")
            break


def switch_to_land(mav: mavutil.mavfile) -> None:
    """Command LAND mode (activates ArduPilot's PLND controller)."""
    print("[MAV] Switching to LAND mode – PLND active.")
    mav.set_mode("LAND")


def wait_for_disarm(mav: mavutil.mavfile, timeout_s: float = 120.0) -> bool:
    """
    Block until the vehicle disarms (has landed) or timeout_s elapses.

    Returns True if disarmed, False on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg:
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if not armed:
                print("[MAV] Disarmed – landing complete.")
                return True
    print("[MAV] Timeout waiting for disarm.")
    return False


# ---------------------------------------------------------------------------
# Precision-landing message
# ---------------------------------------------------------------------------

def send_landing_target(
    mav: mavutil.mavfile,
    angle_x: float,
    angle_y: float,
    distance: float,
    *,
    size_x: float = 0.0,
    size_y: float = 0.0,
) -> None:
    """
    Emit a LANDING_TARGET MAVLink message (body FRD frame).

    Parameters
    ----------
    angle_x  : lateral angle to target (rad, + = right)
    angle_y  : vertical angle to target (rad, + = forward-below)
    distance : slant range to target (m)
    size_x   : angular width of target (rad)  – optional
    size_y   : angular height of target (rad) – optional
    """
    mav.mav.landing_target_send(
        int(time.monotonic() * 1e6),          # time_usec
        0,                                     # target_num
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        angle_x,
        angle_y,
        distance,
        size_x,
        size_y,
    )


# ---------------------------------------------------------------------------
# Velocity setpoint
# ---------------------------------------------------------------------------

# SET_POSITION_TARGET_LOCAL_NED type_mask: ignore pos, acc, yaw, yaw-rate
# Bits 0-2 = ignore pos x/y/z, bits 6-8 = ignore acc, bits 10-11 = ignore yaw
_VEL_ONLY_MASK = 0b0000_111111000111
_VEL_POS_MASK  = 0b0000_111111000000
_POS_ONLY_MASK = 0b0000_111111111000

def send_velocity_ned(
    mav: mavutil.mavfile,
    vx: float,
    vy: float,
    vz: float = 0.0,
) -> None:
    """
    Send a NED velocity setpoint in GUIDED mode via
    SET_POSITION_TARGET_LOCAL_NED (velocity-only type_mask).

    Parameters
    ----------
    vx : north velocity (m/s, + = north)
    vy : east  velocity (m/s, + = east)
    vz : down  velocity (m/s, + = down; keep 0 during lateral approach)
    """
    mav.mav.set_position_target_local_ned_send(
        int(time.monotonic() * 1e3),         # time_boot_ms
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        _VEL_ONLY_MASK,
        0, 0, 0,      # pos x/y/z – ignored
        vx, vy, vz,   # velocity setpoint
        0, 0, 0,      # acc x/y/z – ignored
        0, 0,         # yaw, yaw_rate – ignored
    )

def send_velocity_frd(
    mav: mavutil.mavfile,
    vx: float,
    vy: float,
    vz: float = 0.0,
) -> None:
    """
    Send a NED velocity setpoint in GUIDED mode via
    SET_POSITION_TARGET_LOCAL_NED (velocity-only type_mask).

    Parameters
    ----------
    vx : forward velocity (m/s, + = forward)
    vy : right   velocity (m/s, + = right)
    vz : down    velocity (m/s, + = down; keep 0 during lateral approach)
    """
    mav.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        int(time.monotonic() * 1e3),         # time_boot_ms
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        _VEL_ONLY_MASK,
        0, 0, 0,      # pos x/y/z – ignored
        vx, vy, vz,   # velocity setpoint
        0, 0, 0,      # acc x/y/z – ignored
        0, 0,         # yaw, yaw_rate – ignored
    ))

def send_pos_frd(
    mav: mavutil.mavfile,
    x : float,
    y : float,
    z : float = 0,
    vx: float = 0.2,
    vy: float = 0.2,
    vz: float = 0.0,
) -> None:
    """
    Send a FRD velocity setpoint in GUIDED mode via
    SET_POSITION_TARGET_LOCAL_NED (velocity-only type_mask).

    Parameters
    ----------
    x  : fwd   distance (m,   + = forward)
    y  : right distance (m,   + = right)
    z  : down  distance (m,   + = down; keep 0 during lateral approach)
    vx : fwd   velocity (m/s, + = forward)
    vy : right velocity (m/s, + = right)
    vz : down  velocity (m/s, + = down; keep 0 during lateral approach)
    """
    mav.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        int(time.monotonic() * 1e3),         # time_boot_ms
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        _VEL_POS_MASK,
        x, y, z,      # pos x/y/z – ignored
        vx, vy, vz,   # velocity setpoint
        0, 0, 0,      # acc x/y/z – ignored
        0, 0,         # yaw, yaw_rate – ignored
    ))

def move_relative(mav: mavutil.mavfile, vertical:int|float, horizontal:int|float, updown:int|float):
    """
    Moves the drone (vertical, horizontal, updown) meters relative to the position and direction of the drone.
        
    :param int|float vertical: Positive is forward
    :param int|float horizontal: Positive is right
    :param int|float updown: Positive is down
    """
    mav.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        0, 
        mav.target_system, 
        mav.target_component, 

        mavutil.mavlink.MAV_FRAME_BODY_FRD, 

        0b010111111000, 
        vertical, 
        horizontal, 
        updown, 

        0, 0, 0, 0, 0, 0, 0, 0))

def send_landing_target_pos_quat(
    mav: mavutil.mavfile,
    tvec_body: "np.ndarray",
    q_body: "np.ndarray",
) -> None:
    """
    Emit a LANDING_TARGET MAVLink message using the position + quaternion
    extended fields (MAVLink 2, position_valid = 1).

    ArduPilot uses this variant when PLND_EST_TYPE = 0 (direct position,
    no EKF re-filtering).  The angle fields are zeroed — AP ignores them
    when position_valid is set.

    Parameters
    ----------
    tvec_body : (3,) – target position in body FRD frame [fwd, right, down] (m)
    q_body    : (4,) – target orientation quaternion [w, x, y, z] in body FRD frame
                       (identity = [1, 0, 0, 0] for a flat, level pad)

    ArduPilot parameter
    -------------------
    PLND_EST_TYPE = 0   use position directly (recommended with this call)
    PLND_EST_TYPE = 1   run through EKF (better for noisy position estimates)
    """
    mav.mav.landing_target_send(
        int(time.monotonic() * 1e6),         # time_usec
        0,                                    # target_num
        mavutil.mavlink.MAV_FRAME_BODY_FRD,  # frame
        0.0, 0.0,                             # angle_x, angle_y – unused
        float(np.linalg.norm(tvec_body)),     # distance (slant range)
        0.0, 0.0,                             # size_x, size_y – unused
        float(tvec_body[0]),                  # x (fwd)
        float(tvec_body[1]),                  # y (right)
        float(tvec_body[2]),                  # z (down)
        [float(q_body[0]), float(q_body[1]),
         float(q_body[2]), float(q_body[3])], # q [w, x, y, z]
        2,                                    # type: LANDING_TARGET_TYPE_VISION_FIDUCIAL
        1,                                    # position_valid = True
    )

# ---------------------------------------------------------------------------
# State reads
# ---------------------------------------------------------------------------

def get_heading_rad(mav: mavutil.mavfile) -> float:
    """
    Return the latest vehicle yaw in radians from VFR_HUD (preferred) or
    ATTITUDE.  Returns 0.0 if neither message is buffered yet.
    """
    msg = mav.recv_match(type="VFR_HUD", blocking=False)
    if msg:
        return math.radians(msg.heading)
    msg = mav.recv_match(type="ATTITUDE", blocking=False)
    if msg:
        return msg.yaw
    return 0.0
