"""
ArduPilot Precision Landing (PLND) via MAVLink
================================================
Connects to a flight controller, streams LANDING_TARGET messages derived
from per-frame vision estimates, and commands an autonomous precision landing.

Requirements:
    pip install pymavlink numpy opencv-python

ArduPilot parameters to set on the vehicle:
    PLND_ENABLED  = 1
    PLND_TYPE     = 2   (MAVLink / companion computer)
    PLND_BUS      = 0
    PLND_EST_TYPE = 1   (raw sensor, let AP do the EKF fusion)
              -- or --
    PLND_EST_TYPE = 0   (use the position reported directly, no AP filtering)

Usage:
    python plnd_landing.py --connect udpin:0.0.0.0:14550
    python plnd_landing.py --connect /dev/ttyUSB0 --baud 57600
"""

import argparse
import math
import time
import sys
from typing import Optional, Tuple

import numpy as np
from pymavlink import mavutil



from scipy.spatial.transform import Rotation, Slerp
import cv2

from lib.movement import *
from lib.pose_estimation import *
from lib.aruco_lib import *


DO_LAND = True
DO_TAKEOFF = True

MARKER_SIZE = 1
MARKER_ID = 49

# ---------------------------------------------------------------------------
# Camera initialization
# ---------------------------------------------------------------------------

VIDSRC_PORT_WIDE = 5800
VIDSRC_PORT_NARR = 5600

pipeline_wide = (
    f"udpsrc port={VIDSRC_PORT_WIDE} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)
pipeline_narr = (
    f"udpsrc port={VIDSRC_PORT_NARR} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)

cap_wide = cv2.VideoCapture(pipeline_wide, cv2.CAP_GSTREAMER)
print(cap_wide.isOpened())

cap_narr = cv2.VideoCapture(pipeline_narr, cv2.CAP_GSTREAMER)
print(cap_wide.isOpened())


mtx_wide = get_gazebo_camera_matrix(1536, 864, 102/180*np.pi, 48.8/180*np.pi)
dst_wide = np.zeros(5, dtype=np.float32)
mtx_narr = get_gazebo_camera_matrix(1640, 1232, 62.2/180*np.pi, 67/180*np.pi)
dst_narr = np.zeros(5, dtype=np.float32)

detWide = ArucoDetector(mtx_wide, dst_wide, cv2.aruco.DICT_4X4_50)
detNarr = ArucoDetector(mtx_wide, dst_wide, cv2.aruco.DICT_4X4_50)

T_C_WIDE = np.eye(4)
T_C_WIDE[0, 3] = 0.05
T_C_NARR = np.eye(4)
T_C_NARR[0, 3] = -0.05



# ---------------------------------------------------------------------------
# Frames processing step
# ---------------------------------------------------------------------------

def process_frame() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Capture one camera frame, detect the landing target, and estimate its pose.

    Returns
    -------
    (tvec, rvec) if a target is detected, or None if no target is visible.

    tvec : np.ndarray, shape (3,)
        Translation from the camera origin to the target centre, expressed in
        the CAMERA frame (X right, Y down, Z forward), in metres.
    rvec : np.ndarray, shape (3,)
        Rodrigues rotation vector describing the target's orientation relative
        to the camera.  Not consumed by PLND directly but available for your
        own use (e.g. logging or heading correction).
    """

    ret_w, frame_wide = cap_wide.read()
    ret_n, frame_narr = cap_narr.read()

    rvec_wide = None
    tvec_wide = None
    rvec_narr = None
    tvec_narr = None
    rvec = None
    tvec = None

    if ret_w:
        gray = cv2.cvtColor(frame_wide, cv2.COLOR_BGR2GRAY)
        corners_wide, ids_wide, rejected_wide = detWide.detectMarkers(gray)
        
        indices = np.where(ids_wide == MARKER_ID)[0]
        if len(indices) > 0: rvec_wide, tvec_wide = detWide.estimate_pose(corners_wide[indices[0]], MARKER_SIZE)

    if ret_n:
        gray = cv2.cvtColor(frame_narr, cv2.COLOR_BGR2GRAY)
        corners_narr, ids_narr, rejected_narr = detNarr.detectMarkers(gray)

        indices = np.where(ids_narr == MARKER_ID)[0]
        if len(indices) > 0: rvec_narr, tvec_narr = detNarr.estimate_pose(corners_narr[indices[0]], MARKER_SIZE)

    if ret_w and ret_n: cv2.imshow("img", joint_display(frame_wide, frame_narr))

    # Estimation from both cameras
    if rvec_wide is not None and rvec_narr is not None:
        rvec, tvec = fuse_stereo_aruco_poses(rvec_wide, tvec_wide, T_C_WIDE, rvec_narr, tvec_narr, T_C_NARR)

    elif rvec_wide is not None:
        print("ONLY WIDE DETECTION")
        rvec, tvec = rvec_wide, tvec_wide

    elif rvec_narr is not None:
        print("ONLY NARROW DETECTION")
        rvec, tvec = rvec_narr, tvec_narr
    
    cv2.waitKey(30)

    if rvec is not None and tvec is not None:
        return tvec, rvec
    else: return None
    
    # TODO: replace the body below with your actual detection pipeline
    # Example skeleton using OpenCV + ArUco:
    #
    #   ret, frame = cap.read()
    #   corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    #   if ids is not None:
    #       rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    #           corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS)
    #       return tvecs[0].flatten(), rvecs[0].flatten()
    #   return None
    raise NotImplementedError("Implement process_frame() with your vision pipeline.")


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cam_to_body(tvec: np.ndarray, cam_to_body_R: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from the camera frame into the vehicle body frame.

    Parameters
    ----------
    tvec : (3,) ndarray – target position in camera frame (X right, Y down, Z fwd)
    cam_to_body_R : (3,3) ndarray – rotation matrix from camera to body (FRD)

    Returns
    -------
    (3,) ndarray – target position in body frame (X fwd, Y right, Z down)
    """
    return cam_to_body_R @ tvec


def tvec_to_angles(tvec_body: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a target position vector (body frame, FRD) to the angle offsets
    and slant distance expected by LANDING_TARGET.

    ArduPilot convention
    --------------------
    angle_x : positive → target is to the RIGHT  of the vehicle centre
    angle_y : positive → target is BELOW  (further from the sky) -- forward
              Actually in MAVLink: angle_x/y are offsets in the sensor frame,
              but AP maps them body-forward(x) and body-right(y) when
              PLND_TYPE=2.  Check your AP version; the mapping below matches
              most stable releases.

    Parameters
    ----------
    tvec_body : (3,) ndarray – [fwd, right, down] in metres

    Returns
    -------
    (angle_x, angle_y, distance) – angles in radians, distance in metres
    """

    #print(f"[TVEC2ANGL] tvec: {tvec_body}")
    
    up, bwd, right = tvec_body
    down = -up
    fwd = -bwd

    distance = float(np.linalg.norm(tvec_body))

    if distance < 1e-6:
        return 0.0, 0.0, 0.0

    # Horizontal angle (+ = target right of vehicle)
    angle_x = math.atan2(right, fwd)
    # Vertical angle (+ = target below / camera looking down more)
    angle_y = math.atan2(-down, fwd)   # negative because down is positive Z

    return angle_x, angle_y, distance

# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def joint_display(img_server, img_client):
    # --- DYNAMIC DIMENSION MATCHING (ZERO-PADDING) ---
    h_server, w_server = img_server.shape[:2]
    h_client, w_client = img_client.shape[:2]
    
    # If heights mismatch, pad the bottom of the shorter image with zeros (black)
    if h_server > h_client:
        padding = h_server - h_client
        img_client = cv2.copyMakeBorder(img_client, 0, padding, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif h_client > h_server:
        padding = h_client - h_server
        img_server = cv2.copyMakeBorder(img_server, 0, padding, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])

    combined_frame = cv2.hconcat([img_server, img_client])
    resized = cv2.resize(combined_frame, (0,0), fx = 0.5, fy = 0.5)

    # Safely update the global frame reference
    return resized



# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def request_message_interval(mav: mavutil.mavfile, msg_id: int, hz: float) -> None:
    """
    Ask ArduPilot to stream a specific message at the given rate.
    Uses MAV_CMD_SET_MESSAGE_INTERVAL (preferred on ArduPilot 4.x+).
    interval_us = 0  → use default rate
    interval_us = -1 → disable
    """
    interval_us = int(1e6 / hz)
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,           # confirmation
        msg_id,      # param1: message ID
        interval_us, # param2: interval in microseconds
        0, 0, 0, 0,  # params 3-6 unused
        0,           # param7 unused
    )


def request_data_streams(mav: mavutil.mavfile, rate_hz: float = 4.0) -> None:
    """
    Request all telemetry streams needed for the landing sequence.

    ArduPilot will NOT send anything other than heartbeats until you ask.
    This uses both the legacy REQUEST_DATA_STREAM (broad groups) and
    per-message MAV_CMD_SET_MESSAGE_INTERVAL for critical messages, so it
    works across older and newer firmware versions.

    Streams requested
    -----------------
    GLOBAL_POSITION_INT  – altitude check during takeoff / landing detection
    HEARTBEAT            – arm/disarm detection (already sent, but explicit)
    EKF_STATUS_REPORT    – optional, useful for health monitoring
    """
    # --- Legacy stream groups (work on all ArduPilot versions) -------------
    for stream_id in (
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,   # GLOBAL_POSITION_INT etc.
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,     # ATTITUDE
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,     # VFR_HUD
    ):
        mav.mav.request_data_stream_send(
            mav.target_system,
            mav.target_component,
            stream_id,
            int(rate_hz),
            1,  # 1 = start, 0 = stop
        )

    # --- Per-message overrides (ArduPilot 4.x+) ----------------------------
    # These take precedence over the stream-group rate on newer firmware.
    msg_rates = {
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: rate_hz,
        mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT:           1.0,
    }
    for msg_id, hz in msg_rates.items():
        request_message_interval(mav, msg_id, hz)

    print(f"[PLND] Telemetry streams requested @ {rate_hz} Hz. "
          "Waiting for first GLOBAL_POSITION_INT ...")

    # Confirm we're actually receiving position messages before proceeding
    msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
    if msg is None:
        print("[PLND] WARNING: no GLOBAL_POSITION_INT received after 5 s. "
              "Check SR* params on the flight controller "
              "(SR0_POSITION, SR1_POSITION …) or try a higher rate.")
    else:
        print(f"[PLND] Receiving position. Alt={msg.relative_alt / 1000:.1f} m")


def connect(connection_string: str, baud: int) -> mavutil.mavfile:
    """Open MAVLink connection, wait for heartbeat, then request telemetry."""
    print(f"[PLND] Connecting to {connection_string} ...")
    mav = mavutil.mavlink_connection(connection_string, baud=baud)
    mav.wait_heartbeat()
    print(f"[PLND] Heartbeat received from system {mav.target_system}, "
          f"component {mav.target_component}")
    request_data_streams(mav, rate_hz=4.0)
    return mav


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
    Emit a LANDING_TARGET MAVLink message.

    Parameters
    ----------
    angle_x  : lateral angle to target (rad, + = right)
    angle_y  : longitudinal angle to target (rad, + = forward-down)
    distance : slant range to target (m)
    size_x   : angular width of target (rad)  – optional
    size_y   : angular height of target (rad) – optional
    """
    mav.mav.landing_target_send(
        int(time.monotonic() * 1e6),   # time_usec
        0,                              # target_num
        mavutil.mavlink.MAV_FRAME_BODY_FRD,  # frame (body forward-right-down)
        angle_x,
        angle_y,
        distance,
        size_x,
        size_y,
    )


def arm_and_takeoff(mav: mavutil.mavfile, altitude_m: float) -> None:
    """Switch to GUIDED, arm, and takeoff to the requested altitude."""
    print(f"[PLND] Switching to GUIDED mode ...")
    mav.set_mode("GUIDED")
    time.sleep(1)

    print("[PLND] Arming ...")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    print("[PLND] Armed.")

    print(f"[PLND] Taking off to {altitude_m} m ...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,           # confirmation
        0, 0, 0, 0,  # params 1-4 (unused for copter)
        0, 0,        # lat, lon (unused)
        altitude_m,  # altitude (m)
    )
    # Wait until the vehicle reaches the target altitude
    while True:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        #msg = mav.recv_match(blocking=True, timeout=2)
        #if msg is not None: print(msg)
        if msg: print(f"[PLND] Taking off... Current altitude: {msg.relative_alt / 1000:.1f}m")
        if msg and msg.relative_alt / 1000.0 >= altitude_m * 0.92:
            print(f"[PLND] Reached {msg.relative_alt / 1000:.1f} m. Proceeding.")
            time.sleep(1)
            break


def switch_to_land(mav: mavutil.mavfile) -> None:
    """Command the vehicle into LAND mode so PLND kicks in."""
    print("[PLND] Switching to LAND mode – precision landing active.")
    mav.set_mode("LAND")


def wait_for_disarm(mav: mavutil.mavfile, timeout_s: float = 120) -> bool:
    """Block until the vehicle disarms (landed) or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg:
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if not armed:
                print("[PLND] Vehicle disarmed – landing complete.")
                return True
    print("[PLND] Timeout waiting for disarm.")
    return False


# ---------------------------------------------------------------------------
# Camera-to-body rotation matrix
# ---------------------------------------------------------------------------

def build_cam_to_body_rotation(
    roll_deg: float = 0.0,
    pitch_deg: float = -90.0,   # typical downward-facing camera
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a rotation matrix that transforms vectors from the camera frame
    (X right, Y down, Z forward) to the vehicle body frame (X fwd, Y right,
    Z down), given the camera's mount orientation in roll/pitch/yaw (degrees).

    Defaults represent a camera mounted flat, pointing straight down.
    Adjust for your physical mount.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    Rz = np.array([[ math.cos(y), -math.sin(y), 0],
                   [ math.sin(y),  math.cos(y), 0],
                   [           0,            0, 1]])
    Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                   [           0, 1,           0],
                   [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1,           0,            0],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r),  math.cos(r)]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Precision land loop
# ---------------------------------------------------------------------------

def precision_land_loop(
        mav: mavutil.mavfile,
        cam_to_body_R: np.ndarray,
        loop_hz: float = 10.0,
        send_target:bool = True):
        
    period = 1.0 / loop_hz

    target_visible_prev = False

    while True:
        t0 = time.monotonic()

        result = process_frame()

        if result is not None:
            tvec_cam, rvec = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)
            angle_x, angle_y, dist = tvec_to_angles(tvec_body)
            if send_target: send_landing_target(mav, angle_x, angle_y, dist)

            if not target_visible_prev:
                print(f"[PLND] Target acquired – dist={dist:.2f} m, "
                      f"ax={math.degrees(angle_x):.1f}°, "
                      f"ay={math.degrees(angle_y):.1f}°")
            target_visible_prev = True
        else:
            if target_visible_prev:
                print("[PLND] Target lost – no message sent this frame.")
            target_visible_prev = False

        # Check for disarm (vehicle has landed)
        hb = mav.recv_match(type="HEARTBEAT", blocking=False)
        if hb:
            if hb.type == 2:
                armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if not armed:
                    print("[PLND] Disarmed – precision landing complete.")
                    return

        # Pace the loop
        elapsed = time.monotonic() - t0
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

# ---------------------------------------------------------------------------
# Velocity-based guidance (GUIDED mode, capped speed)
# ---------------------------------------------------------------------------

# Bitmask for SET_POSITION_TARGET_LOCAL_NED:
# ignore pos x/y/z, ignore acc x/y/z, ignore yaw & yaw-rate → velocity only
_VEL_ONLY_MASK = (
    0b0000_111111000111  # ignore pos(0-2), acc(6-8), yaw(10), yaw-rate(11)
)


def send_velocity_ned(
    mav: mavutil.mavfile,
    vx: float,
    vy: float,
    vz: float = 0.0,
) -> None:
    """
    Command a body-NED velocity setpoint in GUIDED mode.

    Parameters
    ----------
    vx : velocity north  (m/s, positive = north)
    vy : velocity east   (m/s, positive = east)
    vz : velocity down   (m/s, positive = down – keep 0 during lateral approach)
    """
    mav.mav.set_position_target_local_ned_send(
        int(time.monotonic() * 1e3),        # time_boot_ms
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        _VEL_ONLY_MASK,
        0, 0, 0,        # pos x/y/z – ignored
        vx, vy, vz,     # velocity setpoint
        0, 0, 0,        # acc x/y/z – ignored
        0, 0,           # yaw, yaw_rate – ignored
    )


def tvec_body_to_ned_velocity(
    tvec_body: np.ndarray,
    heading_rad: float,
    max_speed_mps: float = 0.10,
) -> Tuple[float, float]:
    """
    Convert the lateral offset to the target (body frame) into a NED velocity
    command, capped at max_speed_mps.

    The vector is proportional to the horizontal offset so the drone slows
    down as it gets closer.  A P-gain of 1.0 means 1 m offset → 1 m/s, but
    the clamp prevents it from ever exceeding max_speed_mps.

    Parameters
    ----------
    tvec_body   : (3,) [fwd, right, down] in metres (body frame, FRD)
    heading_rad : current vehicle yaw in radians (from ATTITUDE or VFR_HUD)
    max_speed_mps : velocity clamp (default 0.10 m/s = 10 cm/s)

    Returns
    -------
    (vx_ned, vy_ned) : velocity north and east in m/s
    """
    # Lateral offset in body frame (forward, right) – ignore altitude
    dx_body = tvec_body[0]   # forward offset to target
    dy_body = tvec_body[1]   # rightward offset to target

    # P-controller: desired velocity proportional to offset (gain = 1.0)
    # Increase gain to move faster per metre of error; lower to creep more.
    P_GAIN = 1.0
    vx_body = P_GAIN * dx_body
    vy_body = P_GAIN * dy_body

    # Clamp magnitude to max_speed_mps
    speed = math.hypot(vx_body, vy_body)
    if speed > max_speed_mps:
        scale = max_speed_mps / speed
        vx_body *= scale
        vy_body *= scale

    # Rotate from body frame to NED using vehicle heading
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    vx_ned =  cos_h * vx_body - sin_h * vy_body   # north
    vy_ned =  sin_h * vx_body + cos_h * vy_body   # east

    return vx_ned, vy_ned


def get_heading_rad(mav: mavutil.mavfile) -> float:
    """Return the latest vehicle yaw in radians, or 0.0 if unavailable."""
    msg = mav.recv_match(type="VFR_HUD", blocking=False)
    if msg:
        return math.radians(msg.heading)
    msg = mav.recv_match(type="ATTITUDE", blocking=False)
    if msg:
        return msg.yaw
    return 0.0

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def precision_land(
    mav: mavutil.mavfile,
    cam_to_body_R: np.ndarray,
    loop_hz: float = 10.0,
    takeoff_alt_m: float = 5.0,
    land_timeout_s: float = 120.0,
) -> None:
    """
    Full precision-landing sequence:
      1. Arm & takeoff (skip if vehicle is already airborne)
      2. Stream LANDING_TARGET messages at loop_hz
      3. Switch to LAND mode
      4. Keep streaming until disarmed
    """

    # --- Takeoff -----------------------------------------------------------
    if DO_TAKEOFF: arm_and_takeoff(mav, takeoff_alt_m)

    # --- Switch to LAND mode (PLND activates here) -------------------------
    if DO_LAND: switch_to_land(mav)

    # --- Streaming loop ----------------------------------------------------
    print(f"[PLND] Streaming LANDING_TARGET @ {loop_hz} Hz ...")
    precision_land_loop(mav, cam_to_body_R, loop_hz, DO_LAND)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArduPilot PLND precision landing")
    p.add_argument(
        "--connect", default="udpin:0.0.0.0:14550",
        help="MAVLink connection string (default: udpin:0.0.0.0:14550)",
    )
    p.add_argument("--baud", type=int, default=57600,
                   help="Serial baud rate (ignored for UDP/TCP)")
    p.add_argument("--alt", type=float, default=5.0,
                   help="Takeoff altitude in metres (default: 5)")
    p.add_argument("--hz", type=float, default=10.0,
                   help="Target-message rate in Hz (default: 10)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Max seconds to wait for landing (default: 120)")
    p.add_argument("--cam-pitch", type=float, default=-90.0,
                   help="Camera mount pitch in degrees (default: -90 = straight down)")
    p.add_argument("--cam-roll", type=float, default=0.0,
                   help="Camera mount roll in degrees (default: 0)")
    p.add_argument("--cam-yaw", type=float, default=0.0,
                   help="Camera mount yaw in degrees (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cam_to_body_R = build_cam_to_body_rotation(
        roll_deg=args.cam_roll,
        pitch_deg=args.cam_pitch,
        yaw_deg=args.cam_yaw,
    )

    mav = connect(args.connect, args.baud)

    try:
        precision_land(
            mav,
            cam_to_body_R,
            loop_hz=args.hz,
            takeoff_alt_m=args.alt,
            land_timeout_s=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n[PLND] Interrupted by user.")
        print("[PLND] Switching to LAND mode for safety ...")
        switch_to_land(mav)
    finally:
        mav.close()
        print("[PLND] Connection closed.")


if __name__ == "__main__":
    main()

    '''

    args = parse_args()

    cam_to_body_R = build_cam_to_body_rotation(
        roll_deg=args.cam_roll,
        pitch_deg=args.cam_pitch,
        yaw_deg=args.cam_yaw,
    )

    while True:
        result = process_frame()
        
        print(f"[TEST] Result: {result}")
        if result is not None:
            tvec_cam, rvec = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)

            angle_x, angle_y, dist = tvec_to_angles(tvec_body)
            print(f"[TEST] TVEC BODY: {tvec_body}\nX = {math.degrees(angle_x):.1f}º, Y = {math.degrees(angle_y):.1f}º, dist = {dist:.1f}m")
        
    '''