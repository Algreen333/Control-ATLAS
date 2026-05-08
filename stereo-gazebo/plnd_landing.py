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

# ---------------------------------------------------------------------------
# Stub – replace with your real implementation
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
    fwd, right, down = tvec_body
    distance = float(np.linalg.norm(tvec_body))

    if distance < 1e-6:
        return 0.0, 0.0, 0.0

    # Horizontal angle (+ = target right of vehicle)
    angle_x = math.atan2(right, fwd)
    # Vertical angle (+ = target below / camera looking down more)
    angle_y = math.atan2(-down, fwd)   # negative because down is positive Z

    return angle_x, angle_y, distance


# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def connect(connection_string: str, baud: int) -> mavutil.mavfile:
    """Open MAVLink connection and wait for heartbeat."""
    print(f"[PLND] Connecting to {connection_string} ...")
    mav = mavutil.mavlink_connection(connection_string, baud=baud)
    mav.wait_heartbeat()
    print(f"[PLND] Heartbeat received from system {mav.target_system}, "
          f"component {mav.target_component}")
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
        if msg and msg.relative_alt / 1000.0 >= altitude_m * 0.92:
            print(f"[PLND] Reached {msg.relative_alt / 1000:.1f} m. Proceeding.")
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
    period = 1.0 / loop_hz

    # --- Takeoff -----------------------------------------------------------
    arm_and_takeoff(mav, takeoff_alt_m)

    # --- Switch to LAND mode (PLND activates here) -------------------------
    switch_to_land(mav)

    # --- Streaming loop ----------------------------------------------------
    print(f"[PLND] Streaming LANDING_TARGET @ {loop_hz} Hz ...")
    target_visible_prev = False

    while True:
        t0 = time.monotonic()

        result = process_frame()

        if result is not None:
            tvec_cam, rvec = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)
            angle_x, angle_y, dist = tvec_to_angles(tvec_body)
            send_landing_target(mav, angle_x, angle_y, dist)

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
