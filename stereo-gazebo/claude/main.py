"""
main.py – ArduPilot Precision Landing entry point
==================================================
Orchestrates the two-phase landing sequence:

  Phase 1 (GUIDED) – slow lateral approach
      The drone moves toward the target at a capped speed while
      LANDING_TARGET messages prime ArduPilot's PLND EKF.

  Phase 2 (LAND)   – ArduPilot PLND descent
      Once the drone is directly overhead, LAND mode is engaged.
      LANDING_TARGET messages continue so PLND corrects drift on descent.

ArduPilot parameters required on the vehicle:
    PLND_ENABLED  = 1
    PLND_TYPE     = 2    (MAVLink / companion computer)
    PLND_EST_TYPE = 1    (raw sensor; AP runs the EKF)

Usage:
    python main.py --connect udpin:0.0.0.0:14550
    python main.py --connect /dev/ttyUSB0 --baud 57600 --alt 10 --max-speed 0.05
"""

import argparse
import math
import time

from plnd.guidance import cam_lateral_to_ned_velocity
from plnd.mavlink_bridge import (
    arm_and_takeoff,
    connect,
    get_heading_rad,
    send_landing_target,
    send_velocity_ned,
    switch_to_land,
    send_pos_frd,
    send_velocity_frd,
    move_relative,
    send_landing_target_pos_quat
)
from plnd.transforms import (
    build_cam_to_body_rotation, 
    cam_to_body, 
    tvec_to_angles,
    rvec_cam_to_body_quaternion
)
from plnd.aruco_lib import *

from typing import Optional, Tuple

import cv2
import numpy as np

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
        #print("ONLY WIDE DETECTION")
        rvec, tvec = rvec_wide, tvec_wide

    elif rvec_narr is not None:
        #print("ONLY NARROW DETECTION")
        rvec, tvec = rvec_narr, tvec_narr
    
    cv2.waitKey(30)

    if rvec is not None and tvec is not None:
        return tvec, rvec
    else: return None

# ---------------------------------------------------------------------------
# Landing sequence
# ---------------------------------------------------------------------------

def precision_land(
    mav,
    cam_to_body_R,
    *,
    loop_hz: float = 10.0,
    takeoff_alt_m: float = 5.0,
    land_timeout_s: float = 120.0,
    max_lateral_speed_mps: float = 0.10,
    overhead_threshold_m: float = 0.15,
) -> None:
    """
    Run the full two-phase precision landing sequence.

    Parameters
    ----------
    mav                   : open pymavlink mavfile connection
    cam_to_body_R         : (3×3) camera-to-body rotation matrix
    loop_hz               : control loop rate
    takeoff_alt_m         : climb altitude before starting the approach
    land_timeout_s        : max seconds to wait for disarm in phase 2
    max_lateral_speed_mps : velocity cap during the GUIDED approach
    overhead_threshold_m  : horizontal error (m) that triggers LAND handoff
    """
    period = 1.0 / loop_hz

    # ------------------------------------------------------------------ #
    # Phase 1: arm, climb, then creep laterally toward the target         #
    # ------------------------------------------------------------------ #
    arm_and_takeoff(mav, takeoff_alt_m)

    print(f"[MAIN] Phase 1 – GUIDED approach "
          f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")
    target_visible_prev = False

    while True:
        t0 = time.monotonic()

        result = process_frame()

        if result is not None:
            tvec_cam, _rvec = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)
            angle_x, angle_y, dist = tvec_to_angles(tvec_body)

            # Keep AP's PLND EKF warm throughout the approach
            send_landing_target(mav, angle_x, angle_y, dist)

            lateral_err = math.hypot(tvec_body[0], tvec_body[1])

            if not target_visible_prev:
                print(f"[MAIN] Target acquired – lateral err={lateral_err:.3f} m")
            target_visible_prev = True

            if lateral_err <= overhead_threshold_m:
                print(f"[MAIN] Overhead ({lateral_err:.3f} m ≤ "
                      f"{overhead_threshold_m} m) – handing off to LAND.")
                break

            # Velocity command toward target, capped at max speed
            heading = get_heading_rad(mav)
            vx, vy = tvec_body_to_ned_velocity(
                tvec_body, heading, max_lateral_speed_mps
            )
            print(vx, vy)
            send_velocity_ned(mav, vx, vy)

        else:
            if target_visible_prev:
                print("[MAIN] Target lost – holding position.")
            send_velocity_ned(mav, 0.0, 0.0)   # stop lateral drift
            target_visible_prev = False

        _pace(t0, period)

    # ------------------------------------------------------------------ #
    # Phase 2: LAND mode – ArduPilot PLND descends onto the pad           #
    # ------------------------------------------------------------------ #
    switch_to_land(mav)
    print("[MAIN] Phase 2 – LAND mode, streaming LANDING_TARGET until disarmed ...")
    from pymavlink import mavutil as _mu

    while True:
        t0 = time.monotonic()

        result = process_frame()
        if result is not None:
            tvec_cam, _rvec = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)
            angle_x, angle_y, dist = tvec_to_angles(tvec_body)
            send_landing_target(mav, angle_x, angle_y, dist)

        hb = mav.recv_match(type="HEARTBEAT", blocking=False)
        if hb:
            armed = bool(hb.base_mode & _mu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if not armed:
                print("[MAIN] Disarmed – precision landing complete.")
                return

        _pace(t0, period)

def precision_land_pos_quat(
    mav,
    cam_to_body_R,
    *,
    loop_hz: float = 10.0,
    takeoff_alt_m: float = 5.0,
    land_timeout_s: float = 120.0,
    max_lateral_speed_mps: float = 0.10,
    overhead_threshold_m: float = 0.15,
) -> None:
    """
    Two-phase precision landing using the LANDING_TARGET position + quaternion
    extended fields (MAVLink 2, position_valid = 1).

    Differences from precision_land()
    -----------------------------------
    - Sends target position [x, y, z] in body FRD frame instead of angles.
    - Sends target orientation as a quaternion derived from the rvec returned
      by process_frame(), rotated into the body frame.
    - Sets position_valid = 1 so ArduPilot uses the position directly.

    Recommended ArduPilot parameter
    ---------------------------------
    PLND_EST_TYPE = 0   → use position as-is  (best when your estimator is good)
    PLND_EST_TYPE = 1   → run through AP's EKF (adds smoothing at the cost of lag)

    The two phases (GUIDED lateral approach → LAND descent) are identical to
    precision_land(); only the LANDING_TARGET message format changes.

    Parameters
    ----------
    mav                   : open pymavlink mavfile connection
    cam_to_body_R         : (3×3) camera-to-body rotation matrix
    loop_hz               : control loop rate (Hz)
    takeoff_alt_m         : climb altitude before approach (m)
    land_timeout_s        : max seconds to wait for disarm in phase 2
    max_lateral_speed_mps : velocity cap during the GUIDED approach (m/s)
    overhead_threshold_m  : horizontal error (m) that triggers LAND handoff
    """
    period = 1.0 / loop_hz

    # ------------------------------------------------------------------ #
    # Phase 1: arm, climb, creep laterally toward the target              #
    # ------------------------------------------------------------------ #
    arm_and_takeoff(mav, takeoff_alt_m)

    print(f"[MAIN] Phase 1 – GUIDED approach [pos+quat] "
          f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")
    target_visible_prev = False

    while True:
        t0 = time.monotonic()

        result = process_frame()

        if result is not None:
            tvec_cam, rvec_cam = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)
            q_body    = rvec_cam_to_body_quaternion(rvec_cam, cam_to_body_R)

            # Prime AP's PLND EKF with position + quaternion
            send_landing_target_pos_quat(mav, tvec_body, q_body)

            # Camera-frame lateral guidance – immune to altitude bleed-through
            heading = get_heading_rad(mav)
            vx, vy, lateral_err = cam_lateral_to_ned_velocity(
                tvec_cam, cam_to_body_R, heading, max_lateral_speed_mps
            )

            if not target_visible_prev:
                print(f"[MAIN] Target acquired – lateral err={lateral_err:.3f} m, "
                      f"q=[{q_body[0]:.3f}, {q_body[1]:.3f}, "
                      f"{q_body[2]:.3f}, {q_body[3]:.3f}]")
            target_visible_prev = True

            if lateral_err <= overhead_threshold_m:
                print(f"[MAIN] Overhead ({lateral_err:.3f} m ≤ "
                      f"{overhead_threshold_m} m) – handing off to LAND.")
                break

            send_velocity_ned(mav, vx, vy)

        else:
            if target_visible_prev:
                print("[MAIN] Target lost – holding position.")
            send_velocity_ned(mav, 0.0, 0.0)
            target_visible_prev = False

        _pace(t0, period)

    # ------------------------------------------------------------------ #
    # Phase 2: LAND mode – ArduPilot PLND descends onto the pad           #
    # ------------------------------------------------------------------ #
    switch_to_land(mav)
    print("[MAIN] Phase 2 – LAND mode [pos+quat], streaming until disarmed ...")
    from pymavlink import mavutil as _mu

    while True:
        t0 = time.monotonic()

        result = process_frame()
        if result is not None:
            tvec_cam, rvec_cam = result
            tvec_body = cam_to_body(tvec_cam, cam_to_body_R)
            q_body    = rvec_cam_to_body_quaternion(rvec_cam, cam_to_body_R)
            send_landing_target_pos_quat(mav, tvec_body, q_body)

        hb = mav.recv_match(type="HEARTBEAT", blocking=False)
        if hb:
            armed = bool(hb.base_mode & _mu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if not armed:
                print("[MAIN] Disarmed – precision landing complete.")
                return

        _pace(t0, period)


def _pace(t0: float, period: float) -> None:
    """Sleep for the remainder of the loop period."""
    remaining = period - (time.monotonic() - t0)
    if remaining > 0:
        time.sleep(remaining)
        

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ArduPilot PLND precision landing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--connect", default="udpin:0.0.0.0:14550",
                   help="MAVLink connection string")
    p.add_argument("--baud", type=int, default=57600,
                   help="Serial baud rate (ignored for UDP/TCP)")
    p.add_argument("--alt", type=float, default=5.0,
                   help="Takeoff altitude (m)")
    p.add_argument("--hz", type=float, default=10.0,
                   help="Control loop / message rate (Hz)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Max seconds to wait for landing disarm")
    p.add_argument("--max-speed", type=float, default=0.10,
                   help="Max lateral approach speed (m/s)")
    p.add_argument("--overhead-threshold", type=float, default=0.15,
                   help="Horizontal offset (m) that triggers LAND handoff")
    p.add_argument("--cam-roll",  type=float, default=0.0,
                   help="Camera mount roll (deg)")
    p.add_argument("--cam-pitch", type=float, default=-90.0,
                   help="Camera mount pitch (deg); -90 = straight down")
    p.add_argument("--cam-yaw",   type=float, default=0.0,
                   help="Camera mount yaw (deg)")
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
        precision_land_pos_quat(
            mav,
            cam_to_body_R,
            loop_hz=args.hz,
            takeoff_alt_m=args.alt,
            land_timeout_s=args.timeout,
            max_lateral_speed_mps=args.max_speed,
            overhead_threshold_m=args.overhead_threshold,
        )
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted – switching to LAND for safety.")
        switch_to_land(mav)
    finally:
        mav.close()
        print("[MAIN] Connection closed.")


if __name__ == "__main__":
    main()
