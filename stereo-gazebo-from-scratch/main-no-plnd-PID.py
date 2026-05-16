##### TO WATCH!!!!! https://landmarklanding.com/blogs/landmark-lab-notes/ardupilot-precision-landing?srsltid=AfmBOopRb_8qfYKOU5efXQ0mQedGTP3hKf4MeQ8MP-QD6iZekGplSPor

import cv2
import numpy as np
import argparse

from typing import Optional, Tuple
import time
import math

import os; os.environ["MAVLINK20"] = "1"

from lib.drone_lib import *
from lib.aruco_lib import *
from lib.camera_lib import *
from lib.utils import *
from lib.pid_controller import PIDController


DO_LAND = True
DO_TAKEOFF = True

MARKER_SIZE = 0.48
MARKER_ID = 49

TAKEOFF_ALT = 7

PHASE_1_THRSH_DIST = 1        # Lateral distance at which PHASE 1 will finish
PHASE_1_DIST_VEL_MULT = 0.2     # Multiplier to distance to calculate speed (speed = distance*MULT)
PHASE_1_DIST_VEL_MAX = 0.5

PHASE_2_DIST_VEL_MAX = 0.1
PHASE_2_THRSH_ALT = 0.5
PHASE_2_DESCENT = 0.2
PHASE_2_DIST_VEL_MULT = 0.2

# ---------------------------------------------------------------------------
# Phase 2 PID gains  (tune these!)
# ---------------------------------------------------------------------------
PID_KP = 0.50           # Proportional gain  (m/s per m of offset)
PID_KI = 0.05           # Integral gain      (m/s per m·s of accumulated error)
PID_KD = 0.20           # Derivative gain    (m/s per m/s of offset change)
PID_INTEGRAL_MAX = 0.4  # Anti-windup clamp
PID_DERIV_ALPHA = 0.3   # Derivative EMA filter (0=smooth, 1=raw)

PHASE_2_CENTERED_THRSH = 0.15   # Lateral offset (m) considered "centered"
PHASE_2_TARGET_LOST_TIMEOUT = 3.0  # Seconds without detection before hover

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

mtx_wide = get_gazebo_camera_matrix(1536, 864, 102/180*np.pi, 48.8/180*np.pi)
dst_wide = np.zeros(5, dtype=np.float32)
mtx_narr = get_gazebo_camera_matrix(1640, 1232, 62.2/180*np.pi, 67/180*np.pi)
dst_narr = np.zeros(5, dtype=np.float32)

# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------

def approach(mav, cameras, max_lateral_speed_mps, lateral_speed_mult, overhead_threshold, loop_hz, do_display):
    try:
        print(f"[APPR] Phase 1 – Initial approach "
            f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")

        # PHASE 1: Locate and initial approach (P-only, unchanged)
        target_visible_prev = False
        while True:
            t0 = time.monotonic()

            result = cameras.get_latest_pose()

            if do_display:
                display_frame = cameras.get_latest_frame()
                if display_frame is not None:
                    cv2.imshow("img", display_frame)
                    cv2.waitKey(1)

            if result is not None:
                tvec, rvec = result

                right = float(tvec[0])
                fwd = -float(tvec[1])
                down = float(tvec[2])

                distance = float(np.linalg.norm((fwd, right)))

                mav.send_landing_target_pos_quat(fwd, right, down)

                if not target_visible_prev:
                    print(f"[APPR] Target acquired - Distance ({fwd:.1f}, {right:.1f}, {down:.1f}) m")
                target_visible_prev = True

                if distance < overhead_threshold:
                    print(f"[APPR] Target at {distance:.2f}m - Approach complete")
                    break

                else:
                    print(f"[APPR] Target at {distance:.2f}m - Approaching...")
                    v_fwd = fwd * lateral_speed_mult
                    v_right = right * lateral_speed_mult

                    current_speed = float(np.linalg.norm((v_fwd, v_right)))

                    if current_speed > max_lateral_speed_mps:
                        v_fwd = (v_fwd / current_speed) * max_lateral_speed_mps
                        v_right = (v_right / current_speed) * max_lateral_speed_mps
                        current_speed = max_lateral_speed_mps

                    print(f"[APPR] Target at {distance:.2f}m - Approaching at ({v_fwd:.2f}, {v_right:.2f}, 0) m/s -> {current_speed:.2f} m/s...")
                    mav.move_velocity_body(v_fwd, v_right, 0)

            _pace(t0, 1/loop_hz)

        return 0

    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.switch_to_land()
        cameras.stop()

        return -1

def descent(mav, cameras, max_lateral_speed_mps, descent_speed, threshold_alt, loop_hz, do_display):
    """
    Phase 2 – PID-controlled descent.
    """
    try:
        print(f"[DSCT] Phase 2 – PID descent "
            f"(max lateral {max_lateral_speed_mps * 100:.0f} cm/s, "
            f"descent {descent_speed:.1f} m/s) ...")

        # ── Create PID controllers ──
        pid_fwd = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            setpoint=0.0,
            output_min=-max_lateral_speed_mps, output_max=max_lateral_speed_mps,
            integral_max=PID_INTEGRAL_MAX,
            derivative_filter_alpha=PID_DERIV_ALPHA,
        )
        pid_right = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            setpoint=0.0,
            output_min=-max_lateral_speed_mps, output_max=max_lateral_speed_mps,
            integral_max=PID_INTEGRAL_MAX,
            derivative_filter_alpha=PID_DERIV_ALPHA,
        )

        alt = -1
        distance = -1
        last_seen_time = time.monotonic()

        # PHASE 2: PID descent
        while True:
            t0 = time.monotonic()

            result = cameras.get_latest_pose()

            msg = mav.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            if msg:
                alt = msg.relative_alt / 1000.0

                if alt != -1 and alt < threshold_alt:
                    print(f"[DSCT] Height: {alt:.2f}m, target at {distance:.2f}m - Descent complete")
                    mav.move_velocity_body(0, 0, 0)
                    break

            if do_display:
                display_frame = cameras.get_latest_frame()
                if display_frame is not None:
                    cv2.imshow("img", display_frame)
                    cv2.waitKey(1)

            if result is not None:
                tvec, rvec = result

                right = float(tvec[0])
                fwd = -float(tvec[1])
                down = float(tvec[2])

                distance = float(np.linalg.norm((fwd, right)))
                last_seen_time = time.monotonic()

                mav.send_landing_target_pos_quat(fwd, right, down)

                # Lateral Corrections
                v_fwd = pid_fwd.update(fwd)
                v_right = pid_right.update(right)

                # Clamp combined lateral speed (diagonal can exceed per-axis max)
                lateral_speed = math.sqrt(v_fwd**2 + v_right**2)
                if lateral_speed > max_lateral_speed_mps:
                    scale = max_lateral_speed_mps / lateral_speed
                    v_fwd *= scale
                    v_right *= scale
                    lateral_speed = max_lateral_speed_mps

                # Adaptive descent rate
                if distance < PHASE_2_CENTERED_THRSH:
                    v_down = descent_speed
                elif distance < PHASE_1_THRSH_DIST:
                    ratio = 1.0 - 0.7 * (distance / PHASE_1_THRSH_DIST)
                    v_down = descent_speed * ratio
                else:
                    v_down = 0.0

                print(f"[DSCT] offset=({fwd:+.2f}, {right:+.2f}) m  "
                      f"vel=({v_fwd:+.3f}, {v_right:+.3f}, {v_down:+.2f}) m/s  "
                      f"alt={alt:.1f} m  dist={distance:.2f} m")

                mav.move_velocity_body(v_fwd, v_right, v_down)

            else:
                # Target lost
                time_lost = time.monotonic() - last_seen_time
                if time_lost > PHASE_2_TARGET_LOST_TIMEOUT:
                    print(f"[DSCT] Target lost for {time_lost:.1f}s — hovering...")
                    pid_fwd.reset()
                    pid_right.reset()
                mav.move_velocity_body(0, 0, 0)

            _pace(t0, 1/loop_hz)

        return 0

    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.switch_to_land()
        cameras.stop()

        return -1


# ---------------------------------------------------------------------------
# Additional functions
# ---------------------------------------------------------------------------

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
        description="ArduPilot precision landing (no PLND) with PID lateral control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--connect", default="udpin:0.0.0.0:14550",
                   help="MAVLink connection string")
    p.add_argument("--baud", type=int, default=57600,
                   help="Serial baud rate (ignored for UDP/TCP)")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help="Takeoff altitude (m)")
    p.add_argument("--hz", type=float, default=10.0,
                   help="Control loop / message rate (Hz)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Max seconds to wait for landing disarm")
    p.add_argument("--p1_max_speed", type=float, default=PHASE_1_DIST_VEL_MAX,
                   help="Max lateral approach speed (m/s) in approach")
    p.add_argument("--p1_mult_speed", type=float, default=PHASE_1_DIST_VEL_MULT,
                   help="Lateral approach speed multiplier (m/s) in approach")
    p.add_argument("--p1_threshold", type=float, default=PHASE_1_THRSH_DIST,
                   help="Horizontal offset (m) that triggers descent handoff")
    p.add_argument("--p2_max_speed", type=float, default=PHASE_2_DIST_VEL_MAX,
                   help="Max lateral speed (m/s) in descent")
    p.add_argument("--p2_descent", type=float, default=PHASE_2_DESCENT,
                   help="Max descent speed (m/s)")
    p.add_argument("--p2_threshold", type=float, default=PHASE_2_THRSH_ALT,
                   help="Threshold altitude (m) that triggers LAND handoff")
    p.add_argument("--cam-roll",  type=float, default=0.0,
                   help="Camera mount roll (deg)")
    p.add_argument("--cam-pitch", type=float, default=-90.0,
                   help="Camera mount pitch (deg); -90 = straight down")
    p.add_argument("--cam-yaw",   type=float, default=0.0,
                   help="Camera mount yaw (deg)")
    p.add_argument("--display", action="store_true", help="If set, will display the view of the cameras")

    # PID tuning overrides
    p.add_argument("--pid-kp", type=float, default=PID_KP,
                   help="PID proportional gain")
    p.add_argument("--pid-ki", type=float, default=PID_KI,
                   help="PID integral gain")
    p.add_argument("--pid-kd", type=float, default=PID_KD,
                   help="PID derivative gain")
    return p.parse_args()



def main():
    args = parse_args()

    # Apply CLI PID overrides
    global PID_KP, PID_KI, PID_KD
    PID_KP = args.pid_kp
    PID_KI = args.pid_ki
    PID_KD = args.pid_kd

    print("[SYSTEM] Initializing cameras...")
    cameras = GazeboStereoCapture(pipeline_wide, pipeline_narr, mtx_wide, dst_wide, mtx_narr, dst_narr)
    cameras.start()

    mav = MavlinkConnection(args.connect, args.baud)

    mav.arm_and_takeoff(args.alt)

    if approach(mav, cameras, args.p1_max_speed, args.p1_mult_speed, args.p1_threshold, args.hz, args.display) < 0: return -1

    if descent(mav, cameras, args.p2_max_speed, args.p2_descent, args.p2_threshold, args.hz, args.display) < 0: return -1

    mav.switch_to_land()

if __name__ == "__main__":
    main()