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

MARKER_SIZE = 1
MARKER_ID = 49

# ---------------------------------------------------------------------------
# Phase 1 – Initial lateral approach (P-only, no descent)
# ---------------------------------------------------------------------------
PHASE_1_THRSH_DIST = 1           # Lateral distance (m) at which Phase 1 finishes
PHASE_1_DIST_VEL_MULT = 0.2     # speed = distance * MULT

# ---------------------------------------------------------------------------
# Phase 2 – PID-controlled lateral hold + controlled descent
# ---------------------------------------------------------------------------
# PID gains for lateral axes (fwd and right).  Start conservative:
#   Kp drives toward the marker, Ki removes steady-state offset from wind,
#   Kd dampens overshoot/oscillation.
PID_KP = 0.50         
PID_KI = 0.05          
PID_KD = 0.20          

PID_OUTPUT_MAX = 0.3   # Max lateral correction speed (m/s)
PID_INTEGRAL_MAX = 0.4 # Anti-windup clamp on integrator (m·s)
PID_DERIV_ALPHA = 0.3  # Derivative EMA filter (0=smooth, 1=raw)

PHASE_2_DESCENT_RATE = 0.3      # Constant descent speed (m/s, positive = down in NED)
PHASE_2_HANDOFF_ALT = 1.5       # Altitude (m) at which we hand off to LAND mode
PHASE_2_CENTERED_THRSH = 0.15   # Lateral offset (m) under which we consider "centered"
PHASE_2_TARGET_LOST_TIMEOUT = 3.0  # Seconds without detection before hover-and-wait

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

def precision_land(mav, cameras, max_lateral_speed_mps, loop_hz, overhead_threshold, do_display):
    try:
        # FASE 1: Locate and initial lateral approach (P-only)
        print(f"[APPR] Phase 1 – Initial approach "
            f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")

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

                print(fwd, right, down)
                mav.send_landing_target_pos_quat(fwd, right, down)

                if not target_visible_prev:
                    print(f"[APPR] Target acquired - Distance ({fwd:.1f}, {right:.1f}, {down:.1f}) m")
                target_visible_prev = True

                if distance < overhead_threshold:
                    print(f"[APPR] Phase 1 complete – lateral offset {distance:.2f} m")
                    break

                else:
                    v_fwd = fwd * PHASE_1_DIST_VEL_MULT
                    v_right = right * PHASE_1_DIST_VEL_MULT

                    current_speed = float(np.linalg.norm((v_fwd, v_right)))

                    if current_speed > max_lateral_speed_mps:
                        v_fwd = (v_fwd / current_speed) * max_lateral_speed_mps
                        v_right = (v_right / current_speed) * max_lateral_speed_mps
                        current_speed = max_lateral_speed_mps

                    print(f"[APPR] Target at {distance:.2f}m - Approaching at ({v_fwd:.3f}, {v_right:.3f}, 0) m/s -> {current_speed:.2f} m/s...")
                    mav.move_velocity_body(v_fwd, v_right, 0)

            _pace(t0, 1/loop_hz)

        # FASE 2: PID-controlled lateral hold + controlled descent
        print(f"[DESC] Phase 2 – PID descent  "
              f"(descent {PHASE_2_DESCENT_RATE:.1f} m/s, "
              f"handoff at {PHASE_2_HANDOFF_ALT:.1f} m)")

        pid_fwd = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            setpoint=0.0,
            output_min=-PID_OUTPUT_MAX, output_max=PID_OUTPUT_MAX,
            integral_max=PID_INTEGRAL_MAX,
            derivative_filter_alpha=PID_DERIV_ALPHA,
        )
        pid_right = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            setpoint=0.0,
            output_min=-PID_OUTPUT_MAX, output_max=PID_OUTPUT_MAX,
            integral_max=PID_INTEGRAL_MAX,
            derivative_filter_alpha=PID_DERIV_ALPHA,
        )

        last_seen_time = time.monotonic()

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

                last_seen_time = time.monotonic()

                mav.send_landing_target_pos_quat(fwd, right, down)

                # Correccions PID
                v_fwd = pid_fwd.update(fwd)
                v_right = pid_right.update(right)

                lateral_speed = math.sqrt(v_fwd**2 + v_right**2)
                if lateral_speed > PID_OUTPUT_MAX:
                    scale = PID_OUTPUT_MAX / lateral_speed
                    v_fwd *= scale
                    v_right *= scale
                    lateral_speed = PID_OUTPUT_MAX

                lateral_offset = math.sqrt(fwd**2 + right**2)

                if lateral_offset < PHASE_2_CENTERED_THRSH:
                    v_down = PHASE_2_DESCENT_RATE
                elif lateral_offset < overhead_threshold:
                    ratio = 1.0 - 0.7 * (lateral_offset / overhead_threshold)
                    v_down = PHASE_2_DESCENT_RATE * ratio
                else:
                    v_down = 0.0

                print(f"[DESC] offset=({fwd:+.2f}, {right:+.2f}) m  "
                      f"vel=({v_fwd:+.3f}, {v_right:+.3f}, {v_down:+.2f}) m/s  "
                      f"alt={down:.1f} m  "
                      f"PID_fwd={pid_fwd}  PID_right={pid_right}")

                mav.move_velocity_body(v_fwd, v_right, v_down)

                if down <= PHASE_2_HANDOFF_ALT and lateral_offset < overhead_threshold:
                    print(f"[DESC] Phase 2 complete – alt {down:.1f} m, "
                          f"offset {lateral_offset:.2f} m → switching to LAND")
                    break

            else:
                # Target lost —> hold position (zero velocity), don't descend
                time_lost = time.monotonic() - last_seen_time
                if time_lost > PHASE_2_TARGET_LOST_TIMEOUT:
                    print(f"[DESC] Target lost for {time_lost:.1f}s — hovering...")
                    mav.move_velocity_body(0, 0, 0)
                    # Reset PIDs
                    pid_fwd.reset()
                    pid_right.reset()
                else:
                    # Brief loss —> keep last command for a moment, then hover
                    mav.move_velocity_body(0, 0, 0)

            _pace(t0, 1/loop_hz)

        # FASE 3: LAND mode PLND
        print(f"[PLND] Phase 3 – Switching to LAND mode, "
              f"streaming LANDING_TARGET until landed and disarmed")
        mav.switch_to_land()

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

                print(fwd, right, down)
                mav.send_landing_target_pos_quat(fwd, right, down)

            armed = mav.is_armed()
            if armed is not None:
                if not armed:
                    print("[PLND] Disarmed – precision landing complete.")
                    cameras.stop()
                    return

            _pace(t0, 1/loop_hz)

    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.switch_to_land()
        cameras.stop()

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
        description="ArduPilot PLND precision landing with PID lateral control",
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
                   help="Max lateral approach speed (m/s) for Phase 1")
    p.add_argument("--overhead-threshold", type=float, default=PHASE_1_THRSH_DIST,
                   help="Horizontal offset (m) that triggers Phase 1 → Phase 2 handoff")
    p.add_argument("--cam-roll",  type=float, default=0.0,
                   help="Camera mount roll (deg)")
    p.add_argument("--cam-pitch", type=float, default=-90.0,
                   help="Camera mount pitch (deg); -90 = straight down")
    p.add_argument("--cam-yaw",   type=float, default=0.0,
                   help="Camera mount yaw (deg)")
    p.add_argument("--display", action="store_true",
                   help="If set, will display the view of the cameras")

    # Phase 2 PID tuning overrides
    p.add_argument("--pid-kp", type=float, default=PID_KP,
                   help="PID proportional gain")
    p.add_argument("--pid-ki", type=float, default=PID_KI,
                   help="PID integral gain")
    p.add_argument("--pid-kd", type=float, default=PID_KD,
                   help="PID derivative gain")
    p.add_argument("--descent-rate", type=float, default=PHASE_2_DESCENT_RATE,
                   help="Phase 2 descent speed (m/s)")
    p.add_argument("--handoff-alt", type=float, default=PHASE_2_HANDOFF_ALT,
                   help="Altitude (m) at which Phase 2 hands off to LAND mode")
    return p.parse_args()



def main():
    args = parse_args()

    # Apply CLI PID overrides to module-level constants so Phase 2 uses them
    global PID_KP, PID_KI, PID_KD, PHASE_2_DESCENT_RATE, PHASE_2_HANDOFF_ALT
    PID_KP = args.pid_kp
    PID_KI = args.pid_ki
    PID_KD = args.pid_kd
    PHASE_2_DESCENT_RATE = args.descent_rate
    PHASE_2_HANDOFF_ALT = args.handoff_alt

    print("[SYSTEM] Initializing cameras...")
    cameras = GazeboStereoCapture(pipeline_wide, pipeline_narr, mtx_wide, dst_wide, mtx_narr, dst_narr)
    cameras.start()

    mav = MavlinkConnection(args.connect, args.baud)

    mav.arm_and_takeoff(args.alt)

    precision_land(mav, cameras, args.max_speed, args.hz, args.overhead_threshold, args.display)

if __name__ == "__main__":
    main()