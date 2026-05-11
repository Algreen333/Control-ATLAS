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


DO_LAND = True
DO_TAKEOFF = True

MARKER_SIZE = 1
MARKER_ID = 49

PHASE_1_THRSH_DIST = 1        # Lateral distance at which PHASE 1 will finish 
PHASE_1_DIST_VEL_MULT = 0.2     # Multiplier to distance to calculate speed (speed = distance*MULT)

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
def precision_land(mav, cameras, max_lateral_speed_mps, loop_hz, overhead_threshold):
    try:
        print(f"[APPR] Phase 1 – Initial approach "
            f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")

        # PHASE 1: Locate and initial approach
        target_visible_prev = False
        while True:
            t0 = time.monotonic()
            result = cameras.process_frame()

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

                if distance < PHASE_1_THRSH_DIST:
                    print(f"[APPR] Target at {distance}m - Approach complete")
                    break
                
                else:
                    print(f"[APPR] Target at {distance}m - Approaching...")
                    v_fwd = fwd*PHASE_1_DIST_VEL_MULT
                    v_right = right*PHASE_1_DIST_VEL_MULT
                    
                    current_speed = float(np.linalg.norm((v_fwd, v_right)))
                    
                    if current_speed > max_lateral_speed_mps:
                        v_fwd = (v_fwd / current_speed) * max_lateral_speed_mps
                        v_right = (v_right / current_speed) * max_lateral_speed_mps
                        current_speed = max_lateral_speed_mps

                    print(f"[APPR] Target at {distance:.2f}m - Approaching at ({v_fwd}, {v_right}, 0) m/s -> {current_speed:.2f} m/s...")
                    mav.move_velocity_body(v_fwd, v_right, 0)
                
            _pace(t0, 1/loop_hz)

        print(f"[PLND] Phase 2 - Switching to LAND mode, streaming LANDING_TARGET until landed and disarmed")
        mav.switch_to_land()
        while True:
            t0 = time.monotonic()
            result = cameras.process_frame()

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
                    return
            
            _pace(t0, 1/loop_hz)


    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.switch_to_land()

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



def main():
    args = parse_args()

    print("[SYSTEM] Initializing cameras...")
    cameras = GazeboStereoCapture(pipeline_wide, pipeline_narr, mtx_wide, dst_wide, mtx_narr, dst_narr)

    mav = MavlinkConnection(args.connect, args.baud)

    mav.arm_and_takeoff(args.alt)

    precision_land(mav, cameras, args.max_speed, args.hz, args.overhead_threshold)


if __name__ == "__main__":
    main()