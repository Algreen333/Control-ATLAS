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



T_C_WIDE = np.eye(4)
T_C_WIDE[0, 3] = 0.095675
T_C_NARR = np.eye(4)
T_C_NARR[0, 3] = -0.095675

print(T_C_WIDE, "\n", T_C_NARR)

DO_LAND = True
DO_TAKEOFF = True

MARKER_SIZE = 0.48
MARKER_ID = 26

TAKEOFF_ALT = 4

TARGET_LOST_TIMEOUT = 5.0

PHASE_1_THRSH_DIST = 0.6       # Lateral distance at which PHASE 1 will finish 
PHASE_1_DIST_VEL_MULT = 0.4     # Multiplier to distance to calculate speed (speed = distance*MULT)
PHASE_1_DIST_VEL_MAX = 0.4

PHASE_2_DIST_VEL_MAX = 0.1
PHASE_2_THRSH_ALT = 1
PHASE_2_DESCENT = 0.2
PHASE_2_DIST_VEL_MULT = 0.1

PHASE_3_THRSH_DIST = 0.1        # Lateral distance at which PHASE 3 will finish 
PHASE_3_DIST_VEL_MULT = 0.2     # Multiplier to distance to calculate speed (speed = distance*MULT)
PHASE_3_DIST_VEL_MAX = 0.03


# ---------------------------------------------------------------------------
# Camera initialization (for Gazebo)
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

class TargetTimeout:
    def __init__(self, timeout_sec: float = TARGET_LOST_TIMEOUT):
        self.timeout_sec = timeout_sec
        self._last_seen: Optional[float] = None
        self._lost_since: Optional[float] = None

    def update(self, detected: bool, mav, phase_tag: str = "") -> bool:
        now = time.monotonic()

        if detected:
            self._last_seen = now
            self._lost_since = None
            return False
        
        #Target not detected:

        if self._lost_since is None:
            self._lost_since = now
            print(f'[{phase_tag}] WARNING: target lost - holding position...')


        mav.move_velocity_body(0,0,0)

        elapsed = now - self._lost_since
        if self.timeout_sec != -1 and elapsed >= self.timeout_sec:
            print(f"[{phase_tag}] ABORT: Target lost for {elapsed:.1f}s"
                  f"(timeout={self.timeout_sec:.1f}s). Aborting...")
            
            mav.abort()
            return True
        else:
            print(f"[{phase_tag}] Target lost for {elapsed:.1f}s / {self.timeout_sec:.1f}s")
            return False



# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------

def approach(mav, cameras, max_lateral_speed_mps, lateral_speed_mult, overhead_threshold, loop_hz, do_display, target_timeout):
    try:
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

            if target_timeout.update(result is not None, mav, "APPR"):
                cameras.stop()
                return -1

            if result is not None:
                tvec, rvec = result

                right = float(np.ravel(tvec[0]))
                fwd = -float(np.ravel(tvec[1]))
                down = float(np.ravel(tvec[2]))

                distance = float(np.linalg.norm((fwd, right)))

                # Odometry message
                roll, pitch, yaw = get_drone_attitude_from_marker(rvec)
                q_wxyz = get_quaternions(roll, pitch, yaw, degrees=False)
                mav.send_odometry_message(fwd, right, down, q_wxyz=q_wxyz)

                if not target_visible_prev:
                    print(f"[APPR] Target acquired - Distance ({fwd:.1f}, {right:.1f}, {down:.1f}) m")
                target_visible_prev = True

                if distance < overhead_threshold:
                    print(f"[APPR] Target at {distance:.2f}m - Approach complete")
                    break
                
                else:
                    v_fwd = fwd * lateral_speed_mult
                    v_right = right * lateral_speed_mult
                    
                    current_speed = float(np.linalg.norm((v_fwd, v_right)))
                    
                    if current_speed > max_lateral_speed_mps:
                        v_fwd = (v_fwd / current_speed) * max_lateral_speed_mps
                        v_right = (v_right / current_speed) * max_lateral_speed_mps
                        current_speed = max_lateral_speed_mps

                    print(f"[APPR] Target at {distance:.2f}m - Approaching at ({v_fwd:.2f}, {v_right:.2f}, 0) m/s -> {current_speed:.2f} m/s...")
                    mav.move_velocity_body(v_fwd, v_right, 0)
            else:
                target_visible_prev = False 

            _pace(t0, 1/loop_hz)

        return 0

    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.abort()
        cameras.stop() 

        return -1

def descent(mav, cameras, max_lateral_speed_mps, lateral_speed_mult, descent_speed, threshold_alt, loop_hz, do_display, target_timeout):
    try:
        print(f"[DSCT] Phase 2 – Initial descent "
            f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")

        alt = -1
        distance = -1

        while True:
            t0 = time.monotonic()
            
            result = cameras.get_latest_pose()

            msg = mav.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            if msg: 
                alt = msg.relative_alt / 1000.0

                if alt != -1 and alt < threshold_alt:
                    print(f"[DSCT] Height: {alt}, target at {distance:.2f}m - Descent complete")
                    mav.move_velocity_body(0,0,descent_speed)
                    break

            
            if do_display:
                display_frame = cameras.get_latest_frame()
                if display_frame is not None:
                    cv2.imshow("img", display_frame)
                    cv2.waitKey(1)

            if target_timeout.update(result is not None, mav, "DSCT"):
                cameras.stop()
                return -1

            if result is not None:
                tvec, rvec = result

                right = float(np.ravel(tvec[0]))
                fwd = -float(np.ravel(tvec[1]))
                down = float(np.ravel(tvec[2]))

                distance = float(np.linalg.norm((fwd, right)))

                # Odometry message
                roll, pitch, yaw = get_drone_attitude_from_marker(rvec)
                q_wxyz = get_quaternions(roll, pitch, yaw, degrees=False)
                mav.send_odometry_message(fwd, right, down, q_wxyz=q_wxyz)
                                
                print(f"[DSCT] Target at {distance:.2f}m - Descending...")
                
                v_fwd = fwd * lateral_speed_mult
                v_right = right * lateral_speed_mult
                
                current_speed = float(np.linalg.norm((v_fwd, v_right)))
                
                if current_speed > max_lateral_speed_mps:
                    v_fwd = (v_fwd / current_speed) * max_lateral_speed_mps
                    v_right = (v_right / current_speed) * max_lateral_speed_mps
                    current_speed = max_lateral_speed_mps
                
                print(f"[DSCT] offset=({fwd:+.2f}, {right:+.2f}) m  "
                                f"vel=({v_fwd:+.3f}, {v_right:+.3f}, {descent_speed:+.2f}) m/s  "
                                f"alt={alt:.1f} m  dist={distance:.2f} m")
                mav.move_velocity_body(v_fwd, v_right, descent_speed)
                
            _pace(t0, 1/loop_hz)
        
        return 0

    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.abort()
        cameras.stop()

        return -1
    
def final(mav, cameras, max_lateral_speed_mps, lateral_speed_mult, overhead_threshold, loop_hz, do_display, target_timeout):
    try:
        print(f"[FINAL] Phase 3 – Final "
            f"(max {max_lateral_speed_mps * 100:.0f} cm/s) ...")

        target_timeout = TargetTimeout()
        target_visible_prev = True
        while True:
            t0 = time.monotonic()
            
            result = cameras.get_latest_pose()
            
            if do_display:
                display_frame = cameras.get_latest_frame()
                if display_frame is not None:
                    cv2.imshow("img", display_frame)
                    cv2.waitKey(1)

            if target_timeout.update(result is not None, mav, "FINAL"):
                cameras.stop()
                return -1

            if result is not None:
                tvec, rvec = result

                right = float(np.ravel(tvec[0]))
                fwd = -float(np.ravel(tvec[1]))
                down = float(np.ravel(tvec[2]))

                distance = float(np.linalg.norm((fwd, right)))

                # Odometry message
                roll, pitch, yaw = get_drone_attitude_from_marker(rvec)
                q_wxyz = get_quaternions(roll, pitch, yaw, degrees=False)
                mav.send_odometry_message(fwd, right, down, q_wxyz=q_wxyz)
                
                if not target_visible_prev:
                    print(f"[FINAL] Target acquired - Distance ({fwd:.1f}, {right:.1f}, {down:.1f}) m")
                target_visible_prev = True

                if distance < overhead_threshold:
                    print(f"[FINAL] Target at {distance:.2f}m - Final complete")
                    mav.move_velocity_body(0, 0, 0)
                    break

                else:
                    v_fwd = fwd * lateral_speed_mult
                    v_right = right * lateral_speed_mult
                    
                    current_speed = float(np.linalg.norm((v_fwd, v_right)))
                    
                    if current_speed > max_lateral_speed_mps:
                        v_fwd = (v_fwd / current_speed) * max_lateral_speed_mps
                        v_right = (v_right / current_speed) * max_lateral_speed_mps
                        current_speed = max_lateral_speed_mps

                    mav.move_velocity_body(v_fwd, v_right, 0)
                    print(f"[FINAL] offset=({fwd:+.2f}, {right:+.2f}) m  "
                                                    f"vel=({v_fwd:+.3f}, {v_right:+.3f}, 0) m/s  "
                                                    f"dist={distance:.2f} m")
            else:
                target_visible_prev = False
            _pace(t0, 1/loop_hz)

        return 0

    except KeyboardInterrupt:
        print(f"[SYSTEM] Program aborted. Stopping and landing ...")
        mav.abort()
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
        description="ArduPilot PLND precision landing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    p.add_argument("--connect", default="udpin:0.0.0.0:14550",
                   help="MAVLink connection string")
    p.add_argument("--baud", type=int, default=57600,
                   help="Serial baud rate (ignored for UDP/TCP)")
    p.add_argument("--debug", action="store_true", help="IF ENABLED, WILL NOT SEND MOVE COMMANDS")
    p.add_argument("--onlycam", action="store_true", help="ONLY SHOW CAMERA DISPLAY")

    p.add_argument("--auto_arm", action="store_true", help="Automatically arm")
    p.add_argument("--auto_guided", action="store_true", help="Automatically set guided mode")
    p.add_argument("--takeoff", action="store_true", help="Attempt to take off")
    
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help="Takeoff altitude (m)")
    p.add_argument("--hz", type=float, default=10.0,
                   help="Control loop / message rate (Hz)")
    p.add_argument("--timeout", type=float, default=TARGET_LOST_TIMEOUT,
                   help="Max seconds to wait for landing disarm")
    
    p.add_argument("--p1_max_speed", type=float, default=PHASE_1_DIST_VEL_MAX,
                   help="Max lateral approach speed (m/s) in approach")
    p.add_argument("--p1_mult_speed", type=float, default=PHASE_1_DIST_VEL_MULT,
                   help="Lateral approach speed multiplier (m/s) in approach")
    p.add_argument("--p1_threshold", type=float, default=PHASE_1_THRSH_DIST,
                   help="Horizontal offset (m) that triggers descent handoff")
    
    p.add_argument("--p2_max_speed", type=float, default=PHASE_2_DIST_VEL_MAX,
                   help="Max lateral speed (m/s) in descent")
    p.add_argument("--p2_mult_speed", type=float, default=PHASE_2_DIST_VEL_MULT,
                   help="Lateral speed multiplier (m/s) in descent")
    p.add_argument("--p2_descent", type=float, default=PHASE_2_DESCENT,
                   help="Max descent speed")
    p.add_argument("--p2_threshold", type=float, default=PHASE_2_THRSH_ALT,
                   help="Threshold altitude (m) that triggers final handoff")
    
    p.add_argument("--p3_max_speed", type=float, default=PHASE_3_DIST_VEL_MAX,
                   help="Max lateral approach speed (m/s) in final")
    p.add_argument("--p3_mult_speed", type=float, default=PHASE_3_DIST_VEL_MULT,
                   help="Lateral approach speed multiplier (m/s) in final")
    p.add_argument("--p3_threshold", type=float, default=PHASE_3_THRSH_DIST,
                   help="Horizontal offset (m) that triggers LAND handoff")

    p.add_argument("--gazebo", action="store_true", help="Will use gazebo captures, else RPi Sync captures")
    p.add_argument("--cam-roll",  type=float, default=0.0,
                   help="Camera mount roll (deg)")
    p.add_argument("--cam-pitch", type=float, default=-90.0,
                   help="Camera mount pitch (deg); -90 = straight down")
    p.add_argument("--cam-yaw",   type=float, default=0.0,
                   help="Camera mount yaw (deg)")
    p.add_argument("--display", action="store_true", help="If set, will display the view of the cameras")
    p.add_argument("--server", action="store_true", help="Start telemetry video server to display cameras")
    return p.parse_args()



def main():
    args = parse_args()


    print("[SYSTEM] Initializing cameras...")
    if args.gazebo:
        cameras = GazeboStereoCapture(pipeline_wide, pipeline_narr, mtx_wide, dst_wide, mtx_narr, dst_narr)
        cameras.start()
    else:
        cameras = StereoCapture(0, (1536, 864), 1, (1640, 1232), T_C_WIDE=T_C_WIDE, T_C_NARR=T_C_NARR, 
            configWide=CalibrationConfig.from_path("configs/1536x864-v3.conf"),
            configNarr=CalibrationConfig.from_path("configs/1640x1232-v2.conf"),
            arucoDICT=aruco.DICT_ARUCO_ORIGINAL, marker_id=MARKER_ID, marker_size=MARKER_SIZE)
        cameras.start(True)

    web_server_thread = None
    if args.server:
        web_server = TelemetryServer(capture_instance=cameras, port=5001)
        web_server_thread = web_server.start_background()

    if not args.onlycam:
        target_timeout = TargetTimeout(args.timeout)

        mav = MavlinkConnection(args.connect, args.baud, args.debug)

        if (args.auto_guided): mav.setGuided()
        else: mav.waitGuided()

        if (args.auto_arm): mav.arm()
        else: mav.waitArmed()

        if (args.takeoff): mav.takeoff(args.alt)

        if approach(mav, cameras, args.p1_max_speed, args.p1_mult_speed, args.p1_threshold, args.hz, args.display, target_timeout) < 0: return -1
        
        if descent(mav, cameras, args.p2_max_speed, args.p2_mult_speed, args.p2_descent, args.p2_threshold, args.hz, args.display, target_timeout) < 0: return -1

        if final(mav, cameras, args.p3_max_speed, args.p3_mult_speed, args.p3_threshold, args.hz, args.display, target_timeout) < 0: return -1
        else: mav.switch_to_land()

        mav.wait_for_disarm()

        # Drop and tajeoff sequence
        mav.open_gripper()

        time.sleep(1)
        
        mav.setStabilize()
        
        mav.arm()
        mav.waitArmed()

        mav.setGuided()
        mav.waitGuided()

        mav.takeoff(args.alt)

        time.sleep(1)

        mav.close_gripper()
    
    else:
        import code
        
        banner = """
            [ONLYCAM] Interactive Python shell.

            Example:
            cameras.set_param('marker_id', 42)
            cameras.wide.controls.ExposureTime = 5000
            Type Ctrl-D or exit() to quit.
        """
        local_vars = {**globals(), "cameras": cameras, "args": args}
        code.interact(banner=banner, local=local_vars)
        
        
        if web_server_thread is not None:
            web_server_thread.join()


        if web_server_thread is not None: web_server_thread.join()

if __name__ == "__main__":
    main()