"""
plnd.guidance
=============
Velocity-based guidance: converts a vision offset into a NED velocity
command, applying a proportional controller and a hard speed cap.
 
Why cam_lateral_to_ned_velocity instead of tvec_body_to_ned_velocity
----------------------------------------------------------------------
For a downward-facing camera the cam-to-body rotation maps the depth axis
(tvec_cam[2] = altitude, typically several metres) into the body forward/
backward axis.  That contaminates the horizontal velocity demand with a
large altitude-driven term, causing the drone to fly away at max speed
rather than toward the target.
 
cam_lateral_to_ned_velocity avoids this by zeroing the depth component
*before* rotating, so only the true image-plane lateral offsets
(tvec_cam[0] = right, tvec_cam[1] = down-in-image) feed the controller.
tvec_body_to_ned_velocity is kept for cases where the caller supplies a
pre-corrected body vector with the altitude component already removed.
"""
 
import math
from typing import Tuple
 
import numpy as np
 
 
def cam_lateral_to_ned_velocity(
    tvec_cam: np.ndarray,
    cam_to_body_R: np.ndarray,
    heading_rad: float,
    max_speed_mps: float = 0.10,
    p_gain: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Convert the lateral camera-frame offset to the target into a NED
    velocity command, capped at max_speed_mps.
 
    This is the preferred function for downward-facing cameras.  It ignores
    tvec_cam[2] (depth / altitude) and rotates only the image-plane
    components [tvec_cam[0], tvec_cam[1], 0] into body frame, preventing
    the altitude from polluting the horizontal velocity demand.
 
    Parameters
    ----------
    tvec_cam      : (3,) – raw output of process_frame() in camera frame
                    (X right, Y down-in-image, Z depth/altitude). Any shape
                    broadcastable to (3,) is accepted (ravel() is applied).
    cam_to_body_R : (3×3) – camera-to-body rotation matrix
    heading_rad   : vehicle yaw in radians (from VFR_HUD or ATTITUDE)
    max_speed_mps : hard speed cap (default 0.10 m/s = 10 cm/s)
    p_gain        : proportional gain – 1 m error → p_gain m/s demand
                    before clamping (default 1.0)
 
    Returns
    -------
    (vx_ned, vy_ned, lateral_err_m)
        vx_ned, vy_ned  : NED velocity setpoint (m/s)
        lateral_err_m   : horizontal distance to target (m), for threshold
                          checks; computed in body frame after rotation
    """
    tvec_cam = np.asarray(tvec_cam, dtype=float).ravel()
 
    # Zero the depth axis so altitude never drives horizontal velocity.
    # Only the true image-plane offsets are rotated into body frame.
    lateral_cam = np.array([tvec_cam[0], tvec_cam[1], 0.0])
    lateral_body = cam_to_body_R @ lateral_cam          # (3,), body FRD
 
    lateral_err = math.hypot(lateral_body[0], lateral_body[1])
 
    vx_body = p_gain * lateral_body[0]
    vy_body = p_gain * lateral_body[1]
 
    # Clamp magnitude to max_speed_mps
    speed = math.hypot(vx_body, vy_body)
    if speed > max_speed_mps:
        scale = max_speed_mps / speed
        vx_body *= scale
        vy_body *= scale
 
    # Rotate body → NED via vehicle heading
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    vx_ned =  cos_h * vx_body - sin_h * vy_body
    vy_ned =  sin_h * vx_body + cos_h * vy_body
 
    return vx_ned, vy_ned, lateral_err
 
 
def tvec_body_to_ned_velocity(
    tvec_body: np.ndarray,
    heading_rad: float,
    max_speed_mps: float = 0.10,
    p_gain: float = 1.0,
) -> Tuple[float, float]:
    """
    Convert a body-frame lateral offset into a NED velocity command.
 
    Use this only when tvec_body is known to have the altitude component
    correctly placed in body Z (down) – i.e. the cam-to-body rotation is
    verified correct for your mount.  For downward cameras prefer
    cam_lateral_to_ned_velocity, which is immune to rotation-matrix errors.
 
    Parameters
    ----------
    tvec_body     : (3,) [fwd, right, down] metres – body FRD frame
    heading_rad   : vehicle yaw in radians
    max_speed_mps : hard speed cap (default 0.10 m/s)
    p_gain        : proportional gain (default 1.0)
 
    Returns
    -------
    (vx_ned, vy_ned) : north and east velocity in m/s
    """
    vx_body = p_gain * tvec_body[0]
    vy_body = p_gain * tvec_body[1]
 
    speed = math.hypot(vx_body, vy_body)
    if speed > max_speed_mps:
        scale = max_speed_mps / speed
        vx_body *= scale
        vy_body *= scale
 
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    vx_ned =  cos_h * vx_body - sin_h * vy_body
    vy_ned =  sin_h * vx_body + cos_h * vy_body
 
    return vx_ned, vy_ned
