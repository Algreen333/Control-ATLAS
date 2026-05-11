"""
plnd.transforms
===============
Coordinate-frame conversions between the camera, vehicle body, and MAVLink
LANDING_TARGET angle conventions.

All vectors use right-hand, axis-labelled conventions:
  Camera frame : X right, Y down,  Z forward
  Body frame   : X forward, Y right, Z down  (FRD)
  NED frame    : X north,   Y east,  Z down
"""

import math
from typing import Tuple

import numpy as np


def build_cam_to_body_rotation(
    roll_deg: float = 0.0,
    pitch_deg: float = -90.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a (3×3) rotation matrix that maps vectors from the camera frame
    (X right, Y down, Z forward) to the vehicle body frame (X fwd, Y right,
    Z down), given the camera mount's roll/pitch/yaw in degrees.

    Defaults assume a camera mounted flat and pointing straight down.
    Adjust to match your physical installation.
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


def cam_to_body(tvec_cam: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate a translation vector from the camera frame into the body frame.

    Parameters
    ----------
    tvec_cam : (3,) – target position in camera frame
    R        : (3×3) – rotation matrix from camera to body (see build_cam_to_body_rotation)

    Returns
    -------
    (3,) – target position in body frame [fwd, right, down]
    """
    return R @ tvec_cam


def tvec_to_angles(tvec_body: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a target position vector (body FRD frame) to the angle offsets
    and slant distance expected by the MAVLink LANDING_TARGET message.

    ArduPilot convention (PLND_TYPE = 2)
    -------------------------------------
    angle_x : lateral angle,  + = target is to the RIGHT of the vehicle
    angle_y : vertical angle, + = target is forward-below the vehicle

    Parameters
    ----------
    tvec_body : (3,) – [fwd, right, down] in metres

    Returns
    -------
    (angle_x, angle_y, distance) – angles in radians, distance in metres
    """
    up, bwd, right = tvec_body
    down = -up
    fwd = -bwd
    distance = float(np.linalg.norm(tvec_body))

    if distance < 1e-6:
        return 0.0, 0.0, 0.0

    angle_x = math.atan2(right, fwd)        # + = right
    angle_y = math.atan2(-down, fwd)        # + = forward-below  (down is +Z)

    print(f"{tvec_body} -> {fwd}, {right}, {down} -> {math.degrees(angle_x):.1f}, {math.degrees(angle_y):.1f}, {distance:.1f}")
    return angle_x, angle_y, distance


# ---------------------------------------------------------------------------
# Quaternion helpers (for position+quaternion LANDING_TARGET variant)
# ---------------------------------------------------------------------------

def rvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Convert a Rodrigues rotation vector to a (3×3) rotation matrix.
 
    Uses the closed-form Rodrigues formula:
        R = I + sin(θ)·K + (1 − cos(θ))·K²
    where K is the skew-symmetric cross-product matrix of the unit axis.
 
    Parameters
    ----------
    rvec : array-like, any shape broadcastable to (3,) – Rodrigues vector.
           OpenCV routines (estimatePoseSingleMarkers, solvePnP) often return
           shape (1, 3) or (1, 1, 3); ravel() normalises all of these.
 
    Returns
    -------
    (3×3) rotation matrix
    """
    rvec = np.asarray(rvec, dtype=float).ravel()   # → guaranteed (3,)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-10:
        return np.eye(3)
 
    axis = rvec / angle
    K = np.array([
        [       0, -axis[2],  axis[1]],
        [ axis[2],        0, -axis[0]],
        [-axis[1],  axis[0],        0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
 
 
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a (3×3) rotation matrix to a unit quaternion [w, x, y, z].
 
    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
 
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
 
    return np.array([w, x, y, z], dtype=float)
 
 
def rvec_cam_to_body_quaternion(
    rvec_cam: np.ndarray,
    cam_to_body_R: np.ndarray,
) -> np.ndarray:
    """
    Express the target's orientation (given as a camera-frame Rodrigues vector)
    as a quaternion in the vehicle body frame (FRD).
 
    Steps
    -----
    1. rvec → rotation matrix in camera frame  (Rodrigues formula)
    2. Compose with the camera-to-body mount rotation
    3. Rotation matrix → quaternion  (Shepperd's method)
 
    Parameters
    ----------
    rvec_cam      : (3,) – Rodrigues vector from cv2.aruco / solvePnP
    cam_to_body_R : (3×3) – camera-to-body rotation (from build_cam_to_body_rotation)
 
    Returns
    -------
    (4,) – quaternion [w, x, y, z] in body FRD frame, MAVLink convention
    """
    R_target_cam  = rvec_to_rotation_matrix(np.asarray(rvec_cam, dtype=float).ravel())
    R_target_body = cam_to_body_R @ R_target_cam
    return rotation_matrix_to_quaternion(R_target_body)
