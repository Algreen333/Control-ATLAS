import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def average_pos(positions):
    """
    Given a list of points, calculate the average of the components of the points.

    :param list positions: List of positions. Each position is a tuple of the type (float, float, float)
    :return x_avg (float):
    :return y_avg (float):
    :return z_avg (float):
    """

    xs = 0
    ys = 0
    zs = 0

    for p in positions:
        x, y, z = p
        xs+=x
        ys+=y
        zs+=z
    
    xs /= len(positions)
    ys /= len(positions)
    zs /= len(positions)

    return xs, ys, zs

def clamp(value, min_val, max_val):
    """
    Clamps value to range.

    :param int|float value: Value to be clamped
    :param int|float min_val: Minimum value
    :param int|float max_val: Maximum value

    :return clamped_value (int|float): Resulting clamped value
    """

    value = max(value, min_val)
    value = min(value, max_val)

    return value

def get_quaternions(roll, pitch, yaw, degrees=False):
    """
    Transforms roll, pitch, yaw into a quaternion.
    Returns: A tuple (w, x, y, z) as expected by MAVLink.
    """
    # 'ZYX' is the standard aerospace sequence (Yaw, Pitch, Roll)
    rot = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=degrees)
    
    # SciPy returns [x, y, z, w]
    q = rot.as_quat() 
    
    # Reorder to [w, x, y, z] for ArduPilot/MAVLink compatibility
    return (q[3], q[0], q[1], q[2])

def create_homogeneous_matrix(rvec, tvec):
    """
    Converts OpenCV rvec and tvec into a 4x4 homogeneous transformation matrix.
    """
    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create 4x4 matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return T

def get_drone_attitude_from_marker(rvec):
    """
    Calculates the drone's roll, pitch, and yaw relative to a marker flat on the ground.
    Assumes a downward-facing camera where image-top is drone-forward.
    
    Returns:
        roll, pitch, yaw (in radians)
    """
    # 1. Rotation from Marker frame to Camera frame
    rot_m2c = Rotation.from_rotvec(rvec.flatten())
    
    # 2. Rotation from Camera frame to Drone Body frame (+90 deg around Z)
    rot_c2b = Rotation.from_euler('z', 90, degrees=True)
    
    # 3. Total rotation: Marker to Drone Body
    rot_m2b = rot_c2b * rot_m2c
    
    # Extract Euler angles (Yaw, Pitch, Roll) in radians
    yaw, pitch, roll = rot_m2b.as_euler('ZYX', degrees=False)
    
    return roll, pitch, yaw