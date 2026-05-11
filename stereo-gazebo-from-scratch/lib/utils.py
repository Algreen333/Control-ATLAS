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