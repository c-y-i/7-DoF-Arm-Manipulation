import numpy as np
from math import pi, atan2, sin, cos

def compute_dynamic_intercept(block_position, block_orientation, t_now, omega, theta_pick, R=0.305, z=0.200):
    """
    Compute the intercept time and predicted pose for a dynamic block on a rotating turntable.
    block_position: current [x, y, z]
    block_orientation: 3x3 rotation matrix for block orientation
    t_now: current time (seconds)
    omega: angular velocity (rad/s) of turntable
    theta_pick: angle at which we want to pick the block
    R: radius of turntable
    z: height of turntable surface

    Returns:
    - t_intercept: time in seconds until block reaches theta_pick
    - intercept_pos: [x, y, z] of block at intercept time
    - intercept_orientation: 3x3 rotation matrix of block at intercept time
    """
    x, y, _ = block_position
    theta_now = atan2(y, x)

    dtheta = theta_pick - theta_now
    if dtheta < 0:
        dtheta += 2*pi

    t_intercept = dtheta / omega

    # Future position
    x_intercept = R * cos(theta_pick)
    y_intercept = R * sin(theta_pick)
    z_intercept = z

    # Future orientation
    R_z = np.array([
        [cos(dtheta), -sin(dtheta), 0],
        [sin(dtheta),  cos(dtheta), 0],
        [0,            0,           1]
    ])
    intercept_orientation = R_z @ block_orientation
    intercept_pos = np.array([x_intercept, y_intercept, z_intercept])
    return t_intercept, intercept_pos, intercept_orientation

def orientation_to_rpy(R):
    """
    Convert rotation matrix R to roll, pitch, yaw.
    R: 3x3 rotation matrix
    Returns (roll, pitch, yaw)
    """
    yaw = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = np.arctan2(R[2,1], R[2,2])
    return roll, pitch, yaw
