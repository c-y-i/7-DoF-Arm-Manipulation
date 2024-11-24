import numpy as np
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq_in):
    """
    Function to compute the end-effector linear and angular velocity given joint velocities.

    INPUTS:
    q_in - 1 x 7 configuration vector (of joint angles) [q1, q2, q3, q4, q5, q6, q7]
    dq_in - 1 x 7 vector of joint velocities [dq1, dq2, dq3, dq4, dq5, dq6, dq7]

    OUTPUTS:
    v - 6 x 1 end-effector velocity vector (linear velocity [vx, vy, vz] and angular velocity [wx, wy, wz])
    """
    # Calculate the Jacobian matrix for the given joint configuration
    J = calcJacobian(q_in)

    # Calculate end-effector velocity
    dq_in = np.array(dq_in).reshape(-1, 1)  # Ensure dq_in is a column vector
    v = J @ dq_in

    return v.flatten()
