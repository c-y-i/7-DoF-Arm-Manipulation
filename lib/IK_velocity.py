import numpy as np
from lib.calcJacobian import calcJacobian

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    NaN, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is NaN, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the L2 norm of the error between the target velocity and the achieved velocity.
    """

    # Initialize joint velocities to zeros
    dq = np.zeros((7, 1))

    # Reshape input velocities
    v_in = v_in.reshape((3, 1))
    omega_in = omega_in.reshape((3, 1))

    # Compute the Jacobian for the current configuration
    J = calcJacobian(q_in)

    # Stack linear and angular velocities into a single vector
    X = np.vstack((v_in, omega_in))

    # Find rows without NaN values to filter Jacobian and velocity vector
    valid_indices = ~np.isnan(X[:, 0])
    filtered_X = X[valid_indices]
    filtered_J = J[valid_indices]

    # Solve for the joint velocities using least squares
    result = np.linalg.lstsq(filtered_J, filtered_X, rcond=1e-10)
    dq = result[0]

    return dq.flatten()
