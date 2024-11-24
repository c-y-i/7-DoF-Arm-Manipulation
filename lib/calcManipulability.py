import numpy as np
from lib.calcJacobian import calcJacobian

def calcManipulability(q_in):
    """
    Helper function for calculating manipulability ellipsoid and index

    INPUTS:
    q_in - 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]

    OUTPUTS:
    mu - a float scalar, manipulability index
    M  - 3 x 3 manipulability matrix for the linear portion
    """
    # Calculate the Jacobian matrix
    J = calcJacobian(q_in)

    # Extract the position part of the Jacobian
    J_pos = J[:3, :]

    # Manipulability matrix for the linear portion
    M = J_pos @ J_pos.T

    # Calculate manipulability index using singular value decomposition (SVD)
    _, singular_values, _ = np.linalg.svd(J_pos)
    mu = np.prod(singular_values)

    return mu, M
