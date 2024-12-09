import numpy as np
from typing import Tuple, List

class FrankaIK:
    """
    A refactored class for computing the inverse kinematics of a Franka Panda robot arm.
    This class encapsulates all logic and methods in one place.
    """

    @staticmethod
    def compute_ik(T_array: List[float], q7: float, qa: List[float]) -> np.ndarray:
        """
        Compute the inverse kinematics for a Franka Panda robot given a desired end-effector
        pose and certain joint configuration parameters.

        Parameters
        ----------
        T_array : List[float]
            A flat list representing a 4x4 transformation matrix (row-major order).
        q7 : float
            The angle for joint 7 of the robot.
        qa : List[float]
            Reference angles used to set certain joint angles if alignment conditions are met.

        Returns
        -------
        np.ndarray
            A 4x7 numpy array. Each row contains a possible joint configuration (q1 through q7).
            If no valid solution is found for a particular solution branch, that row is filled with NaN.
        """

        print("Computing IK for Franka Panda robot...")
        # Define robot geometry (link lengths and offsets)
        link_lengths = [0.3330, 0.3160, 0.3840, 0.2104]
        offsets = [0.0825, 0.0880]

        # Precompute various trigonometric and geometric values
        s24, s46, L24, L46, thetaH46, theta342, theta46H = FrankaIK._compute_geometry_terms(link_lengths, offsets)

        # Joint limits for Franka Panda (q1 to q7)
        joint_limits = [
            [-2.8973,  2.8973],
            [-1.7628,  1.7628],
            [-2.8973,  2.8973],
            [-3.0718, -0.0698],
            [-2.8973,  2.8973],
            [-0.0175,  3.7525],
            [-2.8973,  2.8973]
        ]

        # Initialize solution arrays
        solutions = np.full((4, 7), np.nan)
        solution_nan_row = np.array([np.nan] * 7)

        # Check joint 7 limit
        if not FrankaIK._within_joint_limits(q7, joint_limits[6]):
            return solutions
        solutions[:, 6] = q7

        # Reshape input transformation array into 4x4 matrix
        base_to_ee = np.array(T_array).reshape((4, 4))
        Re = base_to_ee[:3, :3]
        ze = base_to_ee[:3, 2]
        pe = base_to_ee[:3, 3]

        # Compute joint 7 position
        p_joint_7 = pe - link_lengths[3] * ze

        # Compute orientation-dependent vector for joint 6
        x6_direction = np.array([np.cos(q7 - np.pi / 4), -np.sin(q7 - np.pi / 4), 0.0])
        Jx_6 = Re @ x6_direction
        Jx_6 /= np.linalg.norm(Jx_6)
        joint_6_reference = p_joint_7 - offsets[1] * Jx_6

        # Compute distances and angles for the 2-4-6 linkage
        joint_2_position = np.array([0.0, 0.0, link_lengths[0]])
        V_2_to_6 = joint_6_reference - joint_2_position
        s26 = np.sum(V_2_to_6 ** 2)
        L26 = np.sqrt(s26)

        # Check triangle feasibility
        if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
            return solutions  # No valid configuration if links cannot form a triangle

        # Compute joint 4 angle
        theta246 = np.arccos((s24 + s46 - s26) / (2.0 * L24 * L46))
        q4 = theta246 + thetaH46 + theta342 - 2.0 * np.pi
        if not FrankaIK._within_joint_limits(q4, joint_limits[3]):
            return solutions
        solutions[:, 3] = q4

        # Additional angles for joint 6 calculation
        theta_462 = np.arccos((s26 + s46 - s24) / (2.0 * L26 * L46))
        theta26H = theta46H + theta_462
        dis_26 = -L26 * np.cos(theta26H)

        # Compute orientation at joint 6
        Z_6 = np.cross(ze, Jx_6)
        Y_6 = np.cross(Z_6, Jx_6)
        R_6 = np.column_stack((Jx_6,
                               Y_6 / np.linalg.norm(Y_6),
                               Z_6 / np.linalg.norm(Z_6)))
        V_6_62 = R_6.T @ (-V_2_to_6)
        Phi_6 = np.arctan2(V_6_62[1], V_6_62[0])
        Theta_6 = np.arcsin(dis_26 / np.sqrt(V_6_62[0] ** 2 + V_6_62[1] ** 2))
        q6_candidates = np.array([np.pi - Theta_6 - Phi_6, Theta_6 - Phi_6])

        # Adjust joint 6 angles to valid range
        for i, q6_val in enumerate(q6_candidates):
            if q6_val <= joint_limits[5][0]:
                q6_val += 2.0 * np.pi
            elif q6_val >= joint_limits[5][1]:
                q6_val -= 2.0 * np.pi

            if q6_val <= joint_limits[5][0] or q6_val >= joint_limits[5][1]:
                solutions[2 * i, 5] = np.nan
                solutions[2 * i + 1, 5] = np.nan
            else:
                solutions[2 * i, 5] = q6_val
                solutions[2 * i + 1, 5] = q6_val

        # If all q6 candidates are invalid, return no solution
        if np.isnan(solutions[0, 5]) and np.isnan(solutions[2, 5]):
            return solutions

        # Compute q1, q2, q3, q5
        thetaP26 = (3.0 * np.pi / 2) - theta_462 - theta246 - theta342
        thetaP = np.pi - thetaP26 - theta26H
        LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

        z_5_storage = np.zeros((4, 3))
        V2P_storage = np.zeros((4, 3))

        # Compute q1 and q2 based on geometric constraints
        for i in range(2):
            q6_val = solutions[2 * i, 5]
            if np.isnan(q6_val):
                continue

            # Compute z_5 direction
            z_6_5 = np.array([np.sin(q6_val), np.cos(q6_val), 0.0])
            z_5 = R_6 @ z_6_5
            V_2_to_P = joint_6_reference - LP6 * z_5 - joint_2_position

            z_5_storage[2 * i] = z_5
            z_5_storage[2 * i + 1] = z_5
            V2P_storage[2 * i] = V_2_to_P
            V2P_storage[2 * i + 1] = V_2_to_P

            L2P = np.linalg.norm(V_2_to_P)
            # If alignment is nearly vertical
            if np.abs(V_2_to_P[2] / L2P) > 0.999:
                solutions[2 * i, 0] = qa[0]
                solutions[2 * i, 1] = 0.0
                solutions[2 * i + 1, 0] = qa[0]
                solutions[2 * i + 1, 1] = 0.0
            else:
                # Compute q1 and q2 solutions
                solutions[2 * i, 0] = np.arctan2(V_2_to_P[1], V_2_to_P[0])
                solutions[2 * i, 1] = np.arccos(V_2_to_P[2] / L2P)
                if solutions[2 * i, 0] < 0:
                    solutions[2 * i + 1, 0] = solutions[2 * i, 0] + np.pi
                else:
                    solutions[2 * i + 1, 0] = solutions[2 * i, 0] - np.pi
                solutions[2 * i + 1, 1] = -solutions[2 * i, 1]

        # Validate angles and compute q3, q5
        for i in range(4):
            if (solutions[i, 0] <= joint_limits[0][0] or solutions[i, 0] >= joint_limits[0][1] or
                solutions[i, 1] <= joint_limits[1][0] or solutions[i, 1] >= joint_limits[1][1]):
                solutions[i] = solution_nan_row
                continue

            V2P = V2P_storage[i]
            if np.isnan(solutions[i, 5]):
                solutions[i] = solution_nan_row
                continue

            # Compute joint 3 direction
            Jz3 = V2P / np.linalg.norm(V2P)

            # Compute joint 3 orientation
            Jy3 = -np.cross(V_2_to_6, V2P)
            Jy3 = Jy3 / np.linalg.norm(Jy3)
            Jx3 = np.cross(Jy3, Jz3)

            # Rotation from base to joint 2
            R_1 = np.array([
                [np.cos(solutions[i, 0]), -np.sin(solutions[i, 0]), 0.0],
                [np.sin(solutions[i, 0]),  np.cos(solutions[i, 0]), 0.0],
                [0.0, 0.0, 1.0]
            ])
            R_1_2 = np.array([
                [np.cos(solutions[i, 1]), -np.sin(solutions[i, 1]), 0.0],
                [0.0, 0.0, 1.0],
                [-np.sin(solutions[i, 1]), -np.cos(solutions[i, 1]), 0.0]
            ])
            R_2 = R_1 @ R_1_2

            # Compute q3
            x_2_3 = R_2.T @ Jx3
            q3 = np.arctan2(x_2_3[2], x_2_3[0])
            if not FrankaIK._within_joint_limits(q3, joint_limits[2]):
                solutions[i] = solution_nan_row
                continue
            solutions[i, 2] = q3

            # Compute vector for joint 4 and use R_5 to find q5
            Vect4 = (joint_2_position +
                     link_lengths[1] * Jz3 +
                     offsets[0] * Jx3 -
                     joint_6_reference +
                     link_lengths[2] * z_5_storage[i])

            q6_val = solutions[i, 5]
            R_56 = np.array([
                [np.cos(q6_val), -np.sin(q6_val), 0.0],
                [0.0, 0.0, -1.0],
                [np.sin(q6_val), np.cos(q6_val),  0.0]
            ])
            R_5 = R_6 @ R_56.T
            V_5_H4 = R_5.T @ Vect4
            q5 = -np.arctan2(V_5_H4[1], V_5_H4[0])

            if not FrankaIK._within_joint_limits(q5, joint_limits[4]):
                solutions[i] = solution_nan_row
                continue
            solutions[i, 4] = q5

        return solutions

    @staticmethod
    def _within_joint_limits(joint_angle: float, limit: List[float]) -> bool:
        """Check if a given joint angle lies within specified limits."""
        return limit[0] <= joint_angle <= limit[1]

    @staticmethod
    def _compute_geometry_terms(link_lengths: List[float], offsets: List[float]) -> Tuple[float, float, float, float, float, float, float]:
        """
        Compute and return geometric and trigonometric values used in IK calculations.

        Returns:
        --------
        distance_squared_24, distance_squared_46, length_24, length_46, angle_theta_h46, angle_theta_342, angle_theta_46h
        """
        wrist_offset = offsets[0]
        upper_arm_length = link_lengths[1]
        forearm_length = link_lengths[2]

        distance_squared_24 = wrist_offset ** 2 + upper_arm_length ** 2
        distance_squared_46 = wrist_offset ** 2 + forearm_length ** 2
        length_24 = np.sqrt(distance_squared_24)
        length_46 = np.sqrt(distance_squared_46)
        angle_theta_h46 = np.arctan(forearm_length / wrist_offset)
        angle_theta_342 = np.arctan(upper_arm_length / wrist_offset)
        angle_theta_46h = np.arctan(wrist_offset / forearm_length)

        return distance_squared_24, distance_squared_46, length_24, length_46, angle_theta_h46, angle_theta_342, angle_theta_46h
