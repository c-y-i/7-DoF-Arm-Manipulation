import numpy as np
from typing import Tuple, List

class FrankaIK:
    @staticmethod
    def compute_ik(transform_array: List[float], joint_7_angle: float, reference_angles: List[float]) -> np.ndarray:
        print("Computing IK for Franka Panda robot...")
        link_lengths = [0.3330, 0.3160, 0.3840, 0.2104]
        link_offsets = [0.0825, 0.0880]

        dist_sq_24, dist_sq_46, length_24, length_46, angle_h46, angle_342, angle_46h = FrankaIK._compute_geometry_terms(link_lengths, link_offsets)

        joint_limits = [
            [-2.8973,  2.8973],
            [-1.7628,  1.7628],
            [-2.8973,  2.8973],
            [-3.0718, -0.0698],
            [-2.8973,  2.8973],
            [-0.0175,  3.7525],
            [-2.8973,  2.8973]
        ]

        solutions = np.full((4, 7), np.nan)
        nan_row = np.array([np.nan] * 7)

        if not FrankaIK._within_joint_limits(joint_7_angle, joint_limits[6]):
            return solutions
        solutions[:, 6] = joint_7_angle

        base_to_ee_transform = np.array(transform_array).reshape((4, 4))
        ee_rotation = base_to_ee_transform[:3, :3]
        ee_z_axis = base_to_ee_transform[:3, 2]
        ee_position = base_to_ee_transform[:3, 3]

        joint_7_position = ee_position - link_lengths[3] * ee_z_axis

        x6_dir = np.array([np.cos(joint_7_angle - np.pi / 4), -np.sin(joint_7_angle - np.pi / 4), 0.0])
        joint_6_x_axis = ee_rotation @ x6_dir
        joint_6_x_axis /= np.linalg.norm(joint_6_x_axis)
        joint_6_reference = joint_7_position - link_offsets[1] * joint_6_x_axis

        joint_2_position = np.array([0.0, 0.0, link_lengths[0]])
        vector_2_to_6 = joint_6_reference - joint_2_position
        dist_sq_26 = np.sum(vector_2_to_6 ** 2)
        length_26 = np.sqrt(dist_sq_26)

        if length_24 + length_46 < length_26 or length_24 + length_26 < length_46 or length_26 + length_46 < length_24:
            return solutions

        angle_246 = np.arccos((dist_sq_24 + dist_sq_46 - dist_sq_26) / (2.0 * length_24 * length_46))
        joint_4_angle = angle_246 + angle_h46 + angle_342 - 2.0 * np.pi
        if not FrankaIK._within_joint_limits(joint_4_angle, joint_limits[3]):
            return solutions
        solutions[:, 3] = joint_4_angle

        angle_462 = np.arccos((dist_sq_26 + dist_sq_46 - dist_sq_24) / (2.0 * length_26 * length_46))
        angle_26h = angle_46h + angle_462
        dis_26 = -length_26 * np.cos(angle_26h)

        z_axis_6 = np.cross(ee_z_axis, joint_6_x_axis)
        y_axis_6 = np.cross(z_axis_6, joint_6_x_axis)
        rotation_6 = np.column_stack((
            joint_6_x_axis,
            y_axis_6 / np.linalg.norm(y_axis_6),
            z_axis_6 / np.linalg.norm(z_axis_6)
        ))
        vector_6_to_2 = rotation_6.T @ (-vector_2_to_6)
        phi_6 = np.arctan2(vector_6_to_2[1], vector_6_to_2[0])
        theta_6 = np.arcsin(dis_26 / np.sqrt(vector_6_to_2[0] ** 2 + vector_6_to_2[1] ** 2))
        joint_6_candidates = np.array([np.pi - theta_6 - phi_6, theta_6 - phi_6])

        for i, joint_6_angle in enumerate(joint_6_candidates):
            if joint_6_angle <= joint_limits[5][0]:
                joint_6_angle += 2.0 * np.pi
            elif joint_6_angle >= joint_limits[5][1]:
                joint_6_angle -= 2.0 * np.pi

            if joint_6_angle <= joint_limits[5][0] or joint_6_angle >= joint_limits[5][1]:
                solutions[2 * i, 5] = np.nan
                solutions[2 * i + 1, 5] = np.nan
            else:
                solutions[2 * i, 5] = joint_6_angle
                solutions[2 * i + 1, 5] = joint_6_angle

        if np.isnan(solutions[0, 5]) and np.isnan(solutions[2, 5]):
            return solutions

        angle_p26 = (3.0 * np.pi / 2) - angle_462 - angle_246 - angle_342
        angle_p = np.pi - angle_p26 - angle_26h
        length_p6 = length_26 * np.sin(angle_p26) / np.sin(angle_p)

        z_axis_5_storage = np.zeros((4, 3))
        vector_2_to_p_storage = np.zeros((4, 3))

        for i in range(2):
            joint_6_angle = solutions[2 * i, 5]
            if np.isnan(joint_6_angle):
                continue

            z_axis_6_5 = np.array([np.sin(joint_6_angle), np.cos(joint_6_angle), 0.0])
            z_axis_5 = rotation_6 @ z_axis_6_5
            vector_2_to_p = joint_6_reference - length_p6 * z_axis_5 - joint_2_position

            z_axis_5_storage[2 * i] = z_axis_5
            z_axis_5_storage[2 * i + 1] = z_axis_5
            vector_2_to_p_storage[2 * i] = vector_2_to_p
            vector_2_to_p_storage[2 * i + 1] = vector_2_to_p

            length_2p = np.linalg.norm(vector_2_to_p)
            if np.abs(vector_2_to_p[2] / length_2p) > 0.999:
                solutions[2 * i, 0] = reference_angles[0]
                solutions[2 * i, 1] = 0.0
                solutions[2 * i + 1, 0] = reference_angles[0]
                solutions[2 * i + 1, 1] = 0.0
            else:
                solutions[2 * i, 0] = np.arctan2(vector_2_to_p[1], vector_2_to_p[0])
                solutions[2 * i, 1] = np.arccos(vector_2_to_p[2] / length_2p)
                if solutions[2 * i, 0] < 0:
                    solutions[2 * i + 1, 0] = solutions[2 * i, 0] + np.pi
                else:
                    solutions[2 * i + 1, 0] = solutions[2 * i, 0] - np.pi
                solutions[2 * i + 1, 1] = -solutions[2 * i, 1]

        for i in range(4):
            if (solutions[i, 0] <= joint_limits[0][0] or solutions[i, 0] >= joint_limits[0][1] or
                solutions[i, 1] <= joint_limits[1][0] or solutions[i, 1] >= joint_limits[1][1]):
                solutions[i] = nan_row
                continue

            vector_2_to_p = vector_2_to_p_storage[i]
            if np.isnan(solutions[i, 5]):
                solutions[i] = nan_row
                continue

            joint_3_direction = vector_2_to_p / np.linalg.norm(vector_2_to_p)

            joint_y3 = -np.cross(vector_2_to_6, vector_2_to_p)
            joint_y3 = joint_y3 / np.linalg.norm(joint_y3)
            joint_x3 = np.cross(joint_y3, joint_3_direction)

            rotation_1 = np.array([
                [np.cos(solutions[i, 0]), -np.sin(solutions[i, 0]), 0.0],
                [np.sin(solutions[i, 0]),  np.cos(solutions[i, 0]), 0.0],
                [0.0, 0.0, 1.0]
            ])
            rotation_1_2 = np.array([
                [np.cos(solutions[i, 1]), -np.sin(solutions[i, 1]), 0.0],
                [0.0, 0.0, 1.0],
                [-np.sin(solutions[i, 1]), -np.cos(solutions[i, 1]), 0.0]
            ])
            rotation_2 = rotation_1 @ rotation_1_2

            x_axis_2_3 = rotation_2.T @ joint_x3
            joint_3_angle = np.arctan2(x_axis_2_3[2], x_axis_2_3[0])
            if not FrankaIK._within_joint_limits(joint_3_angle, joint_limits[2]):
                solutions[i] = nan_row
                continue
            solutions[i, 2] = joint_3_angle

            vector_4 = (
                joint_2_position +
                link_lengths[1] * joint_3_direction +
                link_offsets[0] * joint_x3 -
                joint_6_reference +
                link_lengths[2] * z_axis_5_storage[i]
            )

            joint_6_angle = solutions[i, 5]
            rotation_56 = np.array([
                [np.cos(joint_6_angle), -np.sin(joint_6_angle), 0.0],
                [0.0, 0.0, -1.0],
                [np.sin(joint_6_angle), np.cos(joint_6_angle),  0.0]
            ])
            rotation_5 = rotation_6 @ rotation_56.T
            vector_5_h4 = rotation_5.T @ vector_4
            joint_5_angle = -np.arctan2(vector_5_h4[1], vector_5_h4[0])

            if not FrankaIK._within_joint_limits(joint_5_angle, joint_limits[4]):
                solutions[i] = nan_row
                continue
            solutions[i, 4] = joint_5_angle

        return solutions

    @staticmethod
    def _within_joint_limits(joint_angle: float, limit: List[float]) -> bool:
        return limit[0] <= joint_angle <= limit[1]

    @staticmethod
    def _compute_geometry_terms(link_lengths: List[float], link_offsets: List[float]) -> Tuple[float, float, float, float, float, float, float]:
        wrist_offset = link_offsets[0]
        upper_arm_length = link_lengths[1]
        forearm_length = link_lengths[2]

        dist_sq_24 = wrist_offset ** 2 + upper_arm_length ** 2
        dist_sq_46 = wrist_offset ** 2 + forearm_length ** 2
        length_24 = np.sqrt(dist_sq_24)
        length_46 = np.sqrt(dist_sq_46)
        angle_h46 = np.arctan(forearm_length / wrist_offset)
        angle_342 = np.arctan(upper_arm_length / wrist_offset)
        angle_46h = np.arctan(wrist_offset / forearm_length)

        return dist_sq_24, dist_sq_46, length_24, length_46, angle_h46, angle_342, angle_46h
