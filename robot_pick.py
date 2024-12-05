"""
Pick logic library for the robot arm.
Author: Team 16
"""

import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
from time import sleep

class StaticBlockHandler:
    def __init__(self, fk_solver, ik_solver):
        """Initialize with FK and IK solvers from the main script.""" 
        self.fk_solver = fk_solver
        self.ik_solver = ik_solver

    def plan_observation_pose(self):
        """Calculate observation pose.""" 
        obs_position = np.array([0.5, -0.1, 0.5])
        obs_orientation = R.from_euler('xyz', [pi, 0, 0]).as_matrix()
        
        obs_pose = np.eye(4)
        obs_pose[:3, :3] = obs_orientation
        obs_pose[:3, 3] = obs_position
        
        return obs_pose

    def transform_to_base_frame(self, position, current_joints, H_ee_camera):
        """Transform position from camera frame to robot base frame.""" 
        T_EE_C = H_ee_camera
        _, T_BASE_EE = self.fk_solver.forward([current_joints])
        T_BASE_C = T_BASE_EE[-1] @ T_EE_C
        position_homogeneous = np.append(position, 1)
        base_position = T_BASE_C @ position_homogeneous
        return base_position[:3]

    def plan_block_pickup(self, block_pose, seed_joints):
        """Plan the sequence of poses for picking up a block.""" 
        desired_orientation = R.from_euler('xyz', [pi, 0, 0]).as_matrix()
        
        adjusted_position = np.copy(block_pose)
        adjusted_position[2] = max(adjusted_position[2], 0.05)
        
        pre_grasp_position = adjusted_position + np.array([0, 0, 0.1])
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = desired_orientation
        pre_grasp_pose[:3, 3] = pre_grasp_position
        
        grasp_position = adjusted_position + np.array([0, 0, 0.015])
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = desired_orientation
        grasp_pose[:3, 3] = grasp_position
        
        pre_grasp_joints, _, pre_success, _ = self.ik_solver.inverse(
            pre_grasp_pose, seed_joints, method='J_pseudo', alpha=0.1)
        grasp_joints, _, grasp_success, _ = self.ik_solver.inverse(
            grasp_pose, pre_grasp_joints if pre_success else seed_joints, 
            method='J_pseudo', alpha=0.1)
        
        return {
            'pre_grasp_pose': pre_grasp_pose,
            'grasp_pose': grasp_pose,
            'pre_grasp_joints': pre_grasp_joints if pre_success else None,
            'grasp_joints': grasp_joints if grasp_success else None,
            'success': pre_success and grasp_success
        }

    def plan_block_placement(self, place_position, seed_joints):
        """Plan the sequence of poses for placing a block.""" 
        desired_orientation = R.from_euler('xyz', [pi, 0, 0]).as_matrix()
        
        # Pre-place pose
        pre_place_position = place_position + np.array([0, 0, 0.1])
        pre_place_pose = np.eye(4)
        pre_place_pose[:3, :3] = desired_orientation
        pre_place_pose[:3, 3] = pre_place_position
        
        # Place pose
        place_pose = np.eye(4)
        place_pose[:3, :3] = desired_orientation
        place_pose[:3, 3] = place_position + np.array([0, 0, 0.02])
        
        # Calculate joint angles for each pose
        pre_place_joints, _, pre_success, _ = self.ik_solver.inverse(
            pre_place_pose, seed_joints, method='J_pseudo', alpha=0.1)
        place_joints, _, place_success, _ = self.ik_solver.inverse(
            place_pose, pre_place_joints if pre_success else seed_joints, 
            method='J_pseudo', alpha=0.1)
        
        return {
            'pre_place_pose': pre_place_pose,
            'place_pose': place_pose,
            'pre_place_joints': pre_place_joints if pre_success else None,
            'place_joints': place_joints if place_success else None,
            'success': pre_success and place_success
        }

    def detect_static(self, detector, T_CW, num_samples=5, detection_threshold=0.8, sleep_time=0.1):
        """Block detection with temporal filtering and outlier rejection.""" 
        all_detections = []
        valid_blocks = {}

        # Collect multiple samples
        for _ in range(num_samples):
            current_detections = []
            for block, position in detector.get_detections():
                if isinstance(position, np.ndarray) and position.shape == (4, 4):
                    # Transform position to world frame
                    world_pos = T_CW @ position
                    current_detections.append({
                        'id': block,
                        'pose': world_pos,
                        'position': world_pos[:3, 3]
                    })
            all_detections.append(current_detections)
            sleep(sleep_time)  # Replace rospy.sleep with time.sleep

        # Process and filter detections
        for sample in all_detections:
            for block in sample:
                block_id = block['id']
                if block_id not in valid_blocks:
                    valid_blocks[block_id] = []
                valid_blocks[block_id].append(block['position'])

        filtered_blocks = []
        for block_id, positions in valid_blocks.items():
            if len(positions) >= num_samples * detection_threshold:
                positions = np.array(positions)
                median_pos = np.median(positions, axis=0)
                filtered_blocks.append({
                    'id': block_id,
                    'position': median_pos,
                    'confidence': len(positions) / num_samples
                })

        return sorted(filtered_blocks, key=lambda x: x['confidence'], reverse=True)

    def pick_place_static(self, arm, blocks, place_target, grasp_sleep=0.5, pre_grasp_sleep=0.2, place_sleep=0.3):
        """Execute pick and place sequence for static blocks.""" 
        stack_height = place_target[2]
        current_joints = arm.get_positions()

        for block in blocks:
            # Plan pick-up
            seed_joints = current_joints
            pick_plan = self.plan_block_pickup(block['position'], seed_joints)
            if not pick_plan['success']:
                print(f"Failed to plan pick-up for block {block['id']}")
                continue

            # Execute pick sequence
            arm.safe_move_to_position(pick_plan['pre_grasp_joints'])
            arm.open_gripper()
            sleep(pre_grasp_sleep)  # Replace rospy.sleep

            arm.safe_move_to_position(pick_plan['grasp_joints'])
            arm.exec_gripper_cmd(0.045, 100)
            sleep(grasp_sleep)  # Replace rospy.sleep

            # Check grasp success and plan placement
            place_plan = self.plan_block_placement(place_target, pick_plan['grasp_joints'])
            if not place_plan['success']:
                print(f"Failed to plan placement for block {block['id']}")
                continue

            # Execute place sequence
            arm.safe_move_to_position(place_plan['pre_place_joints'])
            arm.safe_move_to_position(place_plan['place_joints'])
            arm.open_gripper()
            sleep(place_sleep)  # Replace rospy.sleep
            arm.safe_move_to_position(place_plan['pre_place_joints'])

            current_joints = place_plan['pre_place_joints']
            place_target[2] += 0.05

        return True
