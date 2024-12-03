"""
Pick logic for the robot arm.
Author: Team 16

"""

import numpy as np
from math import pi
import rospy
import tf
import geometry_msgs.msg
import visualization_msgs.msg
from scipy.spatial.transform import Rotation as R
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.calculateFK import FK
from lib.IK_position_null import IK

class StaticBlockHandler:
    def __init__(self, fk_solver, ik_solver):
        """Initialize with FK and IK solvers from the main script."""
        self.fk_solver = fk_solver
        self.ik_solver = ik_solver

    def plan_observation_pose(self):
        """Calculate joint angles for observation pose."""
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
        
        # Adjust block position and compute poses
        adjusted_position = np.copy(block_pose)
        adjusted_position[2] = max(adjusted_position[2], 0.05)
        
        # Pre-grasp pose
        pre_grasp_position = adjusted_position + np.array([0, 0, 0.1])
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = desired_orientation
        pre_grasp_pose[:3, 3] = pre_grasp_position
        
        # Grasp pose
        grasp_position = adjusted_position + np.array([0, 0, 0.015])
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = desired_orientation
        grasp_pose[:3, 3] = grasp_position
        
        # Calculate joint angles for each pose
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

class RobotPickAndPlace:
    """Robot pick and place controller that detects blocks and executes pick and place operations."""

    def __init__(self):
        """Initialize the pick and place controller."""
        self.arm = ArmController()
        self.detector = ObjectDetector()
        
        self.fk = FK()
        self.ik = IK(linear_tol=1e-3, angular_tol=1e-2, max_steps=1000)
        self.tf_broad = tf.TransformBroadcaster()
        self.target_pub = rospy.Publisher('/vis/target', visualization_msgs.msg.Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/vis/trajectory', visualization_msgs.msg.Marker, queue_size=10)
        self.latest_blocks = []
        self.detection_rate = rospy.Rate(10)  # 10Hz = 100ms period
        self.blocks_detected = False
        self.block_handler = StaticBlockHandler(self.fk, self.ik)

    def get_current_transformation(self):
        """Get the current end-effector transformation matrix using FK."""
        current_joints = self.arm.get_positions()
        # Use FK to get the transformation matrix
        joint_positions = np.array([current_joints])  # FK expects a 2D array
        T, _ = self.fk.forward(joint_positions)
        return np.array(T)  # Convert to numpy array to ensure proper matrix operations

    def get_block_detections(self):
        """Get the latest block detections."""
        self.latest_blocks = self.detect_static_blocks()
        return self.latest_blocks

    def align_gripper_with_block(self, block_orientation):
        """Compute the required wrist orientation to align the gripper with the block's orientation."""
        try:
            # Construct the desired rotation matrix from block orientation
            desired_rotation = R.from_euler('xyz', block_orientation).as_matrix()
            return desired_rotation
        except Exception as e:
            print(f"Error in align_gripper_with_block: {e}")
            # Return default orientation if error occurs
            return np.eye(3)

    def move_to_pose(self, target_pose, seed=None):
        """Move the robot to a target pose using inverse kinematics."""
        if seed is None:
            seed = self.arm.get_positions()
        
        # Adjust IK parameters for better convergence
        joints, _, success, _ = self.ik.inverse(
            target=target_pose,
            seed=seed,
            method='J_pseudo',
            alpha=0.1
        )
        
        if success:
            self.arm.safe_move_to_position(joints)
            return joints
        else:
            raise Exception("IK failed to find a solution.")

    def transform_to_base_frame(self, position):
        """Transform position from camera frame to robot base frame."""
        H_ee_camera = self.detector.get_H_ee_camera()
        T_EE_C = H_ee_camera
        current_joints = self.arm.get_positions()
        _, T_BASE_EE = self.fk.forward([current_joints])
        T_BASE_C = T_BASE_EE[-1] @ T_EE_C
        position_homogeneous = np.append(position, 1)
        base_position = T_BASE_C @ position_homogeneous
        return base_position[:3]

    def detect_static_blocks(self):
        """Detect static blocks and return their positions."""
        detections = self.detector.get_detections()
        static_blocks = []
        for block_id, matrix in detections:
            position = matrix[0:3, 3]  # Extract position as a 3-element vector
            # Transform position to robot base frame
            position = self.transform_to_base_frame(position)
            static_blocks.append({
                'id': block_id,
                'position': position,
                'orientation': [0, 0, 0]  # Assuming upright orientation
            })
        return static_blocks

    def pick_and_place_static_blocks(self, place_position):
        """Detect static blocks and execute pick and place for one block at a time."""
        static_blocks = self.detect_static_blocks()
        if not static_blocks:
            print("No static blocks detected.")
            return

        # Pick and place one block at a time
        block = static_blocks[0]  # Pick the first block
        print(f"Picking block {block['id']} at position {block['position']}")
        try:
            self.pick_block(block)
            print(f"Placing block {block['id']} at position {place_position}")
            self.place_block(place_position, block['orientation'])
        except Exception as e:
            print(f"Failed to pick and place block {block['id']}: {e}")

    def pick_block(self, block):
        """Execute the pick operation for a given block."""
        # Use a fixed, reachable orientation
        desired_orientation = R.from_euler('xyz', [pi, 0, 0]).as_matrix()
        
        # Adjust block position to ensure it's within reach
        adjusted_block_position = np.copy(block['position'])
        adjusted_block_position[2] = max(adjusted_block_position[2], 0.05)  # Ensure Z is above a minimum height

        # Compute pre-grasp pose above the block
        pre_grasp_position = adjusted_block_position + np.array([0, 0, 0.1])  # Approach from above
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = desired_orientation
        pre_grasp_pose[:3, 3] = pre_grasp_position

        # Move to pre-grasp pose with an appropriate seed
        seed = self.arm.neutral_position()
        try:
            self.move_to_pose(pre_grasp_pose, seed=seed)
        except Exception as e:
            raise Exception(f"IK failed for pre-grasp pose: {e}")

        # Open gripper
        self.arm.open_gripper()

        # Compute grasp pose at the block position
        grasp_position = adjusted_block_position + np.array([0, 0, 0.015])  # Adjust for block height
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = desired_orientation
        grasp_pose[:3, 3] = grasp_position

        # Move to grasp pose
        try:
            self.move_to_pose(grasp_pose, seed=seed)
        except Exception as e:
            raise Exception(f"IK failed for grasp pose: {e}")

        # Close gripper to grasp the block
        self.arm.close_gripper()

        # Lift the block back to pre-grasp pose
        self.move_to_pose(pre_grasp_pose, seed=seed)

    def place_block(self, place_position, block_orientation):
        """Execute the place operation at the specified position."""
        # Compute pre-place pose above the target location
        pre_place_position = place_position + np.array([0, 0, 0.1])
        desired_orientation = self.align_gripper_with_block(block_orientation)
        pre_place_pose = np.eye(4)
        pre_place_pose[:3, :3] = desired_orientation
        pre_place_pose[:3, 3] = pre_place_position

        # Move to pre-place pose
        self.move_to_pose(pre_place_pose)

        # Move to place pose
        place_pose = np.copy(pre_place_pose)
        place_pose[:3, 3] = place_position + np.array([0, 0, 0.02])
        self.move_to_pose(place_pose)

        # Open gripper to release the block
        self.arm.open_gripper()

        # Retract to pre-place pose
        self.move_to_pose(pre_place_pose)

    def execute_pick_and_place(self, place_position):
        """Detect blocks and execute pick and place operations."""
        blocks = self.get_block_detections()
        if not blocks:
            print("No blocks detected.")
            return

        for block in blocks:
            print(f"Picking block {block['id']} at position {block['position']}")
            try:
                self.pick_block(block)
                print(f"Placing block {block['id']} at position {place_position}")
                self.place_block(place_position, block['orientation'])
            except Exception as e:
                print(f"Failed to pick and place block {block['id']}: {e}")

    def move_to_observation_pose(self):
        """Move the robot to a position suitable for observing blocks."""
        try:
            # First try to get current position and print it
            current_pos = self.arm.get_positions()
            print("Current joint positions:", current_pos)

            # Move to neutral with direct joint commands
            neutral = self.arm.neutral_position()
            print("Neutral position:", neutral)
            
            # # Move joints incrementally towards neutral
            # steps = 2
            # for i in range(steps + 1):
            #     fraction = i / steps
            #     intermediate = current_pos + (neutral - current_pos) * fraction
            #     print(f"Moving to intermediate position {i}/{steps}")
            #     self.arm.safe_move_to_position(intermediate)
            #     rospy.sleep(0.2)
            
            print("Reached neutral position")
            rospy.sleep(0.5)  # Wait for stability
            
            # Now move to observation pose
            obs_pose = self.block_handler.plan_observation_pose()
            
            print("Computing IK for observation pose...")
            joints, _, success, message = self.ik.inverse(
                obs_pose,
                neutral,  # Use neutral position as seed
                method='J_pseudo',
                alpha=0.5  # Increased alpha for better convergence
            )
            
            if success:
                print("Moving to observation pose...")
                self.arm.safe_move_to_position(joints)
                print("Successfully reached observation pose")
                return True
            else:
                print(f"IK failed to find solution: {message}")
                return False
                
        except Exception as e:
            print(f"Error during movement: {str(e)}")
            return False

    def run(self):
        """Main routine to execute pick and place operations."""
        rospy.sleep(1)  # Wait for the system to be ready
        
        # Move to observation pose with error handling
        if not self.move_to_observation_pose():
            print("Could not reach observation pose, aborting...")
            return
        
        rospy.sleep(2)  # Wait for the system to settle
        place_position = np.array([0.5, 0.0, 0.1])  # Starting place position
        self.pick_and_place_static_blocks(place_position)
        pick_and_place_done = False
        while not rospy.is_shutdown():
            # Update block detections
            # Execute pick and place once blocks are detected
            if self.latest_blocks and not pick_and_place_done:
                self.execute_pick_and_place(place_position)
                pick_and_place_done = True
            
            self.detection_rate.sleep()

def main():
    """Main function to run the robot pick and place module."""
    rospy.init_node("robot_pick_and_place")
    controller = RobotPickAndPlace()
    controller.run()

if __name__ == "__main__":
    main()