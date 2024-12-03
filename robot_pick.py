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
from detect_visual import DetectVisualTester  # Import the visualization class
import detect_pick

class RobotPickAndPlace:
    """Robot pick and place controller that detects blocks and executes pick and place operations."""

    def __init__(self):
        """Initialize the pick and place controller."""
        self.arm = ArmController()
        self.detector = ObjectDetector()
        
        self.fk = FK()
        self.ik = IK(linear_tol=1e-3, angular_tol=1e-2, max_steps=200)
        self.tf_broad = tf.TransformBroadcaster()
        self.target_pub = rospy.Publisher('/vis/target', visualization_msgs.msg.Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/vis/trajectory', visualization_msgs.msg.Marker, queue_size=10)
        self.latest_blocks = []
        self.detection_rate = rospy.Rate(10)  # 10Hz = 100ms period
        self.blocks_detected = False

    def get_current_transformation(self):
        """Get the current end-effector transformation matrix using FK."""
        current_joints = self.arm.get_positions()
        # Use FK to get the transformation matrix
        joint_positions = np.array([current_joints])  # FK expects a 2D array
        T, _ = self.fk.forward(joint_positions)
        return np.array(T)  # Convert to numpy array to ensure proper matrix operations

    def monitor_blocks(self):
        """Update block detections."""
        try:
            detections = self.detector.get_detections()
            if detections:
                blocks = []
                for block_id, block_pose in detections:
                    print(block_pose)
                    # Transform block pose from camera frame to world frame
                    H_ee_camera = self.detector.get_H_ee_camera()
                    T_base_ee = self.get_current_transformation()
                    
                    # Ensure all matrices are numpy arrays
                    H_ee_camera = np.array(H_ee_camera)
                    block_pose = np.array(block_pose)
                    
                    T_base_camera = np.matmul(T_base_ee, H_ee_camera)
                    block_pose_world = np.matmul(T_base_camera, block_pose)

                    position = block_pose_world[:3, 3]
                    rotation_matrix = block_pose_world[:3, :3]
                    orientation = R.from_matrix(rotation_matrix).as_euler('xyz')

                    block_data = {
                        'id': block_id,
                        'position': position,
                        'orientation': orientation,
                        'transform': block_pose_world
                    }
                    blocks.append(block_data)
                self.latest_blocks = blocks
                self.blocks_detected = True
                return True
        except Exception as e:
            rospy.logwarn(f"Error in monitor_blocks: {e}")
            rospy.logwarn(f"Stack trace:", exc_info=True)  # Add this to get more error details
        return False

    def get_block_detections(self):
        """Get the latest block detections."""
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

    def move_to_pose(self, target_pose, current_joints=None):
        """Move the robot to a target pose using inverse kinematics."""
        if current_joints is None:
            current_joints = self.arm.get_positions()
        
        # Call inverse with correct argument order: target, seed, method, alpha
        joints, _, success, _ = self.ik.inverse(
            target=target_pose,
            seed=current_joints,
            method='J_pseudo',
            alpha=0.1
        )
        
        if success:
            self.arm.safe_move_to_position(joints)
            return joints
        else:
            raise Exception("IK failed to find a solution.")

    def pick_block(self, block):
        """Execute the pick operation for a given block."""
        # Open gripper
        self.arm.open_gripper()

        # Compute pre-grasp pose above the block
        pre_grasp_position = block['position'] + np.array([0, 0, 0.1])  # Approach from above
        desired_orientation = self.align_gripper_with_block(block['orientation'])
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = desired_orientation
        pre_grasp_pose[:3, 3] = pre_grasp_position

        # Move to pre-grasp pose
        self.move_to_pose(pre_grasp_pose)

        # Move to grasp pose
        grasp_pose = np.copy(pre_grasp_pose)
        grasp_pose[:3, 3] = block['position'] + np.array([0, 0, 0.02])  # Adjust for block height
        self.move_to_pose(grasp_pose)

        # Close gripper to grasp the block
        self.arm.close_gripper()

        # Lift the block
        self.move_to_pose(pre_grasp_pose)

        # Visualize the gripper and block positions
        gripper_transform = self.arm.get_current_transformation()
        visualizer = DetectVisualTester()
        visualizer.visualize_gripper_and_block(block, gripper_transform)

    def place_block(self, place_position, block_orientation):
        """Execute the place operation at the specified position and orientation."""
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
            obs_position = np.array([0.5, -0.1, 0.5])  # Even more conservative position
            obs_orientation = R.from_euler('xyz', [pi, 0, 0]).as_matrix()  # Simpler orientation
            
            obs_pose = np.eye(4)
            obs_pose[:3, :3] = obs_orientation
            obs_pose[:3, 3] = obs_position
            
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
        blocks = {}
        place_position = np.array([0.5, 0.0, 0.0])
        pick_and_place_done = False
        while not rospy.is_shutdown():
            # Update block detections
            detections = self.detector.get_detections()

            block_data = detect_pick.Detection().scan_blocks(1, detections) if detections else None
            print(block_data)
            blocks = block_data
            # Execute pick and place once blocks are detected
            if blocks and not pick_and_place_done:
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