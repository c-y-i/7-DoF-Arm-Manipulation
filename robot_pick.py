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

    def monitor_blocks(self):
        """Update block detections."""
        try:
            detections = self.detector.get_detections()
            if detections:
                blocks = []
                for block_id, block_pose in detections:
                    # Transform block pose from camera frame to world frame
                    H_ee_camera = self.detector.get_H_ee_camera()
                    T_base_ee = self.arm.get_current_transformation()
                    T_base_camera = T_base_ee @ H_ee_camera
                    block_pose_world = T_base_camera @ block_pose

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

    def visualize_gripper_and_block(self, block_transform):
        """Visualize the gripper and the block in RViz."""
        # Visualize the block as a marker
        block_marker = visualization_msgs.msg.Marker()
        block_marker.header.frame_id = "world"
        block_marker.header.stamp = rospy.Time.now()
        block_marker.id = 0
        block_marker.type = visualization_msgs.msg.Marker.CUBE
        block_marker.action = visualization_msgs.msg.Marker.ADD
        block_marker.pose.position.x = block_transform[0, 3]
        block_marker.pose.position.y = block_transform[1, 3]
        block_marker.pose.position.z = block_transform[2, 3]
        quaternion = tf.transformations.quaternion_from_matrix(block_transform)
        block_marker.pose.orientation.x = quaternion[0]
        block_marker.pose.orientation.y = quaternion[1]
        block_marker.pose.orientation.z = quaternion[2]
        block_marker.pose.orientation.w = quaternion[3]
        block_marker.scale.x = 0.04  # Block dimensions
        block_marker.scale.y = 0.04
        block_marker.scale.z = 0.04
        block_marker.color.r = 1.0
        block_marker.color.g = 0.0
        block_marker.color.b = 0.0
        block_marker.color.a = 1.0
        self.target_pub.publish(block_marker)

        # Visualize the gripper as a marker
        gripper_pose = self.arm.get_current_transformation()
        gripper_marker = visualization_msgs.msg.Marker()
        gripper_marker.header.frame_id = "world"
        gripper_marker.header.stamp = rospy.Time.now()
        gripper_marker.id = 1
        gripper_marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        gripper_marker.mesh_resource = "package://franka_description/meshes/hand/hand.dae"
        gripper_marker.action = visualization_msgs.msg.Marker.ADD
        gripper_marker.pose.position.x = gripper_pose[0, 3]
        gripper_marker.pose.position.y = gripper_pose[1, 3]
        gripper_marker.pose.position.z = gripper_pose[2, 3]
        quaternion = tf.transformations.quaternion_from_matrix(gripper_pose)
        gripper_marker.pose.orientation.x = quaternion[0]
        gripper_marker.pose.orientation.y = quaternion[1]
        gripper_marker.pose.orientation.z = quaternion[2]
        gripper_marker.pose.orientation.w = quaternion[3]
        gripper_marker.scale.x = 1.0
        gripper_marker.scale.y = 1.0
        gripper_marker.scale.z = 1.0
        gripper_marker.color.r = 0.5
        gripper_marker.color.g = 0.5
        gripper_marker.color.b = 0.5
        gripper_marker.color.a = 1.0
        self.target_pub.publish(gripper_marker)

    def move_to_observation_pose(self):
        """Move the robot to a position suitable for observing blocks."""
        # Define observation pose (adjust these coordinates as needed)
        obs_position = np.array([0.5, -0.1, 0.5])  # Above the workspace
        # Use a downward-facing orientation
        obs_orientation = R.from_euler('xyz', [pi, 0, 0]).as_matrix()
        
        obs_pose = np.eye(4)
        obs_pose[:3, :3] = obs_orientation
        obs_pose[:3, 3] = obs_position
        
        print("Moving to observation position...")
        try:
            self.move_to_pose(obs_pose)
            print("Successfully moved to observation position")
        except Exception as e:
            print(f"Failed to move to observation position: {e}")

    def run(self):
        """Main routine to execute pick and place operations."""
        rospy.sleep(1)  # Wait for the system to be ready
        
        # Move to observation pose
        self.move_to_observation_pose()
        rospy.sleep(2)  # Wait for the system to settle

        place_position = np.array([0.5, 0.0, 0.0])
        pick_and_place_done = False

        while not rospy.is_shutdown():
            # Update block detections
            blocks_found = self.monitor_blocks()
            
            # Execute pick and place once blocks are detected
            if blocks_found and not pick_and_place_done:
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
