"""
Pick logic for the robot arm.
Author: Team 16

NOTE: This is mostly by Copilot, i have no idea whether this will work or not, 
Simulation on VM does not work on my laptop. - Yi (11/23/2024)
"""

import numpy as np
from math import pi
import rospy
import tf
import geometry_msgs.msg
import visualization_msgs
from tf.transformations import quaternion_from_matrix
from core.interfaces import ArmController
from lib.calculateFK import FK
from lib.IK_position_null import IK

class RobotPickAndPlace:
    def __init__(self):
        """Initialize the pick and place controller."""
        print("\nInitializing Robot Pick and Place Controller...")
        self.arm = ArmController()
        self.fk = FK()
        self.ik = IK(linear_tol=1e-3, angular_tol=1e-2, max_steps=200)
        
        # Add visualization publishers
        self.tf_broad = tf.TransformBroadcaster()
        self.target_pub = rospy.Publisher('/vis/target', visualization_msgs.msg.Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/vis/trajectory', visualization_msgs.msg.Marker, queue_size=10)
        
    def generate_trajectory(self, pick_pos, place_pos):
        """Generate waypoints for pick and place operation."""
        print("\nGenerating trajectory waypoints...")
        
        # Define waypoints with safety height offset
        z_offset = 0.3
        start_pos = np.array([0.3, 0, 0.5])  # Starting position
        
        waypoints = [
            start_pos,
            pick_pos + np.array([0, 0, z_offset]),  # Pre-pick
            pick_pos,  # Pick position
            pick_pos + np.array([0, 0, z_offset]),  # Post-pick
            place_pos + np.array([0, 0, z_offset]),  # Pre-place
            place_pos  # Place position
        ]
        
        return waypoints
    
    def move_to_position(self, target_pos, current_joints=None):
        """Move the robot to a target position with IK."""
        target = np.eye(4)
        target[:3,3] = target_pos
        target[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # End effector orientation
        
        if current_joints is None:
            current_joints = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
            
        joints, _, success, _ = self.ik.inverse(target, current_joints, method='J_pseudo', alpha=0.1)
        
        if success:
            self.arm.safe_move_to_position(joints)
            return joints
        else:
            raise Exception(f"IK failed for position {target_pos}")
    
    def show_pose(self, H, frame):
        """Broadcast a frame using the transform from given frame to world frame."""
        self.tf_broad.sendTransform(
            tf.transformations.translation_from_matrix(H),
            tf.transformations.quaternion_from_matrix(H),
            rospy.Time.now(),
            frame,
            "world"
        )
        
    def visualize_target(self, position, color=[1.0, 0.0, 0.0], scale=0.05):
        """Visualize target position as a sphere in RViz."""
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.type = visualization_msgs.msg.Marker.SPHERE
        marker.action = visualization_msgs.msg.Marker.ADD
        
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8
        
        self.target_pub.publish(marker)
        
    def visualize_trajectory(self, waypoints):
        """Visualize planned trajectory in RViz."""
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.type = visualization_msgs.msg.Marker.LINE_STRIP
        marker.action = visualization_msgs.msg.Marker.ADD
        
        for point in waypoints:
            p = geometry_msgs.msg.Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)
            
        marker.scale.x = 0.01  # line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        
        self.trajectory_pub.publish(marker)

    def execute_pick_and_place(self, pick_pos, place_pos):
        """Execute complete pick and place operation with visualization."""
        print("\nExecuting pick and place operation...")
        
        # Visualize pick and place targets
        self.visualize_target(pick_pos, color=[1.0, 0.0, 0.0])  # Red for pick
        self.visualize_target(place_pos, color=[0.0, 1.0, 0.0])  # Green for place
        
        # Generate and visualize waypoints
        waypoints = self.generate_trajectory(pick_pos, place_pos)
        self.visualize_trajectory(waypoints)
        
        # Move to neutral position
        print("Moving to neutral position...")
        self.arm.safe_move_to_position(self.arm.neutral_position())
        
        # Execute trajectory
        current_joints = None
        for i, pos in enumerate(waypoints):
            print(f"Moving to waypoint {i+1}/{len(waypoints)}")
            
            target = np.eye(4)
            target[:3,3] = pos
            target[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            self.show_pose(target, f"target_{i}")
            
            try:
                current_joints = self.move_to_position(pos, current_joints)
                
                # Gripper control with visualization update
                if np.array_equal(pos, pick_pos):
                    print("Closing gripper...")
                    self.arm.close_gripper()
                    # Update pick target visualization
                    self.visualize_target(pick_pos, color=[0.5, 0.5, 0.5])  # Grey out picked target
                elif np.array_equal(pos, place_pos):
                    print("Opening gripper...")
                    self.arm.open_gripper()
                    
            except Exception as e:
                print(f"Error at waypoint {i+1}: {e}")
                return False
                
        # Return to neutral position
        print("Returning to neutral position...")
        self.arm.safe_move_to_position(self.arm.neutral_position())
        return True

def main():
    """Main function to run pick and place operation."""
    rospy.init_node("robot_pick_and_place")
    
    # Define pick and place positions
    pick_pos = np.array([0.3, -0.2, 0.1])
    place_pos = np.array([0.3, 0.2, 0.1])
    
    controller = RobotPickAndPlace()
    
    try:
        success = controller.execute_pick_and_place(pick_pos, place_pos)
        if success:
            print("\nPick and place operation completed successfully!")
        else:
            print("\nPick and place operation failed!")
    except Exception as e:
        print(f"\nError during pick and place operation: {e}")

if __name__ == "__main__":
    main()
