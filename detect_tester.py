"""
Block Detection Tester
Author: Team 16
TODO: semi implementation of the detect_tester.py
"""

import rospy
import numpy as np
from core.interfaces import ObjectDetector
from core.interfaces import ArmController
from scipy.spatial.transform import Rotation as R
import visualization_msgs.msg  # Add this import
import tf  # Add this import
from math import pi

class DetectTester:
    """Tester class for block detection using the ObjectDetector."""

    def __init__(self):
        """Initialize the object detection tester."""
        print("\nInitializing Object Detection Tester...")
        self.detector = ObjectDetector()
        self.arm = ArmController()
        self.target_pub = rospy.Publisher('/visualization_marker', visualization_msgs.msg.Marker, queue_size=10)  # Add this line
    
    def get_detections(self):
        """Get and process the detections from the object detector."""
        try:
            detections = self.detector.get_detections()
            if not detections:
                raise ValueError("No blocks detected by the ObjectDetector.")
            processed_blocks = []
            for block_id, block_pose in detections:
                # Extract position and orientation
                position = block_pose[:3, 3]
                rotation_matrix = block_pose[:3, :3]
                orientation = R.from_matrix(rotation_matrix).as_euler('xyz')

                block_data = {
                    'id': block_id,
                    'position': position,
                    'orientation': orientation,
                    'transform': block_pose
                }
                processed_blocks.append(block_data)
            return processed_blocks
        except Exception as e:
            print(f"Error in get_detections: {e}")
            return []

    def detect_block(self):
        """
        Use camera to locate block and return its position, and orientation
        Return its position, color, and orientation

        :return: dict of block data
        """
        blocks = self.get_detections()
        if not blocks:
            return None
        
        # Return data for first detected block
        block = blocks[0]
        block_data = {
            "position": block['position'].tolist(),
            "orientation": block['orientation'].tolist()
        }

        return block_data

    def confirm_pick(self):
        """
        Confirm if gripper has successfully grasped an object
        
        :return: True if object is grasped, False otherwise
        """
        gripper_state = self.arm.get_gripper_state()
        
        # Check if gripper position indicates a grasp
        gripper_pos = gripper_state['position']
        gripper_force = gripper_state['force']
        
        # If gripper is almost closed without an object, position will be very small
        # Typical values need to be calibrated for your specific setup
        MIN_GRASP_POS = 0.01  # Minimum position indicating a grasp
        MAX_GRASP_POS = 0.04  # Maximum position indicating a grasp
        MIN_GRASP_FORCE = 5.0  # Minimum force indicating a grasp
        
        is_grasped = (MIN_GRASP_POS < gripper_pos[0] < MAX_GRASP_POS and 
                     abs(gripper_force[0]) > MIN_GRASP_FORCE)
        
        return is_grasped


def main():
    """Main function to test block detection"""
    rospy.init_node('detect_tester')
    
    tester = DetectTester()
    
    print("Starting block detection test...")
    
    try:
        # For testing with generated data
        tester.test_with_generated_data()

        # Uncomment the following lines for actual use with the robot
        # tester.display_detections()
        # blocks = tester.get_detections()
        # if blocks:
        #     tester.visualize_blocks(blocks)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    print("\nBlock detection test completed!")

if __name__ == "__main__":
    main()
