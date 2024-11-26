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


class DetectTester:
    def __init__(self):
        """Initialize the object detection tester."""
        print("\nInitializing Object Detection Tester...")
        self.detector = ObjectDetector()
        self.arm = ArmController()
    
    def test_detect(self):
        """Test the object detection algorithm."""
        print("\nTesting object detection algorithm...")
        
        # Get blocks detected by camera
        detections = self.detector.get_detections()
        
        # Process each detection
        processed_blocks = []
        for block_name, block_pose in detections:
            # Extract position from homogeneous transform
            position = block_pose[:3, 3]
            
            # Extract rotation matrix and convert to Euler angles
            rotation = block_pose[:3, :3]
            euler = R.from_matrix(rotation).as_euler('xyz')
            
            block_data = {
                'name': block_name,
                'position': position,
                'orientation': euler,
                'transform': block_pose
            }
            processed_blocks.append(block_data)
            
            # Print detection info
            print(f"\nDetected block: {block_name}")
            print(f"Position: {position}")
            print(f"Orientation (xyz): {euler}")
        
        return processed_blocks

    def detect_block(self):
        """
        Use camera to locate block and return its position, and orientation
        Return its position, color, and orientation

        :return: dict of block data
        """
        blocks = self.test_detect()
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
        # Test block detection
        blocks = tester.test_detect()
        
        if not blocks:
            print("No blocks detected!")
        else:
            print(f"\nSuccessfully detected {len(blocks)} blocks!")
            
        # Test grasp confirmation
        print("\nTesting grasp confirmation...")
        is_grasped = tester.confirm_pick()
        print(f"Grasp detected: {is_grasped}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    print("\nBlock detection test completed!")

if __name__ == "__main__":
    main()
