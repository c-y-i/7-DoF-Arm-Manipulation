
"""
Pick and Place Demo with Detection
Author: Team 16
"""

import rospy
import numpy as np
from math import pi
from detect_tester import DetectTester
from robot_pick import RobotPickAndPlace
from core.utils import transform

class PickPlaceDemo:
    def __init__(self):
        """Initialize the demo with detection and robot control."""
        print("\nInitializing Pick and Place Demo...")
        self.detector = DetectTester()
        self.robot = RobotPickAndPlace()
        
    def run_demo(self):
        """Execute the pick and place demo with detection."""
        print("\nStarting Pick and Place Demo...")
        
        # 1. Detect blocks
        print("\nDetecting blocks...")
        blocks = self.detector.test_detect()
        
        if not blocks:
            print("No blocks detected! Aborting demo.")
            return False
            
        # 2. Process detected blocks
        for block in blocks:
            try:
                # Extract pick position from detection
                pick_pos = block['position']
                
                # Define place position (offset from pick position)
                # Adjust these values based on your setup
                place_pos = np.array([
                    pick_pos[0] + 0.2,  # 20cm offset in x
                    pick_pos[1],        # same y
                    pick_pos[0]         # same z
                ])
                
                print(f"\nAttempting to pick block at position: {pick_pos}")
                print(f"Planning to place at position: {place_pos}")
                
                # Execute pick and place
                success = self.robot.execute_pick_and_place(pick_pos, place_pos)
                
                if success:
                    print(f"Successfully picked and placed block!")
                    
                    # Verify grasp
                    if self.detector.confirm_pick():
                        print("Grasp confirmed by gripper sensors")
                    else:
                        print("Warning: Grasp may have failed according to gripper sensors")
                else:
                    print("Failed to complete pick and place operation")
                    
            except Exception as e:
                print(f"Error during pick and place: {str(e)}")
                continue
                
        return True

def main():
    """Main function to run the demo"""
    rospy.init_node("pick_place_demo")
    
    try:
        demo = PickPlaceDemo()
        success = demo.run_demo()
        
        if success:
            print("\nDemo completed successfully!")
        else:
            print("\nDemo failed!")
            
    except Exception as e:
        print(f"\nError during demo execution: {str(e)}")
        
    print("\nDemo finished.")

if __name__ == "__main__":
    main()