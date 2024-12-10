"""
Mock Core Manager for testing
Author: Team 16

NOTE: This is a mock implementation without ROS dependencies for testing ONLY.
DO NOT RUN DIRECTLY ON THE REAL ROBOT.

"""

import sys
import time
import numpy as np
from math import pi
from lib.calculateFK import FK
from lib.IK_position_null import IK
from lib.calcJacobian import calcJacobian
from lib.calcAngDiff import calcAngDiff

def get_time():
    """Return current time."""
    return time.time()

class MockDetector:
    """Mock detector class for testing."""
    def __init__(self):
        self.blocks = {}
    
    def get_detections(self):
        """Return list of block detections."""
        return [(block_id, block) for block_id, block in self.blocks.items()]

class Block:
    """
    Attributes:
        position (np.ndarray): 3D position of the block
        orientation (np.ndarray): Block orientation
        is_dynamic (bool): Whether block is moving
        last_seen (float): Timestamp of last detection
        tracking_history (list): Historical position data
        velocity (np.ndarray): Current velocity vector
    """
    def __init__(self):
        self.position = None
        self.orientation = None
        self.is_dynamic = False
        self.last_seen = 0
        self.tracking_history = []
        self.velocity = np.zeros(3)
    
    def update_pose(self, pose):
        """Update block pose and tracking history."""
        self.position = np.array([pose[0], pose[1], pose[2]])
        if len(pose) > 3:
            self.orientation = np.array(pose[3:])
        current_time = get_time()
        self.tracking_history.append({
            'position': self.position.copy(),
            'timestamp': current_time
        })
        if len(self.tracking_history) > 10:  # Keep last 10 positions
            self.tracking_history.pop(0)
        self.last_seen = current_time

MAX_DETECTION_RANGE = 1.5  # Maximum detection range in meters
MIN_CONFIDENCE = 0.5       # Minimum confidence for valid detection
DYNAMIC_SPEED_THRESHOLD = 0.05  # Threshold speed to consider a block dynamic
MAX_REACH = 0.8            # Maximum reach of the robot arm
ROBOT_SPEED = 0.2          # Estimated robot end effector speed in m/s
POSITION_TOLERANCE = 0.01  # Tolerance for position checks
MATCH_DURATION = 180       # Duration of the match in seconds

class MockArmController:
    """Mock arm controller for testing."""
    def __init__(self):
        self._state = {
            'position': np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4]),
            'velocity': np.zeros(7),
            'effort': np.zeros(7)
        }
        
    def get_end_effector_position(self):
        """Return the current end effector position."""
        fk = FK()
        _, positions = fk.forward(self._state['position'])
        return positions[-1]
    
    def safe_move_to_position(self, target_q):
        """Move to the target joint positions safely."""
        self._state['position'] = target_q
        return True
        
    def exec_gripper_cmd(self, **kwargs):
        """Execute gripper command."""
        return True
        
    def get_gripper_state(self):
        """Return the current gripper state."""
        return type('GripperState', (), {'position': 0.04})()