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
import json

class Detection:
    def __init__(self):
        """Initialize the pick and place controller."""
        self.arm = ArmController()
        self.detector = ObjectDetector()
        self.blocks = []

    def scan_blocks(self, delay, detected_blocks):
        block_data = {}
        blocks = []
        for block_id, block_pose in detected_blocks:
            block_data.setdefault(block_id, [])
            block_data[block_id] = block_pose
            rospy.sleep(delay)  # Wait for stability
        return block_data