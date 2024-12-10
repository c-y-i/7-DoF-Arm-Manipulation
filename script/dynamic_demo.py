import sys
import numpy as np
from copy import deepcopy
from math import pi, sin, cos, atan2
import rospy
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.IK_position_null import IK
from lib.calculateFK import FK
from time import perf_counter
from scipy.spatial.transform import Rotation
from core.utils import time_in_seconds
from core.utils import trans, roll, pitch, yaw, transform
from lib.franka_IK import FrankaIK
import json

ik = IK()
fk = FK()
franka_ik = FrankaIK()

dynamic_stack_height_1 = 0.0
dynamic_stack_height_2 = 0.0
dynamic_blocks_collected = 0

def load_config(team):
    print(f"Loading configuration for team: {team}")
    with open("config.json", "r") as f:
        data = json.load(f)
    return data["team_data"][team]

def use_franka_ik_as_fallback(target_pose, seed):
    print("Attempting Franka IK as fallback")
    q7 = seed[-1]
    qa = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
    T_array = target_pose.flatten().tolist()
    solutions = FrankaIK.compute_ik(T_array, q7, qa)
    for sol in solutions:
        if not np.isnan(sol).any():
            print("Franka IK found a solution")
            return sol, True
    print("Franka IK failed to find a solution")
    return None, False

def move_to_target_pose(arm, target_pose, seed, ik_solver):
    print(f"Moving to target pose with seed: {seed}")
    joints, _, success, _ = ik_solver.inverse(
        target=target_pose,
        seed=seed,
        method='J_pseudo',
        alpha=0.5
    )
    print(f"IK success: {success}")
    if success:
        arm.safe_move_to_position(joints)
        print(f"Moved to joints: {joints}")
        return joints, True
    else:
        print("IK failed, trying fallback IK")
        fallback_joints, fallback_success = use_franka_ik_as_fallback(target_pose, seed)
        if fallback_success:
            arm.safe_move_to_position(fallback_joints)
            print(f"Moved to fallback joints: {fallback_joints}")
            return fallback_joints, True
    print("Failed to move to target pose")
    return None, False

def place_dynamic_block(arm, data):
    global dynamic_stack_height_1, dynamic_stack_height_2, dynamic_blocks_collected
    print("Placing dynamic block")
    if dynamic_blocks_collected <= 4:
        dynamicPlaceTarget = np.array(data["dynamicPlaceTarget"])
        stack_height = dynamicPlaceTarget[2] + dynamic_stack_height_1
    else:
        dynamicPlaceTarget = np.array(data["dynamicPlaceTarget2"])
        stack_height = dynamicPlaceTarget[2] + dynamic_stack_height_2
    current_joints = arm.get_positions()
    pre_place = transform(
        np.array([dynamicPlaceTarget[0], dynamicPlaceTarget[1], stack_height + 0.1]),
        np.array([0, pi, pi])
    )
    pre_place_joints, success = move_to_target_pose(arm, pre_place, current_joints, ik)
    if not success:
        print("Failed to plan dynamic pre-place")
        return
    place = transform(
        np.array([dynamicPlaceTarget[0], dynamicPlaceTarget[1], stack_height - 0.025]),
        np.array([0, pi, pi])
    )
    place_joints, success = move_to_target_pose(arm, place, pre_place_joints, ik)
    if success:
        rospy.sleep(0.2)
        arm.open_gripper()
        rospy.sleep(0.3)
        print("Dynamic block placed successfully")
    else:
        print("Failed to place dynamic block")
    retreat = transform(
        np.array([dynamicPlaceTarget[0], dynamicPlaceTarget[1], stack_height + 0.15]),
        np.array([0, pi, pi])
    )
    retreat_joints, success = move_to_target_pose(arm, retreat, place_joints, ik)
    if success:
        arm.safe_move_to_position(retreat_joints)
    if dynamic_blocks_collected <= 4:
        dynamic_stack_height_1 += 0.05
    else:
        dynamic_stack_height_2 += 0.05

def retrieve_dynamic_block(arm, data):
    print("Retrieving dynamic block")
    global dynamic_blocks_collected
    a_int_1 = np.array(data["acquireIntermediatePose1"])
    a_int_2 = np.array(data["acquireIntermediatePose2"])
    a_pre = np.array(data["acquirePrePose"])
    a_pose = np.array(data["acquirePose"])
    gripperClosePos = 0.04
    gripperForce = 100
    arm.safe_move_to_position(a_int_2)
    arm.safe_move_to_position(a_pre)
    arm.safe_move_to_position(a_pose)
    block_grasped = False
    while not block_grasped:
        arm.exec_gripper_cmd(gripperClosePos, gripperForce)
        rospy.sleep(5.0)
        gr_state = arm.get_gripper_state()
        if (gr_state['position'][0] + gr_state['position'][1]) > 1.2 * gripperClosePos:
            block_grasped = True
            print("Dynamic block grasped successfully")
        else:
            print("Failed to grasp dynamic block, retrying")
            arm.open_gripper()
            arm.safe_move_to_position(a_pre)
            arm.safe_move_to_position(a_pose)
    arm.safe_move_to_position(a_pre)
    arm.safe_move_to_position(a_int_2)
    arm.safe_move_to_position(a_int_1)
    place_dynamic_block(arm, data)
    dynamic_blocks_collected += 1

def main():
    try:
        team = rospy.get_param("team")
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()
    print("Initializing ROS node")
    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector() # no need for dynamic block, we going in raw
    print("Loading configuration data")
    data = load_config(team)
    start_position = np.array([-0.01779206, -0.76012354, 0.01978261,
                               -2.34205014, 0.02984053, 1.54119353+pi/2,
                               0.75344866])
    arm.safe_move_to_position(start_position)
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")
    global dynamic_blocks_collected
    max_dynamic_blocks = 8
    for i in range(max_dynamic_blocks):
        retrieve_dynamic_block(arm, data)
        if dynamic_blocks_collected >= max_dynamic_blocks:
            print("Collected maximum number of dynamic blocks")
            break

if __name__ == "__main__":
    main()
