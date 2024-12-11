"""
MEAM 5200 Final Project: Pick and Place
Author: Team 16
Description: Main excution script for the final project competition.
This script will perform the following tasks:
1. Move the robot to a starting observation pose.
2. Detect static blocks and pick/place them on the target stack.
3. Detect dynamic blocks and pick/place them on the target stack.
Version: 1.0.0
"""

import sys
import numpy as np
from copy import deepcopy
from math import pi, sin, cos, atan2
import rospy
import time  # Add this import
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
start_time = None

def load_config(team):
    print(f"Loading configuration for team: {team}")
    with open("config.json", "r") as f:
        data = json.load(f)
    return data["team_data"][team]

def filter_block_detections(all_detections, num_samples=5, detection_threshold=0.8):
    print("Filtering block detections")
    valid_blocks = {}
    for sample in all_detections:
        for block in sample:
            block_id = block['id']
            if block_id not in valid_blocks:
                valid_blocks[block_id] = []
            valid_blocks[block_id].append(block['position'])
    filtered_blocks = []
    for block_id, positions in valid_blocks.items():
        if len(positions) >= num_samples * detection_threshold:
            positions = np.array(positions)
            median_pos = np.median(positions, axis=0)
            filtered_blocks.append({
                'id': block_id,
                'position': median_pos,
                'confidence': len(positions) / num_samples
            })
    return sorted(filtered_blocks, key=lambda x: x['confidence'], reverse=True)

def detect_static(detector, T_CW, num_samples=5, detection_threshold=0.8):
    print("Detecting static blocks")
    all_detections = []
    for _ in range(num_samples):
        current_detections = []
        for block, position in detector.get_detections():
            if isinstance(position, np.ndarray) and position.shape == (4, 4):
                world_pos = T_CW @ position
                current_detections.append({
                    'id': block,
                    'pose': world_pos,
                    'position': world_pos[:3, 3]
                })
        all_detections.append(current_detections)
        rospy.sleep(0.1)
    filtered_blocks = filter_block_detections(all_detections, num_samples, detection_threshold)
    print(f"Found {len(filtered_blocks)} static blocks")
    return filtered_blocks

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
    alphas = [0.5, 0.2, 0.1, 0.05, 0.025]
    for alpha in alphas:
        print(f"Attempting IK with alpha = {alpha}")
        joints, _, success, _ = ik_solver.inverse(
            target=target_pose,
            seed=seed,
            method='J_pseudo',
            alpha=alpha
        )
        print(f"IK success with alpha={alpha}: {success}")
        if success:
            arm.safe_move_to_position(joints)
            print(f"Moved to joints: {joints}")
            return joints, True
    print("IK failed with all alpha values, trying fallback IK")
    fallback_joints, fallback_success = use_franka_ik_as_fallback(target_pose, seed)
    if fallback_success:
        arm.safe_move_to_position(fallback_joints)
        print(f"Moved to fallback joints: {fallback_joints}")
        return fallback_joints, True
    print("Failed to move to target pose")
    return None, False

def check_grasp_success(arm):
    gripper_state = arm.get_gripper_state()
    gripper_pos = gripper_state["position"]
    print(f"Gripper state positions: {gripper_pos}")
    if isinstance(gripper_pos, (list, np.ndarray)):
        gripper_pos = np.mean(gripper_pos)
    print(f"Average gripper position: {gripper_pos}")
    return gripper_pos >= 0.01

def pick_place_static(arm, blocks, place_target, data):
    print("Starting pick and place for static blocks")
    stack_height = place_target[2]
    current_joints = arm.get_positions()
    for block in blocks:
        block_start_time = time.time()
        print(f"Picking block ID: {block['id']}")
        pre_grasp = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.15]),
            np.array([0, pi, pi])
        )
        pre_grasp_joints, success = move_to_target_pose(arm, pre_grasp, current_joints, ik)
        if not success:
            print("Failed to plan pre-grasp")
            continue
        arm.open_gripper()
        rospy.sleep(0.2)
        grasp = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.01]),
            np.array([0, pi, pi])
        )
        grasp_joints, success = move_to_target_pose(arm, grasp, pre_grasp_joints, ik)
        if not success:
            print("Failed to plan grasp")
            continue
        arm.exec_gripper_cmd(0.045, 100)
        rospy.sleep(0.5)
        gripper_position = np.linalg.norm(arm.get_gripper_state()["position"])
        print(f"Gripper position after grasp: {gripper_position}")
        if gripper_position < 0.01:
            print("Grasp failed, attempting recovery move")
            recovery = transform(np.array([0.50, -0.2, 0.6]), np.array([0, pi, pi]))
            recovery_joints, success = move_to_target_pose(arm, recovery, grasp_joints, ik)
            if success:
                arm.safe_move_to_position(recovery_joints)
            continue
        lift = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.15]),
            np.array([0, pi, pi])
        )
        lift_joints, success = move_to_target_pose(arm, lift, grasp_joints, ik)
        if success:
            rospy.sleep(0.2)
        arm.set_arm_speed(0.2)
        pre_place = transform(
            np.array([place_target[0], place_target[1], stack_height + 0.1]),
            np.array([0, pi, pi])
        )
        pre_place_joints, success = move_to_target_pose(arm, pre_place, lift_joints, ik)
        place = transform(
            np.array([place_target[0], place_target[1], stack_height - 0.025]),
            np.array([0, pi, pi])
        )
        place_joints, success = move_to_target_pose(arm, place, pre_place_joints, ik)
        if success:
            rospy.sleep(0.2)
            arm.open_gripper()
            rospy.sleep(0.3)
            print("Block placed successfully")
            block_duration = time.time() - block_start_time
            total_duration = time.time() - start_time
            print(f"Block placement took: {block_duration:.2f} seconds")
            print(f"Total elapsed time: {total_duration:.2f} seconds")
        else:
            print("Failed to place block")
        retreat = transform(
            np.array([place_target[0], place_target[1], stack_height + 0.15]),
            np.array([0, pi, pi])
        )
        retreat_joints, success = move_to_target_pose(arm, retreat, place_joints, ik)
        if success:
            arm.safe_move_to_position(retreat_joints)
        arm.set_arm_speed(0.3)
        stack_height += 0.05
        current_joints = retreat_joints

def place_dynamic_block(arm, data):
    global dynamic_stack_height_1, dynamic_stack_height_2, dynamic_blocks_collected
    block_start_time = time.time()
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
        np.array([dynamicPlaceTarget[0], dynamicPlaceTarget[1], stack_height - 0.03]),
        np.array([0, pi, pi])
    )
    place_joints, success = move_to_target_pose(arm, place, pre_place_joints, ik)
    if success:
        rospy.sleep(0.2)
        arm.open_gripper()
        rospy.sleep(0.3)
        print("Dynamic block placed successfully")
        block_duration = time.time() - block_start_time
        total_duration = time.time() - start_time
        print(f"Block placement took: {block_duration:.2f} seconds")
        print(f"Total elapsed time: {total_duration:.2f} seconds")
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
    detector = ObjectDetector()
    print("Loading configuration data")
    data = load_config(team)
    start_position = np.array([-0.01779206, -0.76012354, 0.01978261,
                               -2.34205014, 0.02984053, 1.54119353+pi/2,
                               0.75344866])
    arm.safe_move_to_position(start_position)
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
        target_pose = transform(np.array([0.5, 0.1725, 0.55]), np.array([0, pi, pi]))
        place_target = np.array([1, -0.155, 0.275])
    else:
        print("**  RED TEAM  **")
        target_pose = transform(np.array([0.485, -0.17, 0.55]), np.array([0, pi, pi]))
        place_target = np.array([1, 0.155, 0.27])
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")
    global dynamic_blocks_collected, start_time
    start_time = time.time()
    max_dynamic_blocks = 8
    print("Beginning static block sequence")
    observation_joints, success = move_to_target_pose(arm, target_pose, start_position, ik)
    if not success:
        print("Failed to compute IK for observation pose")
        return
    arm.set_arm_speed(0.3)
    H_ee_camera = detector.get_H_ee_camera()
    joints_array = np.array(observation_joints)
    _, T_EW = fk.forward(joints_array)
    T_CW = T_EW @ H_ee_camera
    block_positions = detect_static(detector, T_CW)
    if block_positions:
        pick_place_static(arm, block_positions, place_target, data)
    else:
        print("No static blocks detected")

    print("Beginning dynamic block sequence")
    for i in range(max_dynamic_blocks):
        print(f"Collecting dynamic block #{i + 1}")
        retrieve_dynamic_block(arm, data)
        if dynamic_blocks_collected >= max_dynamic_blocks:
            print("Collected maximum number of dynamic blocks")
            break

if __name__ == "__main__":
    main()
