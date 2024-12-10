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

ik = IK()  # Primary IK solver
fk = FK()
franka_ik = FrankaIK()

def filter_block_detections(all_detections, num_samples=5, detection_threshold=0.8):
    """
    Filter and median-average block detections (static or dynamic).
    """
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
    """Detect static blocks with temporal filtering."""
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
    return filtered_blocks

def use_franka_ik_as_fallback(target_pose, seed):
    """
    Attempt to solve IK using the FrankaIK class if the primary IK fails.
    target_pose: 4x4 numpy array
    seed: initial guess for joint angles (7-element array)
    """
    q7 = seed[-1]
    # A nominal QA configuration (adjust as needed)
    qa = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]

    T_array = target_pose.flatten().tolist()
    solutions = FrankaIK.compute_ik(T_array, q7, qa)
    
    # Pick the first valid solution
    for sol in solutions:
        if not np.isnan(sol).any():
            return sol, True
    return None, False

def move_to_target_pose(arm, target_pose, seed, ik_solver):
    """
    Attempt to move to target_pose using the primary IK solver.
    If that fails, try FrankaIK as a fallback.
    """
    joints, _, success, _ = ik_solver.inverse(
        target=target_pose,
        seed=seed,
        method='J_pseudo',
        alpha=0.5
    )
    if success:
        arm.safe_move_to_position(joints)
        return joints, True
    else:
        # Try FrankaIK fallback
        fallback_joints, fallback_success = use_franka_ik_as_fallback(target_pose, seed)
        if fallback_success:
            arm.safe_move_to_position(fallback_joints)
            return fallback_joints, True
    return None, False

def check_grasp_success(arm):
    """Check if the grasp was successful by checking gripper position."""
    gripper_state = arm.get_gripper_state()
    gripper_pos = gripper_state["position"]
    if isinstance(gripper_pos, (list, np.ndarray)):
        gripper_pos = np.mean(gripper_pos)
    print(f"Gripper position: {gripper_pos}")
    return gripper_pos >= 0.01

def pick_place_static(arm, blocks, place_target):
    stack_height = place_target[2]
    current_joints = arm.get_positions()  # Current joint positions as IK seed
    
    for block in blocks:
        # Pre-grasp
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
        
        # Grasp
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
            # Recovery move if grasp fails
            recovery = transform(np.array([0.50, -0.2, 0.6]), np.array([0, pi, pi]))
            recovery_joints, success = move_to_target_pose(arm, recovery, grasp_joints, ik)
            if success:
                arm.safe_move_to_position(recovery_joints)
            continue
        
        # Lift
        lift = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.15]),
            np.array([0, pi, pi])
        )
        lift_joints, success = move_to_target_pose(arm, lift, grasp_joints, ik)
        if success:
            rospy.sleep(0.2)
        
        # Place sequence
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

def main():
    try:
        team = rospy.get_param("team")
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()
    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    # Attempt to pick a dynamic block using fallback IK if needed

    
    start_position = np.array([-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position)
    
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
        target_pose = transform(np.array([0.5, 0.1725, 0.55]), np.array([0, pi, pi]))
        place_target = np.array([0.56, -0.155, 0.275])
    else:
        print("**  RED TEAM  **")
        target_pose = transform(np.array([0.485, -0.17, 0.55]), np.array([0, pi, pi]))
        place_target = np.array([0.55, 0.169, 0.27])
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")

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
        pick_place_static(arm, block_positions, place_target)
    else:
        print("No blocks detected (static)")
    
    

if __name__ == "__main__":
    main()
