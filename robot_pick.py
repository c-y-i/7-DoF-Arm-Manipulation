import sys
import numpy as np
from copy import deepcopy
from math import pi
import rospy

# Robot interfaces
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.IK_position_null import IK
from lib.calculateFK import FK

# Utilities
from core.utils import time_in_seconds, transform

import lib.static as static
import lib.dynamic as dynamic

ik = IK()
fk = FK()

# Now attempt to pick a dynamic block
omega = 0.5  # Example angular velocity of turntable in rad/s
theta_pick = 0.0  # Pick angle along x-axis

def detect_static(detector, T_CW, num_samples=5, detection_threshold=0.8):
    """
    Detect static blocks and filter their positions.
    """
    all_detections = []
    for _ in range(num_samples):
        current_detections = []
        for block, position in detector.get_detections():
            if isinstance(position, np.ndarray) and position.shape == (4,4):
                world_pos = T_CW @ position
                current_detections.append({
                    'id': block,
                    'pose': world_pos,
                    'position': world_pos[:3, 3]
                })
        all_detections.append(current_detections)
        rospy.sleep(0.1)
    return static.filter_block_detections(all_detections, num_samples, detection_threshold)

def pick_place_static(arm, blocks, place_target):
    current_joints = arm.get_positions()
    stack_height = place_target[2]

    for block in blocks:
        pose_params = static.get_pick_place_pose_params(block['position'], stack_height, place_target)

        pre_grasp = transform(*pose_params['pre_grasp'])
        grasp = transform(*pose_params['grasp'])
        lift_pose = transform(*pose_params['lift'])
        pre_place = transform(*pose_params['pre_place'])
        place = transform(*pose_params['place'])
        retreat = transform(*pose_params['retreat'])

        # Pre-grasp
        pre_grasp_joints, _, success, _ = ik.inverse(pre_grasp, current_joints, method='J_pseudo', alpha=0.5)
        if not success:
            print("Failed to plan pre-grasp for static block")
            continue
        arm.safe_move_to_position(pre_grasp_joints)
        arm.open_gripper()
        rospy.sleep(0.2)

        # Grasp
        grasp_joints, _, success, _ = ik.inverse(grasp, pre_grasp_joints, method='J_pseudo', alpha=0.5)
        if not success:
            print("Failed to plan grasp for static block")
            continue
        arm.safe_move_to_position(grasp_joints)
        arm.exec_gripper_cmd(0.045, 100)
        rospy.sleep(0.5)

        gripper_position = np.linalg.norm(arm.get_gripper_state()["position"])
        print(f"Static block: Gripper position after grasp: {gripper_position}")

        if gripper_position < 0.01:
            # Failed to pick block, move to a safe position
            recovery = transform(np.array([0.50, -0.2, 0.6]), np.array([0, pi, pi]))
            recovery_joints, _, success, _ = ik.inverse(recovery, grasp_joints, method='J_pseudo', alpha=0.5)
            if success:
                arm.safe_move_to_position(recovery_joints)
            continue

        # Lift
        lift_joints, _, success, _ = ik.inverse(lift_pose, grasp_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(lift_joints)
            rospy.sleep(0.2)

        arm.set_arm_speed(0.2)
        pre_place_joints, _, success, _ = ik.inverse(pre_place, lift_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(pre_place_joints)

        place_joints, _, success, _ = ik.inverse(place, pre_place_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(place_joints)
            rospy.sleep(0.2)
            arm.open_gripper()
            rospy.sleep(0.3)

        retreat_joints, _, success, _ = ik.inverse(retreat, place_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(retreat_joints)

        arm.set_arm_speed(0.3)
        stack_height += 0.05
        current_joints = retreat_joints


def pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=0.0, R=0.305, z=0.200):
    """
    Function to pick a dynamic block off the rotating turntable.
    """
    # Move arm to observation position to see dynamic blocks
    observation_joints = np.array([-0.01779206, -0.76012354, 0.01978261, -2.34205014, 
                                   0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(observation_joints)

    H_ee_camera = detector.get_H_ee_camera()
    _, T_EW = fk.forward(observation_joints)
    T_CW = T_EW @ H_ee_camera

    # Detect dynamic blocks
    all_detections = []
    num_samples = 5
    detection_threshold = 0.8

    for _ in range(num_samples):
        current_detections = []
        for (name, pose) in detector.get_detections():
            if isinstance(pose, np.ndarray) and pose.shape == (4,4):
                if "dynamic" in name.lower():  # Identify dynamic blocks by name
                    world_pos = T_CW @ pose
                    current_detections.append({'id': name, 'position': world_pos[:3,3]})
        all_detections.append(current_detections)
        rospy.sleep(0.1)

    filtered_blocks = static.filter_block_detections(all_detections, num_samples, detection_threshold)
    if not filtered_blocks:
        print("No dynamic blocks detected.")
        return

    dynamic_block = filtered_blocks[0]
    block_position = dynamic_block['position']

    # Get block orientation (assuming we can get it from detector)
    block_orientation = None
    for (name, pose) in detector.get_detections():
        if name == dynamic_block['id']:
            block_orientation = pose[:3,:3]
            break

    if block_orientation is None:
        print("Could not retrieve orientation of dynamic block.")
        return

    t_now = time_in_seconds()

    # Compute intercept
    t_intercept, intercept_pos, intercept_orientation = dynamic.compute_dynamic_intercept(
        block_position, block_orientation, t_now, omega, theta_pick, R=R, z=z
    )

    # Extract roll, pitch, yaw from intercept orientation
    roll, pitch, yaw = dynamic.orientation_to_rpy(intercept_orientation)

    # Approach from above
    intercept_above = np.array([intercept_pos[0], intercept_pos[1], intercept_pos[2] + 0.10])
    final_orientation = np.array([0, pi, pi + yaw])

    pick_pose = transform(intercept_above, final_orientation)

    current_joints = arm.get_positions()
    pick_joints, _, success, _ = ik.inverse(pick_pose, current_joints, method='J_pseudo', alpha=0.5)
    if not success:
        print("Failed to plan intercept pose.")
        return

    arm.safe_move_to_position(pick_joints)

    # Wait until block arrives (simple approach)
    rospy.sleep(t_intercept)

    # Descend to grasp
    grasp_pose = transform(intercept_pos + np.array([0,0,0.01]), final_orientation)
    grasp_joints, _, success, _ = ik.inverse(grasp_pose, pick_joints, method='J_pseudo', alpha=0.5)
    if not success:
        print("Failed IK for final grasp descent.")
        return
    arm.safe_move_to_position(grasp_joints)
    arm.exec_gripper_cmd(0.045, 100)
    rospy.sleep(0.5)

    gripper_position = np.linalg.norm(arm.get_gripper_state()["position"])
    print(f"Dynamic block: Gripper position after grasp: {gripper_position}")
    if gripper_position < 0.01:
        print("Failed to pick dynamic block.")
        return

    # Lift the block
    lift_pose = transform(intercept_above, final_orientation)
    lift_joints, _, success, _ = ik.inverse(lift_pose, grasp_joints, method='J_pseudo', alpha=0.5)
    if success:
        arm.safe_move_to_position(lift_joints)
        rospy.sleep(0.2)

    print("Dynamic block picked successfully!")


def main():
    try:
        team = rospy.get_param("team")
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 
                               0.02984053, 1.54119353+pi/2, 0.75344866])
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

    pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=theta_pick)

    # IK for observation pose
    observation_joints, _, success, _ = ik.inverse(
        target=target_pose,
        seed=start_position,
        method='J_pseudo',
        alpha=0.5
    )
    if not success:
        print("Failed to compute IK for observation pose")
        return

    arm.set_arm_speed(0.3)
    arm.safe_move_to_position(observation_joints)

    H_ee_camera = detector.get_H_ee_camera()
    joints_array = np.array(observation_joints)
    _, T_EW = fk.forward(joints_array)
    T_CW = T_EW @ H_ee_camera

    # Detect and handle static blocks
    block_positions = detect_static(detector, T_CW)
    if block_positions:
        pick_place_static(arm, block_positions, place_target)
    else:
        print("No static blocks detected")

    

if __name__ == "__main__":
    main()
