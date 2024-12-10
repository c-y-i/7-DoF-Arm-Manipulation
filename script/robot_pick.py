import sys
import numpy as np
from math import pi
import rospy
from core.interfaces import ArmController, ObjectDetector
from lib.IK_position_null import IK
from lib.calculateFK import FK
import lib.IK_velocity_null as IK_velocity_null
from core.utils import time_in_seconds, transform
from lib.franka_IK import FrankaIK
from scipy.spatial.transform import Rotation

import lib.dynamic as dynamic  # Our dynamic computations module

ik = IK()
fk = FK()
franka_ik = FrankaIK()

##################################
# Adjustable parameters
##################################
# Dynamic observation pose in Cartesian space
# Adjust so the camera looks down onto the dynamic turntable
dynamic_observation_position = np.array([0, 0.5, 0.7])
dynamic_observation_orientation = np.array([0, pi, pi])  # facing downward

# Angular velocity of the turntable (rad/s)
omega = 0.5

# Desired pick angle for dynamic interception (e.g. 0.0 = along x-axis)
theta_pick = 0.0

def filter_block_detections(all_detections, num_samples=5, detection_threshold=0.8):
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
    return filter_block_detections(all_detections, num_samples, detection_threshold)

def pick_place_static(arm, blocks, place_target):
    stack_height = place_target[2]
    current_joints = arm.get_positions()

    for block in blocks:
        pre_grasp = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.15]),
            np.array([0, pi, pi])
        )
        pre_grasp_joints, _, success, _ = ik.inverse(pre_grasp, current_joints, method='J_pseudo', alpha=0.5)
        if not success:
            print("Failed to plan pre-grasp")
            continue
        
        arm.safe_move_to_position(pre_grasp_joints)
        arm.open_gripper()
        rospy.sleep(0.2)

        grasp = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.01]),
            np.array([0, pi, pi])
        )
        grasp_joints, _, success, _ = ik.inverse(grasp, pre_grasp_joints, method='J_pseudo', alpha=0.5)
        if not success:
            print("Failed to plan grasp")
            continue

        arm.safe_move_to_position(grasp_joints)
        arm.exec_gripper_cmd(0.045, 100)
        rospy.sleep(0.5)

        gripper_position = np.linalg.norm(arm.get_gripper_state()["position"])
        print(f"Gripper position after grasp: {gripper_position}")

        if gripper_position < 0.01:
            recovery = transform(np.array([0.50, -0.2, 0.6]), np.array([0, pi, pi]))
            recovery_joints, _, success, _ = ik.inverse(recovery, grasp_joints, method='J_pseudo', alpha=0.5)
            if success:
                arm.safe_move_to_position(recovery_joints)
            continue

        lift = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.15]),
            np.array([0, pi, pi])
        )
        lift_joints, _, success, _ = ik.inverse(lift, grasp_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(lift_joints)
            rospy.sleep(0.2)

        arm.set_arm_speed(0.2)
        pre_place = transform(
            np.array([place_target[0], place_target[1], stack_height + 0.1]),
            np.array([0, pi, pi])
        )
        pre_place_joints, _, success, _ = ik.inverse(pre_place, lift_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(pre_place_joints)

        place = transform(
            np.array([place_target[0], place_target[1], stack_height - 0.025]),
            np.array([0, pi, pi])
        )
        place_joints, _, success, _ = ik.inverse(place, pre_place_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(place_joints)
            rospy.sleep(0.2)
            arm.open_gripper()
            rospy.sleep(0.3)

        retreat = transform(
            np.array([place_target[0], place_target[1], stack_height + 0.15]),
            np.array([0, pi, pi])
        )
        retreat_joints, _, success, _ = ik.inverse(retreat, place_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(retreat_joints)

        arm.set_arm_speed(0.3)
        stack_height += 0.05
        current_joints = retreat_joints

def move_arm_to_dynamic_observation(arm, ik_solver):
    """
    Move arm to a dynamic observation pose specified in Cartesian coordinates.
    """
    dynamic_observation_pose = transform(dynamic_observation_position, dynamic_observation_orientation)
    current_joints = arm.get_positions()
    obs_joints, _, success, _ = ik_solver.inverse(dynamic_observation_pose, current_joints, method='J_pseudo', alpha=0.5)
    if not success:
        print("Failed to compute IK for dynamic observation position.")
        return False
    arm.safe_move_to_position(obs_joints)
    return True

def pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=0.0, R=0.305, z=0.200):
    """
    Attempt to pick a dynamic block off the rotating turntable.
    """
    print("Starting pick_dynamic_block")

    if not move_arm_to_dynamic_observation(arm, ik):
        print("Failed to move arm to dynamic observation position.")
        return

    obs_joints = arm.get_positions()
    print(f"Observation joints: {obs_joints}")

    _, T_EW = fk.forward(obs_joints)
    H_ee_camera = detector.get_H_ee_camera()
    T_CW = T_EW @ H_ee_camera
    print(f"T_CW Matrix:\n{T_CW}")

    # Detect dynamic blocks
    num_samples = 4
    detection_threshold = 0.8
    all_detections = []
    for _ in range(num_samples):
        current_detections = []
        for (name, pose) in detector.get_detections():
            if isinstance(pose, np.ndarray) and pose.shape == (4,4):
                if "dynamic" in name.lower():
                    world_pos = T_CW @ pose
                    current_detections.append({'id': name, 'position': world_pos[:3,3]})
        all_detections.append(current_detections)
        rospy.sleep(0.1)
    print(f"All detections: {all_detections}")

    filtered_blocks = filter_block_detections(all_detections, num_samples, detection_threshold)
    print(f"Filtered blocks: {filtered_blocks}")

    if not filtered_blocks:
        print("No dynamic blocks detected.")
        return

    dynamic_block = filtered_blocks[0]
    block_position = dynamic_block['position']
    print(f"Selected dynamic block position: {block_position}")

    # Get block orientation
    block_orientation = None
    for (name, pose) in detector.get_detections():
        if name == dynamic_block['id']:
            block_orientation = pose[:3,:3]
            break
    print(f"Block orientation:\n{block_orientation}")

    if block_orientation is None:
        print("Could not retrieve orientation of dynamic block.")
        return

    t_now = time_in_seconds()
    print(f"Current time: {t_now}")

    # Determine y_offset based on the y difference between the observation pose and position
    y_offset = abs(dynamic_observation_position[1] - block_position[1])
    print(f"Determined y_offset: {y_offset}")

    if y_offset > 0.55 or y_offset < 0.3:
        print("Incorrect y offset, using pre-set value")
        y_offset = 0.45
    

    # Compute intercept
    t_intercept, intercept_pos, intercept_orientation = dynamic.compute_dynamic_intercept(
        block_position, block_orientation, t_now, omega, theta_pick, R=R, z=z
    )
    print(f"Intercept time: {t_intercept}")
    print(f"Intercept position: {intercept_pos}")
    print(f"Intercept orientation:\n{intercept_orientation}")

    roll, pitch, yaw = dynamic.orientation_to_rpy(intercept_orientation)
    print(f"Intercept orientation (rpy): roll={roll}, pitch={pitch}, yaw={yaw}")

    intercept_above = np.array([intercept_pos[0], intercept_pos[1], intercept_pos[2] + 0.10])
    final_orientation = np.array([0, pi, pi + yaw])
    pick_pose = transform(intercept_above, final_orientation)
    print(f"Pick pose:\n{pick_pose}")

    current_joints = arm.get_positions()
    # Initialize alpha values to try
    alpha_values = [0.6, 0.5, 0.25, 0.1, 0.05]
    success = False

    # First segment: move above pick position, keep current orientation
    intercept_above = np.array([intercept_pos[0], intercept_pos[1] + y_offset, intercept_pos[2] + 0.10])

    # Use the current end-effector orientation
    _, T_EE_current = fk.forward(current_joints)
    current_orientation_matrix = T_EE_current[:3, :3]
    current_orientation = Rotation.from_matrix(current_orientation_matrix).as_euler('xyz')
    pick_pose_above = transform(intercept_above, current_orientation)
    print(f"Pick pose above (keeping current orientation):\n{pick_pose_above}")

    for alpha in alpha_values:
        print(f"Trying IK with alpha={alpha} for pick_pose_above")
        pick_joints_above, _, success, _ = ik.inverse(pick_pose_above, current_joints, method='J_pseudo', alpha=alpha)
        if success:
            print("IK succeeded for pick_pose_above")
            break
        else:
            print(f"IK failed for alpha={alpha}")
    if not success:
        print("All IK attempts failed for pick_pose_above.")
        # Proceed to try IK_velocity_null or other methods if desired
        return

    arm.safe_move_to_position(pick_joints_above)
    print("Moved to position above pick point.")
    grasp_pose_offset = np.array([intercept_pos[0], intercept_pos[1] + y_offset, intercept_pos[2]])

    # Second segment: change orientation while moving down to pick position
    pick_pose = transform(grasp_pose_offset + np.array([0, 0, 0.05]), final_orientation)
    print(f"Pick pose (changing orientation):\n{pick_pose}")

    for alpha in alpha_values:
        print(f"Trying IK with alpha={alpha} for pick_pose")
        pick_joints, _, success, _ = ik.inverse(pick_pose, pick_joints_above, method='J_pseudo', alpha=alpha)
        if success:
            print("IK succeeded for pick_pose")
            break
        else:
            print(f"IK failed for alpha={alpha}")
    if not success:
        print("All IK attempts failed for pick_pose.")
        # Try IK_velocity_null as the next fallback
        print("Attempting IK_velocity_null as fallback.")
        # Compute desired displacement
        _, T_EE_current = fk.forward(pick_joints_above)
        position_displacement, rotation_axis = IK.displacement_and_axis(pick_pose, T_EE_current)
        v_in = position_displacement
        omega_in = rotation_axis
        b = IK.joint_centering_task(pick_joints_above)
        # Compute joint velocities using IK_velocity_null
        dq = IK_velocity_null(pick_joints_above, v_in, omega_in, b)
        pick_joints = pick_joints_above + dq.flatten()
        # Check if the new joints are valid
        success_ikv, _ = ik.is_valid_solution(pick_joints, pick_pose)
        if success_ikv:
            print("IK_velocity_null succeeded.")
        else:
            print("IK_velocity_null failed, trying Franka IK as last fallback.")
            # Use Franka IK as the final fallback
            q7 = pick_joints_above[-1]
            qa = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
            T_array = pick_pose.flatten().tolist()
            fallback_solutions = franka_ik.compute_ik(T_array, q7, qa)
            fallback_joints = None
            for sol in fallback_solutions:
                if not np.isnan(sol).any():
                    fallback_joints = sol
                    break
            if fallback_joints is None:
                print("Failed to plan pick pose with Franka IK.")
                return
            else:
                print(f"Franka IK solution found: {fallback_joints}")
                pick_joints = fallback_joints

    arm.safe_move_to_position(pick_joints)
    print("Moved to pick position.")

    rospy.sleep(t_intercept)
    print(f"Slept for intercept time: {t_intercept} seconds")

    grasp_pose = transform(intercept_pos + np.array([0,0,0.01]), final_orientation)
    print(f"Grasp pose:\n{grasp_pose}")

    
    grasp_joints, _, success, _ = ik.inverse(grasp_pose_offset, pick_joints, method='J_pseudo', alpha=0.5)
    print(f"IK success: {success}, Grasp joints: {grasp_joints}")

    for alpha in alpha_values:
        print(f"Trying IK with alpha={alpha} for grasp_pose")
        pick_joints_above, _, success, _ = ik.inverse(grasp_pose_offset, pick_joints, method='J_pseudo', alpha=alpha)
        if success:
            print("IK succeeded for grasp_pose")
            break
        else:
            print(f"IK failed for alpha={alpha}")
        
    if not success:
        print("Failed IK for final grasp descent.")
        return

    arm.safe_move_to_position(grasp_joints)
    print("Moved to grasp position.")

    arm.exec_gripper_cmd(0.045, 100)
    rospy.sleep(0.5)
    print("Gripper command executed.")

    gripper_state = arm.get_gripper_state()
    gripper_position = np.linalg.norm(gripper_state["position"])
    print(f"Gripper position after grasp: {gripper_position}")

    if gripper_position < 0.01:
        print("Failed to pick dynamic block.")
        return

    # Lift
    lift_pose = transform(intercept_above, final_orientation)
    lift_joints, _, success, _ = ik.inverse(lift_pose, grasp_joints, method='J_pseudo', alpha=0.5)
    print(f"Lifting IK success: {success}, Lift joints: {lift_joints}")

    if success:
        arm.safe_move_to_position(lift_joints)
        rospy.sleep(0.2)
        print("Lifted the block.")

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
    
    start_position = np.array([-0.01779206, -0.76012354, 0.01978261, -2.34205014, 
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

    # Now attempt dynamic block picking
    pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=theta_pick)
    
    # Compute IK for observation pose (static)
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

    # Detect and place static blocks
    static_blocks = detect_static(detector, T_CW)
    if static_blocks:
        pick_place_static(arm, static_blocks, place_target)
    else:
        print("No static blocks detected")

if __name__ == "__main__":
    main()
