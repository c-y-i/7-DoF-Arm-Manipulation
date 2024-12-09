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

# Import your FrankaIK class
from lib.franka_IK import FrankaIK
from lib.IK_velocity_null import IK_velocity_null  # Import IK_velocity_null

ik = IK()  # Primary IK solver
fk = FK()
franka_ik = FrankaIK()

##################################
# Adjustable parameters
##################################
dynamic_observation_position = np.array([0, 0.5, 0.7])
dynamic_observation_orientation = np.array([0, pi, pi])  # facing downward
omega = 0.5
theta_pick = 0.0

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

def compute_dynamic_intercept(block_position, block_orientation, t_now, omega, theta_pick, R=0.305, z=0.200):
    x, y, _ = block_position
    theta_now = atan2(abs(x), abs(y))
    dtheta = abs(theta_pick - abs(theta_now))
    t_intercept = dtheta / omega
    y_intercept = R * cos(theta_pick)
    x_intercept = R * sin(theta_pick)
    z_intercept = z
    R_z = np.array([
        [cos(dtheta), -sin(dtheta), 0],
        [sin(dtheta),  cos(dtheta), 0],
        [0,            0,           1]
    ])
    intercept_orientation = R_z @ block_orientation
    intercept_pos = np.array([x_intercept, y_intercept, z_intercept])
    return t_intercept, intercept_pos, intercept_orientation

def orientation_to_rpy(R):
    yaw = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = np.arctan2(R[2,1], R[2,2])
    return roll, pitch, yaw

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

    # Compute intercept
    t_intercept, intercept_pos, intercept_orientation = compute_dynamic_intercept(
        block_position, block_orientation, t_now, omega, theta_pick, R=R, z=z
    )
    print(f"Intercept time: {t_intercept}")
    print(f"Intercept position: {intercept_pos}")
    print(f"Intercept orientation:\n{intercept_orientation}")

    roll, pitch, yaw = orientation_to_rpy(intercept_orientation)
    print(f"Intercept orientation (rpy): roll={roll}, pitch={pitch}, yaw={yaw}")

    intercept_above = np.array([intercept_pos[0], intercept_pos[1], intercept_pos[2] + 0.10])
    final_orientation = np.array([0, pi, pi + yaw])
    pick_pose = transform(intercept_above, final_orientation)
    print(f"Pick pose:\n{pick_pose}")

    current_joints = arm.get_positions()
    # Initialize alpha values to try
    alpha_values = [0.5, 0.1, 0.05]
    success = False

    # First segment: move above pick position, keep current orientation
    intercept_above = np.array([intercept_pos[0], intercept_pos[1], intercept_pos[2] + 0.10])

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

    # Second segment: change orientation while moving down to pick position
    pick_pose = transform(intercept_pos + np.array([0, 0, 0.01]), final_orientation)
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

    grasp_joints, _, success, _ = ik.inverse(grasp_pose, pick_joints, method='J_pseudo', alpha=0.5)
    print(f"IK success: {success}, Grasp joints: {grasp_joints}")

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
    # Attempt to pick a dynamic block using fallback IK if needed
    pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=theta_pick)

    
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
    
    # Detect and pick static blocks
    block_positions = detect_static(detector, T_CW)
    if block_positions:
        pick_place_static(arm, block_positions, place_target)
    else:
        print("No blocks detected (static)")
    
    

if __name__ == "__main__":
    main()
