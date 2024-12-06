import sys
import numpy as np
from copy import deepcopy
from math import pi, sin, cos, atan2
import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.IK_position_null import IK
from lib.calculateFK import FK

from time import perf_counter
from scipy.spatial.transform import Rotation

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from core.utils import trans, roll, pitch, yaw, transform

ik = IK() # IK solver
fk = FK()

# Set parameters for dynamic block interception
omega = 0.5  # example angular velocity of turntable in rad/s
theta_pick = 0.0  # pick angle along x-axis

def filter_block_detections(all_detections, num_samples=5, detection_threshold=0.8):
    """
    Filter and median-average block detections (used for static and dynamic blocks).
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
    """Block detection (static) with temporal filtering and outlier rejection."""
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

def move_to_target_pose(arm, target_pose, seed, ik_solver):
    """Helper function to move to a target pose using IK."""
    joints, _, success, _ = ik_solver.inverse(
        target=target_pose,
        seed=seed,
        method='J_pseudo',
        alpha=0.5
    )
    if success:
        arm.safe_move_to_position(joints)
        return joints, True
    return None, False

def check_grasp_success(arm):
    """Check if grasp was successful by analyzing gripper state."""
    gripper_state = arm.get_gripper_state()
    gripper_pos = gripper_state["position"]
    if isinstance(gripper_pos, (list, np.ndarray)):
        gripper_pos = np.mean(gripper_pos)
    print(f"Gripper position: {gripper_pos}")
    return gripper_pos >= 0.01

def pick_place_static(arm, blocks, place_target):
    stack_height = place_target[2]
    current_joints = arm.get_positions()  # Get current joint positions for IK seed
    
    for block in blocks:
        # Pre-grasp
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
        
        # Grasp
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
        
        # Lift
        lift = transform(
            np.array([block['position'][0], block['position'][1], block['position'][2] + 0.15]),
            np.array([0, pi, pi])
        )
        lift_joints, _, success, _ = ik.inverse(lift, grasp_joints, method='J_pseudo', alpha=0.5)
        if success:
            arm.safe_move_to_position(lift_joints)
            rospy.sleep(0.2)
        
        # Place sequence
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

#############################
## Dynamic Block Functions ##
#############################

def compute_dynamic_intercept(block_position, block_orientation, t_now, omega, theta_pick, R=0.305, z=0.200):
    """
    Compute intercept time and predicted pose for a dynamic block on a rotating turntable.
    """
    x, y, _ = block_position
    theta_now = atan2(y, x)

    dtheta = theta_pick - theta_now
    if dtheta < 0:
        dtheta += 2*pi

    t_intercept = dtheta / omega

    # Future position
    x_intercept = R * cos(theta_pick)
    y_intercept = R * sin(theta_pick)
    z_intercept = z

    # Future orientation
    R_z = np.array([
        [cos(dtheta), -sin(dtheta), 0],
        [sin(dtheta),  cos(dtheta), 0],
        [0,            0,           1]
    ])
    intercept_orientation = R_z @ block_orientation
    intercept_pos = np.array([x_intercept, y_intercept, z_intercept])
    return t_intercept, intercept_pos, intercept_orientation

def orientation_to_rpy(R):
    """
    Convert rotation matrix to roll, pitch, yaw.
    """
    yaw = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = np.arctan2(R[2,1], R[2,2])
    return roll, pitch, yaw

def pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=0.0, R=0.305, z=0.200):
    """
    Attempt to pick a dynamic block off the rotating turntable.
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
                # Identify dynamic blocks: assume "dynamic" in name
                if "dynamic" in name.lower():
                    world_pos = T_CW @ pose
                    current_detections.append({'id': name, 'position': world_pos[:3,3]})
        all_detections.append(current_detections)
        rospy.sleep(0.1)

    filtered_blocks = filter_block_detections(all_detections, num_samples, detection_threshold)
    if not filtered_blocks:
        print("No dynamic blocks detected.")
        return

    dynamic_block = filtered_blocks[0]
    block_position = dynamic_block['position']

    # Get block orientation (from a single detection)
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
    t_intercept, intercept_pos, intercept_orientation = compute_dynamic_intercept(
        block_position, block_orientation, t_now, omega, theta_pick, R=R, z=z
    )

    # Extract RPY
    roll, pitch, yaw = orientation_to_rpy(intercept_orientation)

    # Approach from above
    intercept_above = np.array([intercept_pos[0], intercept_pos[1], intercept_pos[2] + 0.10])
    # Base orientation: [0, pi, pi], add yaw
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
    
    # Convert target pose to joint angles using IK
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
    _, T_EW = fk.forward(joints_array)  # Forward kinematics
    T_CW = T_EW @ H_ee_camera
    
    # Detect and pick static blocks
    block_positions = detect_static(detector, T_CW)
    if block_positions:
        pick_place_static(arm, block_positions, place_target)
    else:
        print("No blocks detected (static)")
    
    pick_dynamic_block(arm, detector, fk, ik, omega, theta_pick=theta_pick)

if __name__ == "__main__":
    main()
