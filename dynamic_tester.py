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
from lib.IK2 import IK_plus
from lib.IK_velocity_null import IK_velocity_null  # Import IK_velocity_null

ik = IK()  # Primary IK solver
fk = FK()
arm = ArmController()
detector = ObjectDetector()
target_placing=np.array([0.57, 0.169, 0.27])
q7 = np.pi/4
q_actual_array = [ 0 ]

def complementary_filter(data, reference_data, alpha):
    return alpha * data + (1 - alpha) * reference_data


def block_Detection(T_CW):

    alpha=0.9

    for i in range(5):
        block_pose1=[]
        for (name, pose) in detector.get_detections():
            block_pose1.append(pose)
            print(name,'\n',pose)
        block_pose1=np.array(block_pose1)
        # print(block_pose1.shape)

        block_pose2=[]
        for (name, pose) in detector.get_detections():
            block_pose2.append(pose)
            # print(name,'\n',pose)
        block_pose2=np.array(block_pose2)
        # print(block_pose2.shape)

        block_pose_comp=complementary_filter(block_pose1,block_pose2,alpha)
        block_pose1=block_pose2
        block_pose2=block_pose_comp


    block_pose_world=[]
    for i in range(block_pose_comp.shape[0]):
        block_pose_world.append(T_CW@block_pose_comp[i])

    return block_pose_world

def dynamic_pick():

    arm.open_gripper()
    target= transform( np.array([0.3, 0.745, 0.162]), np.array([ 0,pi,pi]) )
    seed= np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    q, rollout, success, message = ik.inverse(target, seed, method='J_pseudo', alpha=.5)
    #print(q)
    q[4]=q[4]+pi/2-0.45
    q[6] = q[6]-3*pi/4 - pi/8 - pi/20
    q[3] = q[3] - pi/10
    q[5]-=pi/4 -0.3 -0.3 
    q[0]-=pi/24

    # q =[-1.76167489, -1.40459303  ,1.94353466 ,-0.89720455 , 2.28922666 , 1.14506434, -1.14923264]

    q[5]+= pi/10
    # q[4]-=pi/25
    # q[4]-=pi/7
    print(q)

    arm.safe_move_to_position(q)
    
    q[5]-=pi/5
    q[0]+=pi/11
    
    arm.safe_move_to_position(q)
    rospy.sleep(7)

    arm.exec_gripper_cmd(0.0475,60)

    pos=np.linalg.norm(arm.get_gripper_state()["position"])
    print(pos)

    q[5]+=pi/6
    q[0]+=pi/5.5
    q[6]+=pi/8
    q[4]+=pi/18

    arm.safe_move_to_position(q)

    q[1]+=pi/6
    
    arm.safe_move_to_position(q)

    if pos > 0.015:
        target_placing[2]+=0.05
        dynamic_place(target_placing)
        print("Broken")
    
    else:
        target= transform( np.array([0.502, -0.169, 0.38364056]), np.array([ 0,pi,pi]) )
        print(target)
        tar = IK_plus(target, q7, q_actual_array)
        q=tar[2]
        arm.safe_move_to_position(q)
        

   

def dynamic_place(target_placing):
    targets_p = transform(np.array([target_placing[0]+0.1, target_placing[1], target_placing[2]+0.02]), np.array([0, pi, pi]))
    tar = IK_plus(targets_p, q7, q_actual_array)
    q=tar[-1]
    q[5]-=2*pi/15
    arm.safe_move_to_position(q)
    

    targets_p = transform(np.array([target_placing[0]+0.1, target_placing[1], target_placing[2]-0.062]), np.array([0, pi, pi]))
    tar = IK_plus(targets_p, q7, q_actual_array)
    q=tar[-1]
    q[5]-=2*pi/15
    arm.safe_move_to_position(q)
    arm.open_gripper()

    targets_pp = transform(np.array([target_placing[0]+0.05, target_placing[1], target_placing[2]+0.1]), np.array([0, pi, pi]))
    tar = IK_plus(targets_pp, q7, q_actual_array)
    q=tar[-1]
    q[5]-=2*pi/15
    arm.safe_move_to_position(q)

    # target= transform( np.array([0.522, -0.169, 0.58364056]), np.array([ 0,pi,pi]) )

    # print(target)
    # tar = IK_plus(target, q7, q_actual_array)
    # q=tar[2]
    # arm.safe_move_to_position(q)
    # arm.open_gripper()


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!


    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    arm.set_arm_speed(0.3)

    q7 = np.pi/4
    q_actual_array = [ 0 ]

    target= transform( np.array([0.522, -0.169, 0.58364056]), np.array([ 0,pi,pi]) )
    print(target)
    tar = IK_plus(target, q7, q_actual_array)
    q=tar[2]
    arm.safe_move_to_position(q)
    arm.open_gripper()


    H_ee_camera = detector.get_H_ee_camera()
    T_CE = H_ee_camera # T of camera in end_effector
    _,T_EW = fk.forward(q)
    T_CW= T_EW@T_CE

    rospy.sleep(2)

    block_pose_world=block_Detection()


    target_placing=np.array([0.57, 0.169, 0.27])

    dynamic_pick()

    target= transform( np.array([0.502, -0.169, 0.38364056]), np.array([ 0,pi,pi]) )
    print(target)
    tar = IK_plus(target, q7, q_actual_array)
    q=tar[2]
    arm.safe_move_to_position(q)


    dynamic_pick()


    arm.open_gripper()
    target= transform( np.array([0.3, 0.745, 0.162]), np.array([ 0,pi,pi]) )
    seed= np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    q, rollout, success, message = ik.inverse(target, seed, method='J_pseudo', alpha=.5)
    #print(q)
    q[4]=q[4]+pi/2-0.45
    q[6] = q[6]-3*pi/4 - pi/8 - pi/20
    q[3] = q[3] - pi/10
    q[5]-=pi/4 -0.3 -0.3 
    q[0]-=pi/24

    # q =[-1.76167489, -1.40459303  ,1.94353466 ,-0.89720455 , 2.28922666 , 1.14506434, -1.14923264]

    q[5]+= pi/10
    # q[4]-=pi/25
    # q[4]-=pi/7
    print(q)

    arm.safe_move_to_position(q)
    
    q[5]-=pi/5
    q[0]+=pi/11
    
    arm.safe_move_to_position(q)
    rospy.sleep(5)

    # arm.exec_gripper_cmd(0.049,50)

    pos=np.linalg.norm(arm.get_gripper_state()["position"])
    print(pos)

    q[5]+=pi/6
    q[0]+=pi/5.5
    q[6]+=pi/8
    q[4]+=pi/18

    arm.safe_move_to_position(q)

    q[1]+=pi/6
    
    arm.safe_move_to_position(q)

    target= transform( np.array([0.502, -0.169, 0.38364056]), np.array([ 0,pi,pi]) )
    print(target)
    tar = IK_plus(target, q7, q_actual_array)
    q=tar[2]
    arm.safe_move_to_position(q)
