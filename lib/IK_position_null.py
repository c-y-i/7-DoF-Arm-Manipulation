import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
# from lib.IK_velocity import IK_velocity  #optional

"""
Lab 3 and Part 4: Numerical Inverse Kinematics with Secondary Task
"""

class IK:
    # JOINT LIMITS
    joint_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    joint_center_positions = joint_lower_limits + (joint_upper_limits - joint_lower_limits) / 2  # compute middle of range of motion of each joint
    forward_kinematics = FK()

    def __init__(self, linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.
        
        PARAMETERS:
        linear_tol - the maximum position_error in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum orientation_error of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """
        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        position_displacement - a 3-element numpy array containing the position_displacement from
        the current frame to the target frame, expressed in the world frame
        rotation_axis - a 3-element numpy array containing the rotation_axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(orientation_error), where orientation_error is the orientation_error of rotation around this rotation_axis
        """
        position_displacement = target[:3, 3] - current[:3, 3]
        rotation_axis = calcAngDiff(target[:3, :3], current[:3, :3])
        return position_displacement, rotation_axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the position_error and orientation_error between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        position_error - the position_error in meters between the origins of G & H
        orientation_error - the orientation_error in radians between the orientations of G & H
        """
        position_error = np.linalg.norm(G[:3, 3] - H[:3, 3])
        rotation_matrix = G[:3, :3] @ H[:3, :3].T
        orientation_error = acos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))
        return position_error, orientation_error

    def is_valid_solution(self, q, target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """
        success = False
        message = "Solution found/not found + reason"

        _, end_effector_transformation_matrix = IK.forward_kinematics.forward(q)
        position_error, orientation_error = self.distance_and_angle(end_effector_transformation_matrix, target)
        if np.all(q >= self.joint_lower_limits) and np.all(q <= self.joint_upper_limits):
            if abs(position_error) < self.linear_tol and abs(orientation_error) < self.angular_tol:
                success = True
                message = "Solution found"
            else:
                success = False
                message = "Solution not found : tolerances"
        else:
            success = False
            message = "Solution not found: joint angles error"

        return success, message

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q, target, method):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (jacobian_matrix pseudo-inverse or jacobian_matrix transpose) in your algorithm
        
        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """
        jacobian_matrix = calcJacobian(q)
        _, current_transformation_matrix = IK.forward_kinematics.forward(q)
        position_displacement, rotation_axis = IK.displacement_and_axis(target, current_transformation_matrix)
        task_space = np.concatenate((position_displacement, rotation_axis))

        nan_indices = np.isnan(task_space)
        valid_indices = ~nan_indices
        task_space = task_space[valid_indices]
        jacobian_matrix = jacobian_matrix[valid_indices, :]

        if method == 'J_pseudo':
            jacobian_pseudo_inverse = np.linalg.pinv(jacobian_matrix, rcond=1e-10)
            dq = jacobian_pseudo_inverse @ task_space
        elif method == 'J_trans':
            dq = jacobian_matrix.T @ task_space
        else:
            raise ValueError("Invalid method specified: choose 'J_pseudo' or 'J_trans'")
        
        return dq

    @staticmethod
    def joint_centering_task(q, rate=5e-1):
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's orientation_error and the joint_center_positions of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to joint_center_positions the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """
        offset = 2 * (q - IK.joint_center_positions) / (IK.joint_upper_limits - IK.joint_lower_limits)
        dq = rate * -offset  # proportional term (implied quadratic cost)
        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, method, alpha):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (jacobian_matrix pseudo-inverse or jacobian_matrix transpose) in your algorithm

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        iteration_history - a list containing the guess for q at each iteration of the algorithm
        """
        q = seed
        iteration_history = []

        while True:
            iteration_history.append(q)
            primary_task_velocity = IK.end_effector_task(q, target, method)
            secondary_task_velocity = IK.joint_centering_task(q)
            jacobian_matrix = calcJacobian(q)
            jacobian_pseudo_inverse = np.linalg.pinv(jacobian_matrix, rcond=1e-10)
            null_space_projector = np.eye(jacobian_matrix.shape[1]) - jacobian_pseudo_inverse @ jacobian_matrix
            dq = primary_task_velocity + alpha * (null_space_projector @ secondary_task_velocity)

            if np.linalg.norm(dq) < self.min_step_size or len(iteration_history) >= self.max_steps:
                break

            q = q + alpha * dq

        success, message = self.is_valid_solution(q, target)
        return q, iteration_history, success, message

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    target = np.array([
        [0, -1, 0, -0.2],
        [-1, 0, 0, 0],
        [0, 0, -1, 0.5],
        [0, 0, 0, 1],
    ])

    # Using pseudo-inverse
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=0.5)
    for i, q_pseudo in enumerate(rollout_pseudo):
        joints, pose = ik.forward_kinematics.forward(q_pseudo)
        d, ang = IK.distance_and_angle(target, pose)
        print('iteration:', i, ' q =', q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d, ang=ang))

    # Using transpose
    q_trans, rollout_trans, success_trans, message_trans = ik.inverse(target, seed, method='J_trans', alpha=0.5)
    for i, q_trans in enumerate(rollout_trans):
        joints, pose = ik.forward_kinematics.forward(q_trans)
        d, ang = IK.distance_and_angle(target, pose)
        print('iteration:', i, ' q =', q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d, ang=ang))

    # compare
    print("\n method: J_pseudo-inverse")
    print("   Success: ", success_pseudo, ":  ", message_pseudo)
    print("   Solution: ", q_pseudo)
    print("   #Iterations : ", len(rollout_pseudo))
    print("\n method: J_transpose")
    print("   Success: ", success_trans, ":  ", message_trans)
    print("   Solution: ", q_trans)
    print("   #Iterations :", len(rollout_trans), '\n')
