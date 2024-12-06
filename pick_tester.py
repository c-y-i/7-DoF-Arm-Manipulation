"""
Pick and Place Testing Visualizer.
Author: Team 16

TODO: 
- Fix IK failure for certain cases
- More aggressive trajectory plannning? (tho might not be necessary)

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from mock import MockArmController
from lib.calculateFK import FK
from lib.IK_position_null import IK
from math import pi
import time
from detect_visual import DetectVisualTester 
from scipy.spatial.transform import Rotation
from matplotlib.animation import PillowWriter


class VisualTester:
    def __init__(self, blocks_data, place_pos, save_animation=False):
        """Initialize visualizer with blocks data and place positions."""
        self.visualizer = DetectVisualTester()
        self.save_animation = save_animation
        print(f"\nInitializing Visual Tester...")
        self.arm = MockArmController()
        self.fk = FK()
        self.ik = IK(linear_tol=1e-3, angular_tol=1e-2, max_steps=200)
        self.blocks_data = blocks_data
        self.blocks_state = {block['id']: {
            'position': block['position'].copy(),
            'orientation': block['orientation'].copy(),
            'picked': False,
            'placed': False,
            'current_height': 0.05,  # Add initial height
            'original_position': block['position'].copy()  # Store original position
        } for block in blocks_data}
        self.current_block_id = blocks_data[0]['id']
        self.pick_pos = None  # Change this line to initialize as None
        self.place_pos = place_pos.copy()  # Make sure to copy place_pos
        self.trajectories = {}  # Store trajectories for all blocks
        self.joint_angles_all = {}  # Store joint angles for all blocks
        self.trajectory_colors = {  # Add colors for each trajectory
            'block_0': 'blue',
            'block_1': 'green',
            'block_2': 'red',
            'block_3': 'purple'
        }
        self.frame_count = 0  # Total frames across all trajectories
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        self.frame_idx = 0
        self.is_picking = False
        self.block_picked = False
        self.blocks_being_moved = set()  # Add this to track which blocks are moving
        self.grip_state = 0  # Add grip state tracking
        self.grip_angle = 0  # Add grip angle tracking
        self.animation_complete = False
        self.final_positions = None  # Store final positions for alignment plot
        self.grasp_success = False
        self.stack_height = self.place_pos[2]
        self.block_colors = {  # Add block colors
            'block_0': 'blue',
            'block_1': 'green',
            'block_2': 'red',
            'block_3': 'purple'
        }
        self.current_trajectory = None  # Add this line
        self.current_joints = None      # Add this line
        self.active_trajectory_segment = None  # Add this line to track current segment
        print("Visual Tester initialized successfully.")
        
    def create_trajectory(self):
        """
        Generate a complete trajectory through waypoints with IK solutions.
        """
        print("\nGenerating trajectory...")
        start_pos = np.array([0.3, 0, 0.5])
        
        z_offset = 0.15  # Reduced from 0.3 to match pick_place_demo
        wp1 = start_pos
        wp2 = self.pick_pos + np.array([0, 0, z_offset])  # Pre-grasp
        wp3 = self.pick_pos + np.array([0, 0, 0.01])      # Grasp
        wp4 = self.pick_pos + np.array([0, 0, z_offset])  # Post-grasp lift
        wp5 = self.place_pos + np.array([0, 0, z_offset]) # Pre-place
        wp6 = self.place_pos + np.array([0, 0, 0.02])     # Place
        wp7 = self.place_pos + np.array([0, 0, z_offset]) # Retreat
        self.waypoints = np.vstack([wp1, wp2, wp3, wp4, wp5, wp6, wp7])
        print(f"Generated {len(self.waypoints)} waypoints")
        self.trajectory = []
        points_between = 10
        target = np.eye(4)
        target[:3,3] = self.waypoints[0]
        target[:3,:3] = Rotation.from_euler('xyz', [0, pi, pi]).as_matrix()
        initial_seeds = [
            np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4]),
            np.array([0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]),
            np.array([0, pi/4, 0, -pi/4, 0, pi/2, pi/4])
        ]
        print("Finding initial IK solution...")
        for seed in initial_seeds:
            joints, _, success, _ = self.ik.inverse(target, seed, method='J_pseudo', alpha=0.1)
            if success:
                self.joint_angles = [joints]
                print("Initial IK solution found.")
                break
        else:
            raise Exception("Could not find initial IK solution")
        print("\nGenerating trajectory points...")
        total_points = (len(self.waypoints)-1) * points_between
        current_point = 0
        for i in range(len(self.waypoints)-1):
            points = np.linspace(self.waypoints[i], self.waypoints[i+1], points_between)
            for pos in points[1:]:
                current_point += 1
                print(f"Processing point {current_point}/{total_points}", end='\r')
                target = np.eye(4)
                target[:3,3] = pos
                target[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                joints, _, success, _ = self.ik.inverse(target, self.joint_angles[-1], 
                                                      method='J_pseudo', alpha=0.1)
                if success:
                    self.joint_angles.append(joints)
                    self.trajectory.append(pos)
                else:
                    print(f"\nIK failed for position {pos}, using interpolation")
                    self.joint_angles.append(self.joint_angles[-1])
                    self.trajectory.append(pos)
                    
        self.trajectory = np.array(self.trajectory)
        self.joint_angles = np.array(self.joint_angles)
        print(f"\nTrajectory generated with {len(self.trajectory)} points")
        
    def create_all_trajectories(self):
        """Generate trajectories for all blocks at initialization."""
        print("\nGenerating trajectories for all blocks...")
        current_height = self.place_pos[2]
        
        # Store original place position
        original_place_pos = self.place_pos.copy()
        
        for block_id, block_state in self.blocks_state.items():
            print(f"\nGenerating trajectory for {block_id}...")
            self.pick_pos = block_state['position'].copy()  # Set pick position
            temp_place = self.place_pos.copy()
            temp_place[2] = current_height
            self.place_pos = temp_place
            
            # Store the target height for this block
            block_state['current_height'] = current_height
            
            # Generate trajectory for this block
            self.create_trajectory()
            self.trajectories[block_id] = self.trajectory.copy()
            self.joint_angles_all[block_id] = self.joint_angles.copy()
            
            # Update height for next block
            current_height += 0.05
            
        # Restore original place position
        self.place_pos = original_place_pos
        
        # Calculate total frames
        self.frame_count = sum(len(traj) for traj in self.trajectories.values())
        print(f"\nGenerated {len(self.trajectories)} trajectories with total {self.frame_count} frames")

    def generate_single_trajectory(self):
        """Generate trajectory for a single block."""
        # Move existing trajectory generation code here
        # ...copy existing create_trajectory() code here...
        return self.trajectory, self.joint_angles

    def setup_plot(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 0.7])
        self.ax.set_title('MEAM 5200 Final Project\nPick Logic Tester (Visualization)', 
                         pad=20, fontsize=14, fontweight='bold')
        self.fig.text(0.02, 0.98, 'MEAM 5200 - Team 16', 
                     fontsize=12, transform=self.fig.transFigure)
        self.fig.text(0.02, 0.96, 'Fall 2024', 
                     fontsize=12, transform=self.fig.transFigure)
        
    def plot_robot_arm(self, joint_angles):
        """
        Render the robot arm in 3D space.
        Args:
            joint_angles (np.ndarray): Current joint angles of the robot
        """
        positions, _ = self.fk.forward(joint_angles)
        for i in range(len(positions)-1):
            self.ax.plot([positions[i,0], positions[i+1,0]],
                        [positions[i,1], positions[i+1,1]],
                        [positions[i,2], positions[i+1,2]],
                        color='orange', linewidth=3)
        self.ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                       color='purple', s=70)
        self.ax.scatter(positions[-1,0], positions[-1,1], positions[-1,2],
                       color='cyan', s=120, label='End Effector')
    
    def plot_trajectory(self):
        """Draw the planned trajectory path in the visualization."""
        self.ax.plot(self.trajectory[:,0], self.trajectory[:,1], 
                    self.trajectory[:,2], 'g--', alpha=0.5, label='Planned Path')
    
    def plot_blocks(self):
        """Visualize all blocks in the 3D space."""
        for block in self.blocks_data:
            pos = block['position']
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color='blue', s=100)
        self.ax.scatter([self.place_pos[0]], [self.place_pos[1]], [self.place_pos[2]], 
                       color='green', s=100, label='Place Position')
    
    def update(self, frame):
        """Update the visualization for each animation frame."""
        self.ax.clear()
        self.setup_plot()
        
        # Find current block and frame
        total_frames = 0
        current_frame = 0
        for block_id, trajectory in self.trajectories.items():
            if frame < total_frames + len(trajectory):
                self.current_block_id = block_id
                current_frame = frame - total_frames
                self.current_trajectory = trajectory
                self.current_joints = self.joint_angles_all[block_id]
                break
            total_frames += len(trajectory)
        
        # Get current position and joint angles
        current_pos = self.current_trajectory[current_frame]
        current_joint_angles = self.current_joints[current_frame]
        positions, _ = self.fk.forward(current_joint_angles)
        
        # Update grip state based on height difference
        height_diff = current_pos[2] - self.blocks_state[self.current_block_id]['original_position'][2]
        if abs(height_diff) < 0.03:  # Close to pick height
            self.grip_state = min(1.0, self.grip_state + 0.2)
            if self.grip_state >= 1.0:
                self.blocks_state[self.current_block_id]['picked'] = True
        elif height_diff > 0.1:  # Moving up with block
            self.grip_state = 1.0
        
        # Update block positions
        for block_id, block_state in self.blocks_state.items():
            if block_id == self.current_block_id:
                if block_state['picked'] and not block_state['placed']:
                    # Move block with end effector
                    block_state['position'] = positions[-1] - np.array([0, 0, 0.03])
                elif not block_state['picked']:
                    # Keep at original position
                    block_state['position'] = block_state['original_position']
            
            # Draw block
            self.visualizer.plot_block(self.ax, {
                'id': block_id,
                'position': block_state['position'],
                'orientation': block_state['orientation']
            })
        
        # Plot robot arm and trajectories
        self.plot_robot_arm(current_joint_angles)
        
        # Plot current trajectory more prominently
        for block_id, trajectory in self.trajectories.items():
            color = self.trajectory_colors[block_id]
            if block_id == self.current_block_id:
                # Plot completed path
                if current_frame > 0:
                    self.ax.plot(trajectory[:current_frame,0], 
                               trajectory[:current_frame,1], 
                               trajectory[:current_frame,2], 
                               color=color, alpha=0.8, linewidth=2)
                # Plot future path
                if current_frame < len(trajectory):
                    self.ax.plot(trajectory[current_frame:,0], 
                               trajectory[current_frame:,1], 
                               trajectory[current_frame:,2], 
                               '--', color=color, alpha=0.3)
            else:
                self.ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 
                           '--', color=color, alpha=0.2)
        
        # Update gripper
        gripper_transform = np.eye(4)
        gripper_transform[:3, :3] = Rotation.from_euler('xyz', [0, pi, 0]).as_matrix()
        gripper_transform[:3, 3] = positions[-1]
        self.visualizer.plot_gripper(self.ax, gripper_transform, self.grip_state)
        
        # Show place position
        self.ax.scatter([self.place_pos[0]], [self.place_pos[1]], [self.place_pos[2]], 
                       color='green', s=100, label='Place Position')
        
        self.ax.legend()
        return self.ax  # Return the axes for proper animation

    def is_near_pick_position(self, idx):
        """Check if the current position is near the pick position."""
        threshold = 0.05  # Set a threshold for proximity
        current_pos = self.trajectory[idx]
        distance = np.linalg.norm(current_pos - self.pick_pos)
        
        # Update picking state
        if distance < threshold and not self.block_picked:
            self.is_picking = True
            if distance < 0.01:  # Very close to pick position
                self.block_picked = True
        return self.is_picking

    def check_grasp_success(self, frame):
        """Simulate grasp success check based on proximity and timing."""
        if not self.grasp_success:
            # Check if we're at the grasp position (wp3)
            current_pos = self.trajectory[frame]
            distance_to_pick = np.linalg.norm(current_pos - self.pick_pos)
            if distance_to_pick < 0.02:  # If very close to pick position
                self.grasp_success = True
                self.grip_state = 1.0  # Close gripper
                print("Grasp successful!")
        return self.grasp_success

    def plot_final_alignment(self):
        """Plot 2D visualization of final gripper-block alignment."""
        if self.final_positions is None:
            return
            
        positions, T = self.final_positions
        
        # Create new figure for alignment plot
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Block dimensions
        block_size = 0.05
        block_pos = self.place_pos
        
        # Calculate gripper finger positions (in gripper frame)
        gripper_width = 0.05  # Half the width between fingers
        finger_positions = np.array([
            [0, gripper_width, 0],    # Left finger
            [0, -gripper_width, 0],   # Right finger
        ])
        
        # Transform finger positions to world frame
        gripper_transform = np.eye(4)
        gripper_transform[:3, :3] = np.array([[0, -1, 0],
                                            [1, 0, 0],
                                            [0, 0, 1]])  # Align gripper with block
        gripper_transform[:3, 3] = positions[-1]  # End effector position
        
        # Transform fingers to world frame
        fingers_h = np.hstack([finger_positions, np.ones((2, 1))])
        fingers_world = (gripper_transform @ fingers_h.T).T[:, :3]
        
        # Plot top view (XY plane)
        ax_xy.set_title('Top View (XY Plane)')
        # Plot block
        block_rect = plt.Rectangle(
            (block_pos[0] - block_size/2, block_pos[1] - block_size/2),
            block_size, block_size, 
            color='blue', alpha=0.3, label='Block'
        )
        ax_xy.add_patch(block_rect)
        # Plot gripper fingers
        ax_xy.scatter(fingers_world[:, 0], fingers_world[:, 1], 
                     color='red', s=100, label='Gripper Fingers')
        # Draw line connecting fingers
        ax_xy.plot(fingers_world[:, 0], fingers_world[:, 1], 'r-', alpha=0.5)
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.grid(True)
        ax_xy.axis('equal')
        
        # Plot front view (XZ plane)
        ax_xz.set_title('Front View (XZ Plane)')
        # Plot block
        block_rect = plt.Rectangle(
            (block_pos[0] - block_size/2, block_pos[2] - block_size/2),
            block_size, block_size, 
            color='blue', alpha=0.3, label='Block'
        )
        ax_xz.add_patch(block_rect)
        # Plot gripper fingers
        ax_xz.scatter(fingers_world[:, 0], fingers_world[:, 2], 
                     color='red', s=100, label='Gripper Fingers')
        # Draw line connecting fingers in front view
        ax_xz.plot(fingers_world[:, 0], fingers_world[:, 2], 'r-', alpha=0.5)
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.grid(True)
        ax_xz.axis('equal')
        
        # Add legends and adjust plot
        ax_xy.legend()
        ax_xz.legend()
        plt.tight_layout()
        
        # Add some metrics
        alignment_error = np.abs(fingers_world[0, 1] - fingers_world[1, 1])
        fig.suptitle(f'Gripper Alignment (Error: {alignment_error:.3f}m)', y=1.05)
        
        # Add grasp success indicator
        if self.grasp_success:
            fig.text(0.02, 0.94, 'Grasp: SUCCESS', 
                    color='green', fontsize=10, transform=fig.transFigure)
        else:
            fig.text(0.02, 0.94, 'Grasp: FAILED', 
                    color='red', fontsize=10, transform=fig.transFigure)
        
        plt.show()

        
    def run(self):
        """
        Execute visualization for all blocks sequentially.
        """
        self.create_all_trajectories()
        
        # Create animation with proper frame handling
        ani = FuncAnimation(
            self.fig, 
            self.update,
            frames=self.frame_count,
            interval=50,  # Slower for smoother animation
            blit=False,
            repeat=False
        )
        
        if self.save_animation:
            print("\nSaving animation as GIF...")
            ani.save(
                "media/pick_place_visualization.gif",
                writer='pillow',
                fps=20,
                progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
            )
            print("Animation saved successfully!")
        
        plt.show()

def main():
    """
    Generates random pick and place positions following competition specifications,
    scaled to be within robot workspace.
    """
    print("\n=== Pick and Place Visualization Testing ===")
    
    # Define block positions in a square pattern
    block_positions = [
        {'pos': [0.3, 0.3], 'orient': [0, pi, 0]},
        {'pos': [0.3, -0.3], 'orient': [0, pi, pi/4]},
        {'pos': [-0.3, 0.3], 'orient': [0, pi, -pi/4]},
        {'pos': [-0.3, -0.3], 'orient': [0, pi, pi/2]}
    ]
    
    blocks_data = []
    for i, pos in enumerate(block_positions):
        blocks_data.append({
            'id': f'block_{i}',
            'position': np.array([pos['pos'][0], pos['pos'][1], 0.05]),
            'orientation': np.array(pos['orient'])
        })
    
    # Place position (stack blocks at this location)
    place_pos = np.array([0.4, 0, 0.05])
    
    print("\nStarting visualization...")
    # Change this line to enable animation saving:
    visualizer = VisualTester(blocks_data, place_pos, save_animation=True)
    visualizer.run()

if __name__ == "__main__":
    main()