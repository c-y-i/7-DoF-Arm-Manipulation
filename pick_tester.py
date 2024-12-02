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
from manager import MockArmController
from lib.calculateFK import FK
from lib.IK_position_null import IK
from math import pi
import time
from detect_visual import DetectVisualTester 


class VisualTester:
    def __init__(self, pick_pos, place_pos, save_animation=False):
        """Initialize visualizer with pick and place positions."""
        self.visualizer = DetectVisualTester()  # Initialize once for block visualization
        self.save_animation = save_animation
        print(f"\nInitializing Visual Tester...")
        self.arm = MockArmController()
        self.fk = FK()
        self.ik = IK(linear_tol=1e-3, angular_tol=1e-2, max_steps=200)
        self.pick_pos = pick_pos
        self.place_pos = place_pos
        self.create_trajectory()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        self.frame_idx = 0
        self.is_picking = False
        self.block_picked = False
        self.grip_state = 0  # Add grip state tracking
        self.grip_angle = 0  # Add grip angle tracking
        self.animation_complete = False
        self.final_positions = None  # Store final positions for alignment plot
        print("Visual Tester initialized successfully.")
        
    def create_trajectory(self):
        """
        Generate a complete trajectory through waypoints with IK solutions.
        """
        print("\nGenerating trajectory...")
        start_pos = np.array([0.3, 0, 0.5])
        
        z_offset = 0.3
        wp1 = start_pos
        wp2 = self.pick_pos + np.array([0, 0, z_offset])  
        wp3 = self.pick_pos  
        wp4 = self.pick_pos + np.array([0, 0, z_offset])
        wp5 = self.place_pos + np.array([0, 0, z_offset])
        wp6 = self.place_pos
        self.waypoints = np.vstack([wp1, wp2, wp3, wp4, wp5, wp6])
        print(f"Generated {len(self.waypoints)} waypoints")
        self.trajectory = []
        points_between = 10
        target = np.eye(4)
        target[:3,3] = self.waypoints[0]
        target[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
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
        
    def setup_plot(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 0.7])
        self.ax.set_title('MEAM 5200 Final Project\nPick Logic Tester (Visualization)', 
                         pad=20, fontsize=12, fontweight='bold')
        self.fig.text(0.02, 0.98, 'MEAM 5200 - Team 16', 
                     fontsize=8, transform=self.fig.transFigure)
        self.fig.text(0.02, 0.96, 'Fall 2024', 
                     fontsize=8, transform=self.fig.transFigure)
        
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
        """Visualize pick and place positions in the 3D space."""
        self.ax.scatter([self.pick_pos[0]], [self.pick_pos[1]], [self.pick_pos[2]], 
                       color='blue', s=100, label='Pick Position')
        self.ax.scatter([self.place_pos[0]], [self.place_pos[1]], [self.place_pos[2]], 
                       color='green', s=100, label='Place Position')
    
    def update(self, frame):
        """Update the visualization for each animation frame."""
        self.ax.clear()
        self.setup_plot()
        current_joints = self.joint_angles[frame]
        positions, T = self.fk.forward(current_joints)
        self.plot_robot_arm(current_joints)
        self.plot_trajectory()
        
        # Update grip state based on proximity to block
        if self.is_near_pick_position(frame):
            self.grip_state = min(1.0, self.grip_state + 0.2)  # Faster closing
        
        # Create proper gripper transform
        gripper_transform = np.eye(4)
        gripper_transform[:3, :3] = np.array([[0, -1, 0],
                                            [1, 0, 0],
                                            [0, 0, 1]])  # Align gripper with block
        gripper_transform[:3, 3] = positions[-1]  # End effector position
        
        # Create block data
        block_pos = self.pick_pos if not self.block_picked else positions[-1]
        block_data = {
            'id': 'Pick Block',
            'position': block_pos,
            'orientation': np.zeros(3),
            'transform': np.eye(4)
        }
        block_data['transform'][:3, 3] = block_pos
        
        # Visualize block and gripper
        try:
            self.visualizer.plot_grip_sequence(self.ax, block_data, gripper_transform, self.grip_state)
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Show place position
        self.ax.scatter([self.place_pos[0]], [self.place_pos[1]], [self.place_pos[2]], 
                       color='green', s=100, label='Place Position')
        
        self.ax.legend()
        
        # Update frame counter
        self.frame_idx = frame

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
        
        plt.show()

        
    def run(self):
        """
        Execute the visualization and optionally save as GIF.

        Handles both real-time display and animation saving functionality.
        """
        ani = FuncAnimation(self.fig, self.update, frames=len(self.trajectory),
                            interval=50, repeat=False, blit=False)  # Disable blit for proper updates
        
        # Store final positions for alignment plot
        final_frame = len(self.trajectory) - 1
        positions, T = self.fk.forward(self.joint_angles[final_frame])
        self.final_positions = (positions, T)
        
        if self.save_animation:
            print("\nSaving animation as GIF...")
            if not os.path.exists('media'):
                os.makedirs('media')
            frames = []
            for frame in range(len(self.trajectory)):
                self.update(frame)
                fig_canvas = self.fig.canvas
                fig_canvas.draw()
                frame_data = np.frombuffer(fig_canvas.tostring_rgb(), dtype=np.uint8)
                frame_data = frame_data.reshape(fig_canvas.get_width_height()[::-1] + (3,))
                frames.append(Image.fromarray(frame_data))
            filename = f'media/pick_place_{time.strftime("%Y%m%d_%H%M%S")}.gif'
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0
            )
            print(f"Animation saved as: {filename}")
        plt.show(block=True)  # Ensure animation completes
        
        # After animation completes, show alignment plot
        self.plot_final_alignment()

def main():
    """
    Generates random pick and place positions following competition specifications,
    scaled to be within robot workspace.
    """
    print("\n=== Pick and Place Visualization Testing ===")
    
    # Platform specifications (scaled for visualization)
    scale_factor = 0.4  # Scale factor to bring positions into workspace
    platform_height = 0.23 * scale_factor
    platform_center_x = 0.562 * scale_factor
    platform_center_y = 1.159 * scale_factor
    block_size = 0.05  # 50mm blocks
    noise_radius = 0.025 * scale_factor
    
    def noise(radius):
        return float(radius * (np.random.rand() - 0.5))
    
    def generate_static_position():
        # Generate position following block_spawner logic but scaled
        i = np.random.choice([-1, 1])
        j = np.random.choice([-1, 1])
        x = platform_center_x + 2.5*0.0254 * i
        y = platform_center_y + 2.5*0.0254 * j
        
        # Transform to robot frame and scale
        pos = np.array([
            x + noise(noise_radius),
            y + noise(noise_radius),
            platform_height
        ], dtype=float)
        
        # Optional: Add position validation
        if np.linalg.norm(pos[:2]) > 0.4:  # If position is too far
            pos[:2] = pos[:2] * 0.4 / np.linalg.norm(pos[:2])  # Scale back to reasonable range
            
        return pos

    print("\nGenerating positions following competition specifications...")
    pick_pos = generate_static_position()
    place_pos = generate_static_position()
    
    # Ensure pick and place positions are different but not too far
    while 0.05 > np.linalg.norm(pick_pos - place_pos) or np.linalg.norm(pick_pos - place_pos) > 0.3:
        place_pos = generate_static_position()
        
    print(f"Pick position:  {pick_pos}")
    print(f"Place position: {place_pos}")
    print("\nStarting visualization...")
    save_animation = True
    
    visualizer = VisualTester(pick_pos, place_pos, save_animation=save_animation)
    visualizer.run()

if __name__ == "__main__":
    main()