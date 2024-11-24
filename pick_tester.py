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

class VisualTester:
    def __init__(self, pick_pos, place_pos, save_animation=False):
        """Initialize visualizer with pick and place positions."""
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
        """
        Update the visualization for each animation frame. 
        TODO: utlize frame arg
        Args:
            frame (int): Current frame number in the animation
        """
        self.ax.clear()
        self.setup_plot()
        current_joints = self.joint_angles[self.frame_idx]
        self.plot_robot_arm(current_joints)
        self.plot_trajectory()
        self.plot_blocks()
        
        self.ax.legend()
        
        self.frame_idx = (self.frame_idx + 1) % len(self.trajectory)
        
    def run(self):
        """
        Execute the visualization and optionally save as GIF.

        Handles both real-time display and animation saving functionality.
        """
        ani = FuncAnimation(self.fig, self.update, frames=len(self.trajectory),
                            interval=50, repeat=False)
        
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
        plt.show()

def main():
    """
    Generates random pick and place positions within the workspace and
    initiates the visualization process.
    """
    print("\n=== Pick and Place Visualization Testing ===")
    x_range = (-0.25, 0.25)
    y_range = (-0.25, 0.25)
    z_height = 0.25
    def random_pos():
        return np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z_height
        ])
    print("\nGenerating random positions within workspace...")
    while True:
        pick_pos = random_pos()
        place_pos = random_pos()
        if np.linalg.norm(pick_pos - place_pos) > 0.2:
            break
    print(f"Pick position:  {pick_pos}")
    print(f"Place position: {place_pos}")
    print("\nStarting visualization...")
    save_animation = True
    
    visualizer = VisualTester(pick_pos, place_pos, save_animation=save_animation)
    visualizer.run()

if __name__ == "__main__":
    main()