"""
Block Detection Visualization Tester
Author: Team 16
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

class DetectVisualTester:
    """Tester class for block detection visualization."""

    def __init__(self):
        self.gripper_width = 0.08  # Increased width
        self.finger_length = 0.05  # Increased length
        self.grip_state = 0
        self.block_size = 0.05  # 5cm blocks

    def test_with_generated_data(self):
        """
        Function for testing with generated block and gripper data.
        """
        # Generate test data
        position = np.array([0.562, 0.0, 0.2])
        orientation = np.array([0, 0, pi / 6])
        block_transform = np.eye(4)
        block_transform[:3, :3] = R.from_euler('xyz', orientation).as_matrix()
        block_transform[:3, 3] = position
        block_data = {
            'id': 'test_block',
            'position': position,
            'orientation': orientation,
            'transform': block_transform
        }
        gripper_position = np.array([0.55, 0.0, 0.25])
        gripper_orientation = np.array([0, 0, 0])
        gripper_transform = np.eye(4)
        gripper_transform[:3, :3] = R.from_euler('xyz', gripper_orientation).as_matrix()
        gripper_transform[:3, 3] = gripper_position
        # Display and visualize generated block and gripper
        self.display_block(block_data)
        # Initialize figure and axes only once
        fig = plt.figure(figsize=(18, 6))
        ax2d_top = fig.add_subplot(131)
        ax2d_front = fig.add_subplot(132)
        ax3d = fig.add_subplot(133, projection='3d')
        
        # Pass the axes to the visualization function
        self.visualize_gripper_and_block(block_data, gripper_transform, ax2d_top, ax2d_front, ax3d)
        plt.show()

    def display_block(self, block):
        """Display block information."""
        print(f"Block {block['id']}:")
        print(f"  Position: {block['position']}")
        print(f"  Orientation (xyz Euler angles): {block['orientation']}")

    def visualize_gripper_and_block(self, block, gripper_transform, ax2d_top, ax2d_front, ax3d):
        """Visualize the block and gripper alignment."""
        # Use the passed axes for plotting
        # Plot block in 2D (top view)
        block_size = 0.05  # Assuming block is a 5cm cube
        block_x, block_y = block['position'][0], block['position'][1]
        block_rect_top = plt.Rectangle((block_x - block_size / 2, block_y - block_size / 2), block_size, block_size, edgecolor='b', facecolor='none')
        ax2d_top.add_patch(block_rect_top)
        ax2d_top.text(block_x, block_y, block['id'], fontsize=12, ha='center')

        # Plot gripper in 2D (top view)
        gripper_x, gripper_y = gripper_transform[0, 3], gripper_transform[1, 3]
        ax2d_top.plot(gripper_x, gripper_y, 'ro')  # Gripper position
        ax2d_top.text(gripper_x, gripper_y, 'Gripper', fontsize=10, ha='right')

        ax2d_top.set_xlabel('X Position (m)')
        ax2d_top.set_ylabel('Y Position (m)')
        ax2d_top.set_title('Gripper and Block Alignment (Top View)')
        ax2d_top.set_aspect('equal')
        ax2d_top.axis('off')  # Remove grid lines

        # Plot block in 2D (front view)
        block_z = block['position'][2]
        block_rect_front = plt.Rectangle((block_x - block_size / 2, block_z - block_size / 2), block_size, block_size, edgecolor='b', facecolor='none')
        ax2d_front.add_patch(block_rect_front)
        ax2d_front.text(block_x, block_z, block['id'], fontsize=12, ha='center')

        # Plot gripper in 2D (front view)
        gripper_z = gripper_transform[2, 3]
        ax2d_front.plot(gripper_x, gripper_z, 'ro')  # Gripper position
        ax2d_front.text(gripper_x, gripper_z, 'Gripper', fontsize=10, ha='right')

        ax2d_front.set_xlabel('X Position (m)')
        ax2d_front.set_ylabel('Z Position (m)')
        ax2d_front.set_title('Gripper and Block Alignment (Front View)')
        ax2d_front.set_aspect('equal')
        ax2d_front.axis('off')  # Remove grid lines

        # Plot block and gripper in 3D
        self.plot_block_3d(ax3d, block)
        ax3d.scatter(gripper_x, gripper_y, gripper_z, color='r', s=100, label='Gripper')
        
        ax3d.set_xlabel('X Position (m)')
        ax3d.set_ylabel('Y Position (m)')
        ax3d.set_zlabel('Z Position (m)')
        ax3d.set_title('Gripper and Block Alignment (3D View)')
        ax3d.legend()
        ax3d.grid(False)  # Remove grid lines

    def plot_block_3d(self, ax, block):
        """Plot the block as a 3D cube."""
        block_size = 0.05  # Assuming block is a 5cm cube
        block_x, block_y, block_z = block['position']
        orientation = block['orientation']
        rotation = R.from_euler('xyz', orientation).as_matrix()

        # Define the vertices of the cube
        vertices = np.array([
            [-block_size / 2, -block_size / 2, -block_size / 2],
            [ block_size / 2, -block_size / 2, -block_size / 2],
            [ block_size / 2,  block_size / 2, -block_size / 2],
            [-block_size / 2,  block_size / 2, -block_size / 2],
            [-block_size / 2, -block_size / 2,  block_size / 2],
            [ block_size / 2, -block_size / 2,  block_size / 2],
            [ block_size / 2,  block_size / 2,  block_size / 2],
            [-block_size / 2,  block_size / 2,  block_size / 2]
        ])

        # Rotate and translate the vertices
        vertices = np.dot(vertices, rotation.T) + block['position']

        # Define the 12 edges of the cube
        edges = [
            [vertices[0], vertices[1]], [vertices[1], vertices[2]], [vertices[2], vertices[3]], [vertices[3], vertices[0]],  # Bottom face
            [vertices[4], vertices[5]], [vertices[5], vertices[6]], [vertices[6], vertices[7]], [vertices[7], vertices[4]],  # Top face
            [vertices[0], vertices[4]], [vertices[1], vertices[5]], [vertices[2], vertices[6]], [vertices[3], vertices[7]]   # Side edges
        ]

        # Plot the edges
        for edge in edges:
            ax.plot3D(*zip(*edge), color='b')

    def plot_gripper_3d(self, ax, transform, grip_state=0):
        """Plot gripper in 3D with opening/closing animation."""
        if not isinstance(transform, np.ndarray) or transform.shape != (4, 4):
            print("Warning: Invalid transform, using identity")
            transform = np.eye(4)
        
        # Base gripper parameters
        width = max(0.01, self.gripper_width * (1 - grip_state))  # Prevent complete closure
        length = self.finger_length
        height = 0.02  # Increased thickness
        
        # Define gripper points including palm
        palm_width = 0.03
        palm = np.array([
            [-0.01, -palm_width/2, 0],
            [-0.01, palm_width/2, 0],
            [-0.01, palm_width/2, height],
            [-0.01, -palm_width/2, height]
        ])
        
        left_finger = np.array([
            [0, width/2, 0],
            [length, width/2, 0],
            [length, width/2, height],
            [0, width/2, height]
        ])
        
        right_finger = np.array([
            [0, -width/2, 0],
            [length, -width/2, 0],
            [length, -width/2, height],
            [0, -width/2, height]
        ])
        
        try:
            # Transform and plot all parts
            for part in [palm, left_finger, right_finger]:
                points_h = np.hstack([part, np.ones((part.shape[0], 1))])
                transformed = np.dot(transform, points_h.T).T[:, :3]
                
                # Create surface
                poly = Poly3DCollection([transformed], alpha=0.5, color='red')
                ax.add_collection3d(poly)
                
                # Plot edges
                ax.plot(transformed[[0,1,2,3,0], 0],
                       transformed[[0,1,2,3,0], 1],
                       transformed[[0,1,2,3,0], 2],
                       'r-', linewidth=2)
            
            # Add gripper base point
            base_point = transform[:3, 3]
            ax.scatter([base_point[0]], [base_point[1]], [base_point[2]], 
                      color='red', s=100, marker='o')
        
        except Exception as e:
            print(f"Error plotting gripper: {e}")
            ax.scatter([transform[0,3]], [transform[1,3]], [transform[2,3]], 
                      color='red', s=100, marker='o')

    def plot_grip_sequence(self, ax, block_data, gripper_transform, grip_state):
        """Plot both block and gripper with current gripping state."""
        self.plot_block_3d(ax, block_data)
        self.plot_gripper_3d(ax, gripper_transform, grip_state)

    def calculate_grip_transform(self, block_pos, grip_angle=0):
        """
        Calculate gripper transform for gripping the block.
        Args:
            block_pos: 3D position of block center
            grip_angle: Angle around z-axis for gripper approach
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        # Rotate gripper to align with block
        transform[:3, :3] = R.from_euler('xyz', [0, 0, grip_angle]).as_matrix()
        # Position gripper at block
        transform[:3, 3] = block_pos
        return transform

    def plot_block(self, ax, block_data):
        """Plot a block as a colored cube."""
        pos = block_data['position']
        block_size = 0.05  # 5cm blocks
        color = block_data.get('color', 'blue')  # Use provided color or default to blue
        
        # Create cube vertices
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * (block_size/2)
        
        # Transform vertices
        R = Rotation.from_euler('xyz', block_data['orientation']).as_matrix()
        vertices = (R @ vertices.T).T + pos
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
        
        # Plot faces
        collection = Poly3DCollection(faces, alpha=0.5)
        collection.set_facecolor(color)
        ax.add_collection3d(collection)

    def plot_gripper(self, ax, transform, grip_state):
        """Plot gripper visualization with opening state."""
        # Gripper dimensions
        gripper_width = 0.05 * (1 - grip_state)  # Gripper closes as state increases
        finger_length = 0.08
        
        # Base gripper finger positions (in gripper frame)
        fingers = np.array([
            [0, gripper_width, 0],     # Right finger
            [0, -gripper_width, 0],    # Left finger
            [finger_length, gripper_width, 0],    # Right finger tip
            [finger_length, -gripper_width, 0],   # Left finger tip
        ])
        
        # Transform fingers to world frame
        fingers_h = np.hstack([fingers, np.ones((4, 1))])
        fingers_world = (transform @ fingers_h.T).T[:, :3]
        
        # Draw gripper fingers
        ax.plot([fingers_world[0,0], fingers_world[2,0]], 
                [fingers_world[0,1], fingers_world[2,1]],
                [fingers_world[0,2], fingers_world[2,2]], 'r-', linewidth=2)
        ax.plot([fingers_world[1,0], fingers_world[3,0]], 
                [fingers_world[1,1], fingers_world[3,1]],
                [fingers_world[1,2], fingers_world[3,2]], 'r-', linewidth=2)

def main():
    """Main function to test block detection visualization"""
    tester = DetectVisualTester()
    
    print("Starting block detection visualization test...")
    
    try:
        tester.test_with_generated_data()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    print("\nBlock detection visualization test completed!")

if __name__ == "__main__":
    main()