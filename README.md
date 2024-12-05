# 5200-Project
MEAM 5200 Final Project

## Project Overview
This project implements a robotic manipulation system capable of performing pick and place operations using the Lynx robot arm. The implementation includes forward kinematics, inverse kinematics, and trajectory planning.

## Project Structure
    .
    ├── lib/                      # Core kinematics libraries
    │   ├── calculateFK.py        # Forward Kinematics calculation
    │   ├── calcJacobian.py       # Jacobian matrix calculation
    │   ├── IK_position_null.py   # Position-based Inverse Kinematics
    │   ├── IK_velocity_null.py   # Velocity-based Inverse Kinematics
    │   └── FK_velocity.py        # Forward Velocity Kinematics
    ├── core/                     # Core implementation files
    │   ├── interfaces.py        # Interface definitionsn
    │   ├── safety.py           # Safety checks and constraints
    │   └── utils.py            # Utility functions
    └── pick_place_demo.py        # Main demonstration script

## Pick and Place Demonstrations

### Single Block Demonstration
The robot demonstrates precise control by picking and placing a single block:

![Single Pick and Place Demo](media/pick_animation.gif)

### Multiple Block Manipulation
Advanced demonstration showing the robot handling multiple blocks in sequence:

![Multiple Block Pick and Place Demo](media/pick_place_visualization.gif)

## Implementation Details
The `pick_place_demo.py` script implements the core functionality for both single and multiple block manipulation. It utilizes:
- Forward and Inverse Kinematics
- Trajectory Planning
- Obstacle Avoidance
- Gripper Control

