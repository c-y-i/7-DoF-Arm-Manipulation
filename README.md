# MEAM 5200 Final Project
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Franka ROS](https://img.shields.io/badge/Franka-ROS-orange)](https://frankaemika.github.io/docs/franka_ros.html)
[![Panda Robot](https://img.shields.io/badge/Panda-Robot-brightgreen)](https://github.com/justagist/panda_robot)


## Project Overview
This project implements a robotic manipulation system capable of performing pick and place operations using the Panda robot arm. The implementation includes forward kinematics, inverse kinematics, and trajectory planning.

## Project Structure
    .
    ├── lib/                      # Core kinematics libraries
    │   ├── calculateFK.py        # Forward Kinematics calculation
    │   ├── IK_position_null.py   # Position-based Inverse Kinematics
    │   └── franka_IK.py         # Franka-specific Inverse Kinematics (fallback)
    ├── core/                     # Core implementation files
    │   ├── interfaces.py         # Interface definitions
    │   ├── safety.py            # Safety checks and constraints
    │   └── utils.py             # Utility functions
    ├── scripts/                  # Testing and demonstration scripts
    │   ├── mock.py              # Local simulation library for arm control
    │   ├── pick_tester.py       # Testing script using mock controller
    │   ├── static_demo.py       # Static block manipulation demo
    │   └── dynamic_demo.py      # Dynamic block retrieval demo
    ├── config.json              # Team-specific configuration parameters
    └── final.py                 # Main execution script

## Prerequisites
- Python 3.8+
- Franka ROS
- libfranka
- panda_robot Python package
- Ubuntu 20.04 LTS

## Installation
1. Clone the repository

## Pick and Place Demonstrations

### Single Block Demonstration
The robot demonstrates precise control by picking and placing a single block:

![Single Pick and Place Demo](media/pick_animation.gif)

### Multiple Block Manipulation
Advanced demonstration showing the robot handling multiple blocks in sequence:

![Multiple Block Pick and Place Demo](media/pick_place_visualization.gif)


## Final Implementation

The `final.py` script is the main execution file for the project. It integrates all functionalities, including dynamic block retrieval, static block manipulation, and coordinated pick and place operations.

### Key Features

- **Dynamic Block Retrieval**: The robot autonomously retrieves dynamic blocks using predefined poses from the `config.json` file.
- **Static Block Manipulation**: The robot detects and manipulates static blocks using computer vision and kinematic calculations.
- **Configurable Parameters**: Team-specific configurations are loaded from `config.json`, allowing easy adjustments for different scenarios.
- **Error Handling**: Includes fallback mechanisms for inverse kinematics failures and grasping retries.

## How to Use

1. **Setup Environment**:
   - Ensure all dependencies are installed.
   - Verify that the robot arm is properly connected and configured.
   - Launch necessary ROS nodes.

2. **Load Configuration**:
   - The script uses `config.json` for team-specific settings.
   - Update `config.json` with your team's parameters if necessary.

3. **Run the Script**:
   - Execute the main script using ROS:

