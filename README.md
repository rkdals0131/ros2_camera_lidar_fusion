# ROS2 Camera-LiDAR Fusion

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![ROS2 Version](https://img.shields.io/badge/ROS-Humble-green)](https://docs.ros.org/en/humble/index.html)

A ROS2 package for calculating **intrinsic** and **extrinsic calibration** between camera and LiDAR sensors. This repository provides an intuitive workflow to fuse data from these sensors, enabling precise projection of LiDAR points into the camera frame and offering an efficient approach to sensor fusion.

## Visual Overview

| **Static Sensors** | **Moving Sensors** |
|---------------------|--------------------|
| <img src="https://github.com/CDonosoK/ros2_camera_lidar_fusion/blob/dev/assets/static_lidar.gif" alt="Static Sensors" width="400"> | <img src="https://github.com/CDonosoK/ros2_camera_lidar_fusion/blob/dev/assets/moving_sensors.gif" alt="Moving Sensors" width="400"> |

---

## Table of Contents
1. [Get Started](#get-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
2. [Usage](#usage)
   - [Node Overview](#node-overview)
   - [Workflow](#workflow)
   - [Running Nodes](#running-nodes)
3. [License](#license)
4. [TODO](#todo)

---

## Get Started

### Prerequisites

To run this package, ensure the following dependencies are installed:
- **Git**: For version control and repository management.
- **Docker**: To streamline the environment setup and execution.
- **NVIDIA Container Toolkit** (if using an NVIDIA GPU): For hardware acceleration.

### Installation

#### Clone the Repository
Start by cloning the repository:
```bash
git clone git@github.com:CDonosoK/ros2_camera_lidar_fusion.git
```

#### Build Using Docker
This repository includes a pre-configured Docker setup for easy deployment. To build the Docker image:
1. Navigate to the `docker` directory:
   ```bash
   cd ros2_camera_lidar_fusion/docker
   ```
2. Run the build script:
   ```bash
   sh build.sh
   ```
   This will create a Docker image named `ros2_camera_lidar_fusion`.

#### Run the Docker Container
Once built, launch the container using:
```bash
sh run.sh
```

---

## Usage

### Node Overview
This package includes the following ROS2 nodes for camera and LiDAR calibration:

| **Node Name**           | **Description**                                                                                       | **Output**                                     |
|--------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `camera_calibration.py`  | Computes the intrinsic calibration of the camera.                                                    | Camera intrinsic calibration file.            |
| `save_sensor_data.py`    | Records synchronized data from camera and LiDAR sensors.                                             | Sensor data file.                             |
| `extract_points.py`      | Allows manual selection of corresponding points between camera and LiDAR.                            | Corresponding points file.                    |
| `calculate_calibration.py` | Computes the extrinsic calibration between camera and LiDAR sensors.                                | Extrinsic calibration file.                   |
| `lidar_camera_projection.py` | Projects LiDAR points into the camera frame using intrinsic and extrinsic calibration parameters. | Visualization of projected points.            |

### Workflow
Follow these steps to perform calibration and data fusion:

1. **Intrinsic Calibration**  
   Run `camera_calibration.py` to generate the intrinsic calibration file for the camera.

2. **Data Recording**  
   Use `save_sensor_data.py` to capture and save synchronized data from the camera and LiDAR.

3. **Point Correspondence**  
   Execute `extract_points.py` to manually select corresponding points between camera and LiDAR.

4. **Extrinsic Calibration**  
   Run `calculate_calibration.py` to compute the transformation matrix between camera and LiDAR.

5. **LiDAR Projection**  
   Use `lidar_camera_projection.py` to project LiDAR points into the camera frame for visualization and analysis.

### Running Nodes
To execute a specific node, use the following command:
```bash
ros2 run ros2_camera_lidar_fusion <node_name>
```
For example:
```bash
ros2 run ros2_camera_lidar_fusion camera_calibration.py
```

---

## License
This project is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for details.

---

## TODO
- [ ] Add functionality to streamline intrinsic calibration for the camera.
- [ ] Enable support for configuration files.
- [ ] Create a unified flow to execute all calibration steps in a single node.

---

Contributions and feedback are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
