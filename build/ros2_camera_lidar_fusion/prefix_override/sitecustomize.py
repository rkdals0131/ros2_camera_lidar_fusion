import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/user1/ros2_fusion_ws/src/ros2_camera_lidar_fusion/install/ros2_camera_lidar_fusion'
