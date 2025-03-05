from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package path
    pkg_share = get_package_share_directory('ros2_camera_lidar_fusion')

    # Path to RViz config file
    default_rviz_config = os.path.join(pkg_share, 'config', 'calib_rviz_config.rviz')

    # Declare Launch Arguments
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='/home/user1/ros2_fusion_ws/src/ros2_camera_lidar_fusion/config/calib_rviz_config.rviz'
    )

    # RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen'
    )

    # Launch Description
    return LaunchDescription([
        rviz_config_arg,
        rviz_node
    ])
