#!/usr/bin/env python3

import rclpy, os, cv2, datetime
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import threading

class SaveData(Node):
    def __init__(self):
        super().__init__('save_data_node')
        self.get_logger().info('Save data node has been started')

        self.declare_parameter('max_file_saved', 10)
        self.declare_parameter('storage_path', '/ros2_ws/src/ros2_camera_lidar_fusion/data')
        self.declare_parameter('camera_topic', '/pcl_human_segmentation/camera/raw_image')
        self.declare_parameter('lidar_topic', '/rslidar_points')
        self.declare_parameter('keyboard_listener', True)
        self.declare_parameter('slop', 0.1)

        self.max_file_saved = self.get_parameter('max_file_saved').get_parameter_value().integer_value
        self.storage_path = self.get_parameter('storage_path').get_parameter_value().string_value
        self.keyboard_listener_enabled = self.get_parameter('keyboard_listener').get_parameter_value().bool_value

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.get_logger().warn(f'Data will be saved at {self.storage_path}')

        self.image_sub = Subscriber(
            self,
            Image,
            self.get_parameter('camera_topic').get_parameter_value().string_value
        )
        self.pointcloud_sub = Subscriber(
            self,
            PointCloud2,
            self.get_parameter('lidar_topic').get_parameter_value().string_value
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub],
            queue_size=10,
            slop=self.get_parameter('slop').get_parameter_value().double_value
        )
        self.ts.registerCallback(self.synchronize_data)

        self.save_data_flag = not self.keyboard_listener_enabled
        if self.keyboard_listener_enabled:
            self.start_keyboard_listener()

    def start_keyboard_listener(self):
        """Starts a thread to listen for keyboard events."""
        def listen_for_space():
            while True:
                key = input("Press 'Enter' to save data (keyboard listener enabled): ")
                if key.strip() == '':
                    self.save_data_flag = True
                    self.get_logger().info('Space key pressed, ready to save data')
        thread = threading.Thread(target=listen_for_space, daemon=True)
        thread.start()

    def synchronize_data(self, image_msg, pointcloud_msg):
        """Handles synchronized messages and saves data if the flag is set."""
        if self.save_data_flag:
            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.get_logger().info(f'Synchronizing data at {file_name}')
            total_files = len(os.listdir(self.storage_path))
            if total_files < self.max_file_saved:
                self.save_data(image_msg, pointcloud_msg, file_name)
                if self.keyboard_listener_enabled:
                    self.save_data_flag = False

    def pointcloud2_to_open3d(self, pointcloud_msg):
        """Converts a PointCloud2 message to an Open3D point cloud."""
        points = []
        for p in point_cloud2.read_points(pointcloud_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
        return pointcloud

    def save_data(self, image_msg, pointcloud_msg, file_name):
        """Saves image and point cloud data to the storage path."""
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        pointcloud = self.pointcloud2_to_open3d(pointcloud_msg)
        o3d.io.write_point_cloud(f'{self.storage_path}/{file_name}.pcd', pointcloud)
        cv2.imwrite(f'{self.storage_path}/{file_name}.png', image)
        self.get_logger().info(f'Data has been saved at {self.storage_path}/{file_name}.png')


def main(args=None):
    rclpy.init(args=args)
    node = SaveData()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
