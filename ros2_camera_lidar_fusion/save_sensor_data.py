#!/usr/bin/env python3

import rclpy, os, cv2, datetime
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import threading

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class SaveData(Node):
    def __init__(self):
        super().__init__('save_data_node')
        self.get_logger().info('Save data node has been started')

        config_file = extract_configuration() # 함수에서 반환된 컨픽 딕셔너리 로드
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        # QoS Policy 수정
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.max_file_saved = config_file['general']['max_file_saved']
        self.storage_path = config_file['general']['data_folder']
        self.image_topic = config_file['camera']['image_topic']
        self.lidar_topic = config_file['lidar']['lidar_topic']
        self.keyboard_listener_enabled = config_file['general']['keyboard_listener']
        self.slop = config_file['general']['slop']

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.get_logger().warn(f'Data will be saved at {self.storage_path}') # 데이터 저장 경로 출력

        self.image_sub = Subscriber( # 지정된 이미지 토픽 구독
            self,
            Image,
            self.image_topic
        )
        self.pointcloud_sub = Subscriber( # 지정된 라이다 포인트 토픽 구독
            self,
            PointCloud2,
            self.lidar_topic,
            qos_profile=best_effort_qos
        )

        self.ts = ApproximateTimeSynchronizer( # 이미지와 라이다 포인트 토픽 동기화
            [self.image_sub, self.pointcloud_sub],
            queue_size=10,
            slop=self.slop
        )
        self.ts.registerCallback(self.synchronize_data)

        self.save_data_flag = not self.keyboard_listener_enabled # 키보드 리스너 활성 여부에 따라 데이터 저장 플래그 초기화
        if self.keyboard_listener_enabled: # 키보드 입력으로 데이터 저장을 제어하는 리스너를 비동기로 실행
            self.start_keyboard_listener()

    # 키보드 리스너
    def start_keyboard_listener(self):
        """Starts a thread to listen for keyboard events."""
        def listen_for_space():
            while True:
                key = input("Press 'Enter' to save data (keyboard listener enabled): ")
                if key.strip() == '':
                    self.save_data_flag = True
                    self.get_logger().info('Space key pressed, ready to save data')
        thread = threading.Thread(target=listen_for_space, daemon=True) # 키보드 입력을 별도 스레드에서 처리, 노드 메시지 처리와 병행
        thread.start()

    # 데이터 동기화 처리
    def synchronize_data(self, image_msg, pointcloud_msg):
        """Handles synchronized messages and saves data if the flag is set."""
        if self.save_data_flag: # 데이터 저장 여부 제어, 키보드 리스너 사용시 저장 후 플래그 비활성화
            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.get_logger().info(f'Synchronizing data at {file_name}')
            total_files = len(os.listdir(self.storage_path)) # 저장 경로에 있는 파일 수
            if total_files < self.max_file_saved: # 최대 저장 파일 수보다 저장된 파일 수가 적으면
                self.save_data(image_msg, pointcloud_msg, file_name) # 데이터 저장
                if self.keyboard_listener_enabled: # 키보드 리스너 사용시 저장 후 플래그 비활성화
                    self.save_data_flag = False

    # ROS의 PointCloud2 메시지를 Open3D 포인트 클라우드로 변환
    def pointcloud2_to_open3d(self, pointcloud_msg):
        """Converts a PointCloud2 message to an Open3D point cloud."""
        points = []
        for p in point_cloud2.read_points(pointcloud_msg, skip_nans=True): # 메시지에서 점 데이터를 읽음
            points.append([p[0], p[1], p[2]])
        pointcloud = o3d.geometry.PointCloud() # Open3D 포인트 클라우드 객체 생성
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32)) # 점 데이터를 벡터로 변환
        return pointcloud

    # 데이터 저장
    def save_data(self, image_msg, pointcloud_msg, file_name):
        """Saves image and point cloud data to the storage path."""
        bridge = CvBridge()                                 # 이미지 메시지를 
        image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')     # OpenCV 이미지로 변환
        pointcloud = self.pointcloud2_to_open3d(pointcloud_msg) # 라이다 포인트 토픽을 Open3D 포인트 클라우드로 변환
        o3d.io.write_point_cloud(f'{self.storage_path}/{file_name}.pcd', pointcloud) # 포인트 클라우드 데이터 저장
        cv2.imwrite(f'{self.storage_path}/{file_name}.png', image) # 이미지 데이터 저장
        self.get_logger().info(f'Data has been saved at {self.storage_path}/{file_name}.png') # 데이터 저장 경로 출력


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