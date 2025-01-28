#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
from rclpy.node import Node
import rclpy

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class CameraLidarExtrinsicNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_extrinsic_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.corr_file = config_file['general']['correspondence_file']                          # 라이다, 카메라 좌표 쌍 파일 경로
        self.corr_file = f'/ros2_ws/src/ros2_camera_lidar_fusion/data/{self.corr_file}'         # 파일 경로 완성
        self.camera_yaml = config_file['general']['camera_intrinsic_calibration']               # 카메라 내부 파라미터 파일 경로
        self.camera_yaml = f'/ros2_ws/src/ros2_camera_lidar_fusion/config/{self.camera_yaml}'
        self.output_dir = config_file['general']['config_folder']                               # 출력 파일 경로
        self.file = config_file['general']['camera_extrinsic_calibration']                      # 출력 파일 이름

        self.get_logger().info('Starting extrinsic calibration...')
        self.solve_extrinsic_with_pnp()    # 클래스 생성되자마자 바로 실행

    def load_camera_calibration(self, yaml_path: str):
        """Loads camera calibration parameters from a YAML file."""
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        mat_data = config['camera_matrix']['data']
        camera_matrix = np.array(mat_data, dtype=np.float64)
        dist_data = config['distortion_coefficients']['data']
        dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))

        return camera_matrix, dist_coeffs

    def solve_extrinsic_with_pnp(self):
        """Solves for extrinsic parameters using 2D-3D correspondences and camera calibration."""
        camera_matrix, dist_coeffs = self.load_camera_calibration(self.camera_yaml)
        self.get_logger().info(f"Camera matrix:\n{camera_matrix}")
        self.get_logger().info(f"Distortion coefficients: {dist_coeffs}")

        if not os.path.isfile(self.corr_file):
            raise FileNotFoundError(f"Correspondence file not found: {self.corr_file}")

        # 좌표 쌍 파일 읽기
        pts_2d = []
        pts_3d = []
        with open(self.corr_file, 'r') as f: # 파일 열기
            for line in f: # 파일 한 줄씩 읽기
                line = line.strip() # 줄 양쪽 공백 제거
                if not line or line.startswith('#'): # 줄이 비어있거나 주석인 경우 건너뜀
                    continue
                splitted = line.split(',')
                if len(splitted) != 5: # 줄이 5개의 쉼표로 구분되지 않은 경우 건너뜀
                    continue
                u, v, X, Y, Z = [float(val) for val in splitted] # 줄을 쉼표로 구분하여 분리하고 각 값을 실수로 변환
                pts_2d.append([u, v]) # 이미지 좌표 리스트에 추가
                pts_3d.append([X, Y, Z]) # 라이다 좌표 리스트에 추가

        pts_2d = np.array(pts_2d, dtype=np.float64) # 이미지 좌표 리스트를 numpy 배열로 변환
        pts_3d = np.array(pts_3d, dtype=np.float64) # 라이다 좌표 리스트를 numpy 배열로 변환

        num_points = len(pts_2d) # 좌표 쌍 개수
        self.get_logger().info(f"Loaded {num_points} correspondences from {self.corr_file}")

        if num_points < 4: # 좌표 쌍 개수가 4개 미만인 경우 예외 발생
            raise ValueError("At least 4 correspondences are required for solvePnP")

        success, rvec, tvec = cv2.solvePnP(
            pts_3d,
            pts_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise RuntimeError("solvePnP failed to find a solution.")

        self.get_logger().info("solvePnP succeeded.")
        self.get_logger().info(f"rvec: {rvec.ravel()}")
        self.get_logger().info(f"tvec: {tvec.ravel()}")

        R, _ = cv2.Rodrigues(rvec) # 회전 벡터를 3x3 회전 행렬로 변환

        T_lidar_to_cam = np.eye(4, dtype=np.float64) # LiDAR -> 카메라 변환 4x4 행렬
        T_lidar_to_cam[0:3, 0:3] = R
        T_lidar_to_cam[0:3, 3] = tvec[:, 0]

        self.get_logger().info(f"Transformation matrix (LiDAR -> Camera):\n{T_lidar_to_cam}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        out_yaml = os.path.join(self.output_dir, self.file)
        data_out = {
            "extrinsic_matrix": T_lidar_to_cam.tolist() # 출력 파일에 변환 행렬 저장
        }

        with open(out_yaml, 'w') as f:
            yaml.dump(data_out, f, sort_keys=False)

        self.get_logger().info(f"Extrinsic matrix saved to: {out_yaml}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarExtrinsicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()