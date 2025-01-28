#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from datetime import datetime

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')

        config_file = extract_configuration() # 설정 파일 로드

        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        # 체스판 정보
        self.chessboard_rows = config_file['chessboard']['pattern_size']['rows']
        self.chessboard_cols = config_file['chessboard']['pattern_size']['columns']
        self.square_size = config_file['chessboard']['square_size_meters']

        # 이미지 정보 <- general_configuration.yaml에서 로드
        self.image_topic = config_file['camera']['image_topic']
        self.image_width = config_file['camera']['image_size']['width']
        self.image_height = config_file['camera']['image_size']['height']

        # 출력 경로
        self.output_path = config_file['general']['config_folder']
        self.file = config_file['general']['camera_intrinsic_calibration']

        # 이미지 구독
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.bridge = CvBridge() # ROS 이미지를 OpenCV 이미지로 변환

        self.obj_points = [] # 현실 좌표(체커보드)
        self.img_points = [] # 픽셀 좌표

        # 체스판 3D 점 초기화
        self.objp = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32) # 체스판 칸 수마다 3차원 좌표 있음. 
        self.objp[:, :2] = np.mgrid[0:self.chessboard_cols, 0:self.chessboard_rows].T.reshape(-1, 2) # z 정보는 건드리지 않고 0 유지
        # 각 열이 0~self.chessboard_cols, 각 행이 0~self.chessboard_rows인 그리드를 각각 생성, (2, rows, cols) -> (rows, cols, 2)로 전치
        # reshape로 2차원 배열을 [(0,0), (1,0), ... (5,4), (6,4)] 로 변환
        self.objp *= self.square_size # self.objp의 모든 x, y 좌표를 square_size로 스케일링

        self.get_logger().info("Camera calibration node initialized. Waiting for images...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # 반환값은 변환된 OpenCV 이미지
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) # 이미지를 그레이스케일 이미지1로 변환

            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_cols, self.chessboard_rows), None) # 체스판 코너 검출
            # ret은 코너 검출 성공 여부, corners는 코너 2D 좌표 배열

            if ret:
                self.obj_points.append(self.objp) # 초기화된 체스보드 현실 3D 좌표 추가
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), # 원본 그레이스케일 이미지1, 코너 좌표, 검색 윈도우 크기, 검색 영역 데드존(없음)
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 코너 검출 종료 조건
                )
                self.img_points.append(refined_corners) # 2D 코너 좌표 추가

                cv2.drawChessboardCorners(cv_image, (self.chessboard_cols, self.chessboard_rows), refined_corners, ret) # 체스판 코너 그리기
                self.get_logger().info("Chessboard detected and points added.")
            else:
                self.get_logger().warn("Chessboard not detected in image.")

            cv2.imshow("Image", cv_image) # 이미지 표시
            cv2.waitKey(1) # 키보드 입력 대기, 1ms 대기 후 다음 프레임 처리

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def save_calibration(self):
        if len(self.obj_points) < 10:  # 최소 10장 이상 캘리브레이션 이미지
            self.get_logger().error("Not enough images for calibration. At least 10 are required.")
            return

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera( # 카메라 보정 수행
            # 입력은 3D 좌표 리스트, 각 이미지에서 감지된 체스보드 2D 좌표 리스트, 이미지 크기, 초기 카메라 행렬과 왜곡 계수는 전달하지 않음
            self.obj_points, self.img_points, (self.image_width, self.image_height), None, None
        )

        calibration_data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'camera_matrix': {
                'rows': 3,
                'columns': 3,
                'data': camera_matrix.tolist()
            },
            'distortion_coefficients': {
                'rows': 1,
                'columns': len(dist_coeffs[0]),
                'data': dist_coeffs[0].tolist()
            },
            'chessboard': {
                'pattern_size': {
                    'rows': self.chessboard_rows,
                    'columns': self.chessboard_cols
                },
                'square_size_meters': self.square_size
            },
            'image_size': {
                'width': 640,
                'height': 480
            },
            'rms_reprojection_error': ret   # RMS 재투영 오차(보정 정확도)
        }

        output_file = f"{self.output_path}/{self.file}" # 보정 데이터를 YAML 파일로 저장
        try:
            with open(output_file, 'w') as file:
                yaml.dump(calibration_data, file)
            self.get_logger().info(f"Calibration saved to {output_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:                                           # 돌다가 키보드로 중지하면 캘리브레이션 끝
        node.save_calibration()
        node.get_logger().info("Calibration process completed.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()