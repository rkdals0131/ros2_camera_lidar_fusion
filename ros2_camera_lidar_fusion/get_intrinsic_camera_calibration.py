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


        self.obj_points = []
        self.img_points = []
        self.latest_image = None  # 최신 이미지를 저장할 변수

        # 체스보드 코너의 3D 좌표 생성 (평면상의 좌표)
        self.objp = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_cols, 0:self.chessboard_rows].T.reshape(-1, 2)
        self.objp *= self.square_size


        # 1초에 한 번 실행되는 타이머 설정
        self.timer = self.create_timer(0.3, self.process_latest_image)

        self.get_logger().info("Camera calibration node initialized. Waiting for images...")

    def image_callback(self, msg):
        """이미지 콜백: 최신 이미지만 저장하고, 처리 로직은 타이머에서 실행"""
        try:
            # Check if image message has data
            if not msg.data:
                self.get_logger().warn("Received empty image message, skipping processing")
                return
                
            # First try with passthrough to preserve original encoding
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Then convert to BGR if necessary
            if msg.encoding != 'bgr8':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            self.latest_image = image
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e} (encoding: {msg.encoding})")

    def process_latest_image(self):
        """타이머에서 호출: 최신 이미지를 처리"""
        if self.latest_image is None:
            self.get_logger().warn("아직 수신된 이미지가 없습니다.")
            return

        # Make a copy to avoid modifying the original
        display_image = self.latest_image.copy()
        
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)

        # 체스보드 패턴 검출
        ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_cols, self.chessboard_rows), None)

        if ret:
            self.obj_points.append(self.objp.copy())
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            self.img_points.append(refined_corners)

            cv2.drawChessboardCorners(display_image, (self.chessboard_cols, self.chessboard_rows), refined_corners, ret)
            self.get_logger().info("체스보드가 감지되어 점들이 추가되었습니다.")
        else:
            self.get_logger().warn("체스보드가 이미지에서 감지되지 않았습니다.")

        # 캡처된 이미지 수를 이미지에 표시
        cv2.putText(display_image, f"Captured Images: {len(self.obj_points)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.save_calibration()
            self.get_logger().info("캘리브레이션 저장 후 노드를 종료합니다.")
            rclpy.shutdown()

    def save_calibration(self):
        if len(self.obj_points) < 10:  # 최소 10장 이상 캘리브레이션 이미지
            self.get_logger().error("Not enough images for calibration. At least 10 are required.")
            return

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera( # 카메라 보정 수행
            # 입력인자는 3D 좌표 리스트, 각 이미지에서 감지된 체스보드 2D 좌표 리스트, 이미지 크기, 초기 카메라 행렬과 왜곡 계수는 전달하지 않음
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
                'width': self.image_width,
                'height': self.image_height
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
        node.get_logger().info("캘리브레이션 과정 완료.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()