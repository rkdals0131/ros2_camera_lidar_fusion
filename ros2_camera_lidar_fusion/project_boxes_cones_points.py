import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import torch

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from custom_interface.msg import ModifiedFloat32MultiArray
from std_msgs.msg import String

from ros2_camera_lidar_fusion.read_yaml import extract_configuration
from ultralytics.engine.results import Results, Boxes


def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    return T


def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs


def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> np.ndarray:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)
    field_names = [f.name for f in cloud_msg.fields]
    if not all(k in field_names for k in ('x','y','z')):
        return np.zeros((0,3), dtype=np.float32)
    x_field = next(f for f in cloud_msg.fields if f.name=='x')
    y_field = next(f for f in cloud_msg.fields if f.name=='y')
    z_field = next(f for f in cloud_msg.fields if f.name=='z')
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])
    raw_data = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points[:,0] = raw_data['x']
    points[:,1] = raw_data['y']
    points[:,2] = raw_data['z']
    if skip_rate > 1:
        points = points[::skip_rate]
    return points


class FusionBoxesProjectionNode(Node):  
    def __init__(self):
        super().__init__('fusion_boxes_projection_node')
        
        # z 좌표를 파라미터로 선언
        self.declare_parameter('cone_z_offset', -0.6)  # 기본값 -0.6m (콘이 라이다보다 아래에 있다고 가정)
        self.cone_z_offset = self.get_parameter('cone_z_offset').value
        self.get_logger().info(f"Using cone z offset: {self.cone_z_offset} meters")
        
        # Additional visualization parameters
        self.declare_parameter('box_width', 50)  # Default bounding box width
        self.declare_parameter('box_height', 100)  # Default bounding box height
        self.declare_parameter('draw_circles', True)  # Whether to draw circles for cones
        self.declare_parameter('draw_boxes', True)  # Whether to draw bounding boxes
        
        # Get parameters
        self.box_width = self.get_parameter('box_width').value
        self.box_height = self.get_parameter('box_height').value
        self.draw_circles = self.get_parameter('draw_circles').value
        self.draw_boxes = self.get_parameter('draw_boxes').value
        
        # 파라미터 변경 콜백 설정
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Color mapping for cone visualization
        self.color_mapping = {
            "Crimson Cone": (0, 0, 255),   # Red in BGR
            "Yellow Cone":  (0, 255, 255), # Yellow in BGR
            "Blue Cone":    (255, 0, 0),   # Blue in BGR
            "Unknown":      (0, 255, 0)    # Green in BGR (default)
        }
        
        # Class names for YOLO results mock
        self.class_names = {0: "Blue Cone", 1: "Crimson Cone", 2: "Yellow Cone"}
        
        config_file = extract_configuration() 
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_extrinsic_calibration'])
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        cones_topic = "/sorted_cones_time"  # cones 토픽

        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")

        # message_filters를 이용해 3개 토픽을 동기화
        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic, qos_profile=best_effort_qos)
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.cones_sub],
            queue_size=5,
            slop=0.07
        )
        self.ts.registerCallback(self.sync_callback)

        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()

        # Add a publisher for cone info string
        self._info_pub = self.create_publisher(String, "cone_info", 10)

        self.skip_rate = 1

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'cone_z_offset':
                self.cone_z_offset = param.value
                self.get_logger().info(f"Updated cone z offset to: {self.cone_z_offset} meters")
            elif param.name == 'box_width':
                self.box_width = param.value
                self.get_logger().info(f"Updated box width to: {self.box_width} pixels")
            elif param.name == 'box_height':
                self.box_height = param.value
                self.get_logger().info(f"Updated box height to: {self.box_height} pixels")
            elif param.name == 'draw_circles':
                self.draw_circles = param.value
                self.get_logger().info(f"Updated draw_circles to: {self.draw_circles}")
            elif param.name == 'draw_boxes':
                self.draw_boxes = param.value
                self.get_logger().info(f"Updated draw_boxes to: {self.draw_boxes}")
        return SetParametersResult(successful=True)

    def draw_detections(self, cv_image: np.ndarray, results: Results,
                      verified_labels: list = None) -> np.ndarray:
        """
        Draw detection results on the image with color verification.
        
        This function is adapted from YoloDebugNode's draw_detections method.

        Args:
            cv_image: OpenCV image (in BGR format)
            results: YOLO detection results
            verified_labels: List of verified color labels

        Returns:
            Image with drawn detections
        """
        # Create a copy of the image to draw on
        debug_image = cv_image.copy()

        # Draw bounding boxes for detected objects
        if results.boxes:
            for i, box in enumerate(results.boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class information
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Use verified labels if available
                if verified_labels and i < len(verified_labels):
                    cls_name = verified_labels[i]
                else:
                    cls_name = self.class_names[cls_id]

                # Get color from color mapping
                color = self.color_mapping.get(cls_name, (0, 255, 0))  # Default: green

                # Draw rectangle
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)

                # Add label text
                label = f"{cls_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(debug_image, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 10), color, -1)
                cv2.putText(debug_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return debug_image

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2, cones_msg: ModifiedFloat32MultiArray):
        try:
            if not image_msg.data:
                self.get_logger().warn("Received empty image message, skipping processing")
                return
                
            # 1. 이미지 메시지를 OpenCV 이미지로 변환
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
                if image_msg.encoding != 'bgr8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")
                return

            # 2. 포인트 클라우드 메시지(LiDAR)를 XYZ 배열로 변환
            xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
            n_points = xyz_lidar.shape[0]
            
            if n_points > 0:  # LiDAR 데이터가 있는 경우 처리
                # 3. 포인트 클라우드 데이터 준비: 동차 좌표 변환
                xyz_lidar_f64 = xyz_lidar.astype(np.float64)
                ones = np.ones((n_points, 1), dtype=np.float64)
                xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))  # 동차 좌표 (N, 4)

                # 4. LiDAR 좌표계를 카메라 좌표계로 변환
                xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
                xyz_cam = xyz_cam_h[:, :3]  # 동차 좌표에서 3D 좌표 추출

                # 5. 카메라 앞에 있는 포인트 필터링
                mask_in_front = (xyz_cam[:, 2] > 0.0)  # z > 0: 카메라 앞에 위치한 포인트
                xyz_cam_front = xyz_cam[mask_in_front]
                
                n_front = xyz_cam_front.shape[0]
                if n_front > 0:
                    # 6. 3D 포인트를 2D 이미지로 투영
                    rvec = np.zeros((3,1), dtype=np.float64)
                    tvec = np.zeros((3,1), dtype=np.float64)
                    lidar_image_points, _ = cv2.projectPoints(
                        xyz_cam_front,
                        rvec, tvec,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    lidar_image_points = lidar_image_points.reshape(-1, 2)

                    # 7. 투영된 LiDAR 포인트를 이미지 위에 시각화 (초록색 점으로)
                    h, w = cv_image.shape[:2]
                    for (u, v) in lidar_image_points:
                        u_int = int(round(u))
                        v_int = int(round(v))
                        if 0 <= u_int < w and 0 <= v_int < h:
                            cv2.circle(cv_image, (u_int, v_int), 2, (0, 255, 0), -1)  # 초록색 원
                else:
                    self.get_logger().info("No LiDAR points in front of camera (z>0).")

            # 8. cones 메시지 처리
            cone_data = np.array(cones_msg.data, dtype=np.float32)
            if cone_data.size == 0:
                self.get_logger().warn("Empty cones data.")
            else:
                # 레이아웃에서 포인트 수 확인
                num_points = cones_msg.layout.dim[0].size
                if num_points * 2 != cone_data.size:
                    self.get_logger().error(f"Cone data size ({cone_data.size}) does not match layout dimensions ({num_points}*2).")
                else:
                    cones_xy = cone_data.reshape(num_points, 2)  # (N,2) 배열
                    
                    # z 좌표를 파라미터 값으로 설정 (0 대신 self.cone_z_offset 사용)
                    cones_xyz = np.hstack((cones_xy, np.ones((num_points, 1), dtype=np.float32) * self.cone_z_offset))
                    
                    # 9. cone 포인트를 이미지 평면으로 투영
                    # 먼저 cone 좌표를 라이다 좌표계에서 카메라 좌표계로 변환
                    cones_xyz_h = np.hstack((cones_xyz, np.ones((cones_xyz.shape[0], 1), dtype=np.float32)))  # 동차 좌표
                    cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T  # 카메라 좌표계로 변환
                    cones_cam = cones_cam_h[:, :3]  # 동차 좌표에서 3D 좌표 추출
                    
                    # 카메라 앞에 있는 cone 포인트만 필터링
                    mask_cones_front = (cones_cam[:, 2] > 0.0)
                    cones_cam_front = cones_cam[mask_cones_front]
                    
                    if cones_cam_front.shape[0] > 0:
                        # 투영
                        rvec = np.zeros((3,1), dtype=np.float64)
                        tvec = np.zeros((3,1), dtype=np.float64)
                        cone_image_points, _ = cv2.projectPoints(
                            cones_cam_front.astype(np.float64),
                            rvec, tvec,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                        cone_image_points = cone_image_points.reshape(-1, 2)
                        
                        # 10. 투영된 cone 포인트를 이미지에 시각화
                        h, w = cv_image.shape[:2]
                        detection_info = []
                        
                        # Draw circles if enabled
                        if self.draw_circles:
                            for (u, v) in cone_image_points:
                                u_int = int(round(u))
                                v_int = int(round(v))
                                if 0 <= u_int < w and 0 <= v_int < h:
                                    cv2.circle(cv_image, (u_int, v_int), 4, (0, 0, 255), -1)  # 빨간색 원
                                    cv2.circle(cv_image, (u_int, v_int), 6, (255, 255, 255), 1)  # 흰색 테두리
                        
                        # Draw bounding boxes if enabled
                        if self.draw_boxes:
                            # Create mock YOLO results for bounding box drawing
                            mock_results = Results(
                                orig_img=cv_image,
                                path="",
                                names=self.class_names
                            )
                            
                            # Create bounding boxes for detected cones
                            boxes_list = []
                            for i, (u, v) in enumerate(cone_image_points):
                                u_int = int(round(u))
                                v_int = int(round(v))
                                if 0 <= u_int < w and 0 <= v_int < h:
                                    # For each cone, create a bounding box (x1, y1, x2, y2)
                                    x1 = max(0, u_int - self.box_width // 2)
                                    y1 = max(0, v_int - self.box_height)
                                    x2 = min(w, x1 + self.box_width)
                                    y2 = min(h, v_int)
                                    
                                    # Add bounding box - default to yellow cones (class 2)
                                    boxes_list.append([x1, y1, x2, y2, 0.95, 2])  # [x1, y1, x2, y2, conf, cls]
                                    
                                    # Add to detection info
                                    cone_type = "Yellow Cone"  # Default, can be improved with color detection
                                    detection_info.append(f"{cone_type}: ({u_int}, {v_int})")
                            
                            if boxes_list:
                                # Convert to tensor
                                boxes_tensor = torch.tensor(boxes_list)
                                # Create Boxes object and assign to results
                                mock_results.boxes = Boxes(boxes_tensor, mock_results.orig_shape)
                                
                                # Use our draw_detections method
                                cv_image = self.draw_detections(cv_image, mock_results)
                        
                        # Publish cone info
                        if detection_info:
                            info_msg = String()
                            info_msg.data = "; ".join(detection_info)
                            self._info_pub.publish(info_msg)

            # 11. 결과 이미지를 퍼블리시
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = FusionBoxesProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
