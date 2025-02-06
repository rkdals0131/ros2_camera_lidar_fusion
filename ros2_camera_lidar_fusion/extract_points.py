#!/usr/bin/env python3

import os
import cv2
import open3d as o3d
import numpy as np
from rclpy.node import Node
import rclpy

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class ImageCloudCorrespondenceNode(Node):  
    def __init__(self):
        super().__init__('image_cloud_correspondence_node') 

        config_file = extract_configuration()  
        if config_file is None:  
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        # 데이터 디렉토리 및 파일 이름 설정
        self.data_dir = config_file['general']['data_folder']
        self.file = config_file['general']['correspondence_file']

        # 데이터 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(self.data_dir):
            self.get_logger().warn(f"Data directory '{self.data_dir}' does not exist.")
            os.makedirs(self.data_dir)

        self.get_logger().info(f"Looking for .png and .pcd file pairs in '{self.data_dir}'")
        self.process_file_pairs()  # 파일 쌍 처리 시작

    def get_file_pairs(self, directory): # 이미지와 .pcd 파일 쌍 배열 반환
        files = os.listdir(directory)  # 디렉토리 내 파일 목록 가져오기
        pairs_dict = {}  # 파일 쌍을 저장할 딕셔너리
        for f in files:
            full_path = os.path.join(directory, f)  # 파일의 전체 경로
            if not os.path.isfile(full_path):  # 파일이 아니면 건너뜀
                continue
            name, ext = os.path.splitext(f)  # 파일명과 확장자 분리

            # 이미지 및 포인트 클라우드 파일만 처리
            if ext.lower() in [".png", ".jpg", ".jpeg", ".pcd"]:
                if name not in pairs_dict:
                    pairs_dict[name] = {}
                if ext.lower() == ".png":
                    pairs_dict[name]['png'] = full_path
                elif ext.lower() == ".pcd":
                    pairs_dict[name]['pcd'] = full_path

        file_pairs = []  # 유효한 파일 쌍 리스트
        for prefix, d in pairs_dict.items():
            if 'png' in d and 'pcd' in d:  # 이미지와 포인트 클라우드 쌍이 모두 존재할 때
                file_pairs.append((prefix, d['png'], d['pcd']))

        file_pairs.sort()  # 파일 쌍 정렬
        return file_pairs

    def pick_image_points(self, image_path): # 선택한 이미지 픽셀 좌표 배열 반환 
        img = cv2.imread(image_path)  # 이미지 파일 읽기
        if img is None:  # 이미지 로드 실패 시
            self.get_logger().error(f"Error loading image: {image_path}")
            return []

        points_2d = []  # 선택된 2D 이미지 포인트 저장할 배열
        window_name = "Select points on the image (press 'q' or ESC to finish)"

        def mouse_callback(event, x, y, flags, param):  # 마우스 클릭 콜백 함수
            if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 시
                points_2d.append((x, y))  # 클릭한 좌표 추가
                self.get_logger().info(f"Image: click at ({x}, {y})")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 윈도우 생성
        cv2.setMouseCallback(window_name, mouse_callback)  # 마우스 콜백 설정

        while True:
            display_img = img.copy()  # 이미지 복사
            for pt in points_2d:  # 선택된 포인트 그리기
                cv2.circle(display_img, pt, 5, (0, 0, 255), -1)

            cv2.imshow(window_name, display_img)  # 이미지 표시
            key = cv2.waitKey(10)  # 키 입력 대기
            if key == 27 or key == ord('q'):  # ESC 또는 'q' 입력 시 종료
                break

        cv2.destroyWindow(window_name)  # 윈도우 닫기
        return points_2d

    def pick_cloud_points(self, pcd_path): # 선택한 포인트 클라우드 좌표 배열 반환
        pcd = o3d.io.read_point_cloud(pcd_path)  # 포인트 클라우드 파일 읽어서 o3d PC 객체 생성
        if pcd.is_empty():  # 포인트 클라우드가 비어있을 경우
            self.get_logger().error(f"Empty or invalid point cloud: {pcd_path}")
            return []

        # Open3D 포인트 선택 안내 메시지
        self.get_logger().info("\n[Open3D Instructions]")
        self.get_logger().info("  - Shift + left click to select a point")
        self.get_logger().info("  - Press 'q' or ESC to close the window when finished\n")

        vis = o3d.visualization.VisualizerWithEditing()  # 포인트 선택을 위한 시각화 도구
        vis.create_window(window_name="Select points on the cloud", width=1280, height=720)
        vis.add_geometry(pcd)  # 포인트 클라우드 추가

        render_opt = vis.get_render_option()  # 렌더링 옵션 설정
        render_opt.point_size = 2.0  # 포인트 크기 설정


        vis.run()  # 시각화 실행
        vis.destroy_window()  # 윈도우 닫기
        picked_indices = vis.get_picked_points()  # 사용자가 선택한 포인트의 인덱스를 리스트로 반환 

        # pcd.points는 PC의 점 데이터를 o3d 포맷으로 반환
        # np.asarray(pcd.points)는 o3d 포맷을 numpy 배열로 변환 -> (N, 3) 크기의 배열, 각 행은 (x, y, z) 좌표를 나타냄
        np_points = np.asarray(pcd.points)  # 포인트 클라우드를 numpy 배열로 변환

        picked_xyz = []  # 선택된 포인트의 좌표 저장할 리스트
        for idx in picked_indices: # 선택된 포인트의 인덱스동안
            xyz = np_points[idx]  # 인덱스에 해당하는 좌표
            picked_xyz.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))  # 좌표 추가
            self.get_logger().info(f"Cloud: index={idx}, coords=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")

        return picked_xyz

    def process_file_pairs(self): # 파일 쌍 처리
        file_pairs = self.get_file_pairs(self.data_dir)  # 파일 쌍 가져오기
        if not file_pairs:  # 파일 쌍이 없을 경우
            self.get_logger().error(f"No .png / .pcd pairs found in '{self.data_dir}'")
            return

        self.get_logger().info("Found the following pairs:")  # 발견된 파일 쌍 출력
        for prefix, png_path, pcd_path in file_pairs:
            self.get_logger().info(f"  {prefix} -> {png_path}, {pcd_path}")


        # 출력 파일 경로 설정
        out_txt = os.path.join(self.data_dir, self.file)
        # 기존 파일 내용을 초기화하고 헤더를 작성합니다.
        with open(out_txt, 'w') as f:
            f.write("# u, v, x, y, z\n")

        # 각 파일 쌍에 대해 대응점을 선택하고, 결과를 파일에 이어서 기록합니다.
        for prefix, png_path, pcd_path in file_pairs:

            self.get_logger().info("\n========================================")
            self.get_logger().info(f"Processing pair: {prefix}")
            self.get_logger().info(f"Image: {png_path}")
            self.get_logger().info(f"Point Cloud: {pcd_path}")
            self.get_logger().info("========================================\n")

            image_points = self.pick_image_points(png_path)  # 이미지 포인트 선택
            self.get_logger().info(f"\nSelected {len(image_points)} points in the image.\n")

            cloud_points = self.pick_cloud_points(pcd_path)  # 포인트 클라우드 포인트 선택
            self.get_logger().info(f"\nSelected {len(cloud_points)} points in the cloud.\n")


            min_len = min(len(image_points), len(cloud_points))
            # append 모드('a')로 파일을 열어 대응점을 이어서 기록합니다.
            with open(out_txt, 'a') as f:
                for i in range(min_len):
                    (u, v) = image_points[i]
                    (x, y, z) = cloud_points[i]
                    f.write(f"{u},{v},{x},{y},{z}\n")


            self.get_logger().info(f"Appended {min_len} correspondences for pair '{prefix}' to: {out_txt}")
            self.get_logger().info("========================================")

        self.get_logger().info("\nProcessing complete! Correspondences saved for all pairs.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCloudCorrespondenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()