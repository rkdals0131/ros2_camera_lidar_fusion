#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2

def load_camera_calibration(yaml_path: str):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No se encontró el archivo de calibración: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    mat_data = config['camera_matrix']['data']
    camera_matrix = np.array(mat_data, dtype=np.float64)
    dist_data = config['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))

    return camera_matrix, dist_coeffs


def solve_extrinsic_with_pnp(
    corr_file: str,
    camera_yaml: str,
    output_dir: str = "/ros2_ws/src/ros2_pcl_segmentation/pcl_camera_lidar_fusion/config"
):
    camera_matrix, dist_coeffs = load_camera_calibration(camera_yaml)
    print("[INFO] Camera matrix:\n", camera_matrix)
    print("[INFO] Distortion coefficients:", dist_coeffs)
    if not os.path.isfile(corr_file):
        raise FileNotFoundError(f"No existe el archivo de correspondencias: {corr_file}")

    pts_2d = []
    pts_3d = []
    with open(corr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            splitted = line.split(',')
            if len(splitted) != 5:
                continue
            u, v, X, Y, Z = [float(val) for val in splitted]
            pts_2d.append([u, v])
            pts_3d.append([X, Y, Z])

    pts_2d = np.array(pts_2d, dtype=np.float64)
    pts_3d = np.array(pts_3d, dtype=np.float64)

    num_points = len(pts_2d)
    print(f"[INFO] Se leyeron {num_points} correspondencias desde {corr_file}.")

    if num_points < 4:
        raise ValueError("Necesitas al menos 4 correspondencias para solvePnP")

    success, rvec, tvec = cv2.solvePnP(
        pts_3d,
        pts_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP falló en encontrar una solución.")

    print("[INFO] solvePnP tuvo éxito.")
    print("[INFO] rvec =", rvec.ravel())
    print("[INFO] tvec =", tvec.ravel())

    R, _ = cv2.Rodrigues(rvec)

    T_lidar_to_cam = np.eye(4, dtype=np.float64)
    T_lidar_to_cam[0:3, 0:3] = R
    T_lidar_to_cam[0:3, 3] = tvec[:, 0]

    print("\n[INFO] Matriz de transformación (LiDAR -> Camera) estimada:\n", T_lidar_to_cam)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_yaml = os.path.join(output_dir, "camera_lidar_extrinsic.yaml")
    data_out = {
        "extrinsic_matrix": T_lidar_to_cam.tolist()
    }

    with open(out_yaml, 'w') as f:
        yaml.dump(data_out, f, sort_keys=False)

    print(f"\n[INFO] Matriz extrínseca guardada en: {out_yaml}")


def main():
    corr_file = "/ros2_ws/src/ros2_pcl_segmentation/pcl_camera_lidar_fusion/data/20250112_005752_correspondences.txt"
    camera_yaml = "/ros2_ws/src/ros2_pcl_segmentation/pcl_human_segmentation/config/camera_calibration.yaml"
    output_dir = "/ros2_ws/src/ros2_pcl_segmentation/pcl_camera_lidar_fusion/config"
    solve_extrinsic_with_pnp(corr_file, camera_yaml, output_dir)

if __name__ == "__main__":
    main()
