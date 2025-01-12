import os
import cv2
import open3d as o3d
import numpy as np

DATA_DIR = "/ros2_ws/src/ros2_pcl_segmentation/pcl_camera_lidar_fusion/data"

def get_file_pairs(directory):
    files = os.listdir(directory)
    pairs_dict = {}
    for f in files:
        full_path = os.path.join(directory, f)
        if not os.path.isfile(full_path):
            continue
        name, ext = os.path.splitext(f)

        if ext.lower() in [".png", ".jpg", ".jpeg", ".pcd"]:
            if name not in pairs_dict:
                pairs_dict[name] = {}
            if ext.lower() == ".png":
                pairs_dict[name]['png'] = full_path
            elif ext.lower() == ".pcd":
                pairs_dict[name]['pcd'] = full_path

    file_pairs = []
    for prefix, d in pairs_dict.items():
        if 'png' in d and 'pcd' in d:
            file_pairs.append((prefix, d['png'], d['pcd']))

    file_pairs.sort()
    return file_pairs


def pick_image_points(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[pick_image_points] Error cargando imagen: {image_path}")
        return []

    points_2d = []
    window_name = "Selecciona puntos en la imagen (pulsa 'q' o ESC para terminar)"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_2d.append((x, y))
            print(f"Imagen: clic en ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_img = img.copy()
        for pt in points_2d:
            cv2.circle(display_img, pt, 5, (0, 0, 255), -1)

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return points_2d


def pick_cloud_points(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        print(f"[pick_cloud_points] Nube de puntos vacía o error leyendo {pcd_path}")
        return []

    print("\n[Open3D] Instrucciones:")
    print("  - Shift + clic izquierdo para seleccionar un punto")
    print("  - Pulsa Q o ESC para cerrar la ventana cuando termines\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Selecciona puntos en la nube", width=1280, height=720)
    vis.add_geometry(pcd)
    
    render_opt = vis.get_render_option()
    render_opt.point_size = 1.0 

    vis.run()
    vis.destroy_window()
    picked_indices = vis.get_picked_points()

    np_points = np.asarray(pcd.points)
    picked_xyz = []
    for idx in picked_indices:
        xyz = np_points[idx]
        picked_xyz.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        print(f"Nube: índice={idx}, coords=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")

    return picked_xyz


def main():
    file_pairs = get_file_pairs(DATA_DIR)
    if not file_pairs:
        print(f"No se encontraron pares .png / .pcd en {DATA_DIR}")
        return

    print("Se encontraron los siguientes pares:")
    for prefix, png_path, pcd_path in file_pairs:
        print(f"  {prefix} -> {png_path}, {pcd_path}")

    for prefix, png_path, pcd_path in file_pairs:
        print("\n========================================")
        print(f"Procesando par: {prefix}")
        print("Imagen:", png_path)
        print("Nube  :", pcd_path)
        print("========================================\n")

        image_points = pick_image_points(png_path)
        print(f"\nSe seleccionaron {len(image_points)} puntos en la imagen.\n")

        cloud_points = pick_cloud_points(pcd_path)
        print(f"\nSe seleccionaron {len(cloud_points)} puntos en la nube.\n")

        out_txt = os.path.join(DATA_DIR, f"{prefix}_correspondences.txt")
        with open(out_txt, 'w') as f:
            f.write("# u, v, x, y, z\n")
            min_len = min(len(image_points), len(cloud_points))
            for i in range(min_len):
                (u, v) = image_points[i]
                (x, y, z) = cloud_points[i]
                f.write(f"{u},{v},{x},{y},{z}\n")

        print(f"Guardadas {min_len} correspondencias en: {out_txt}")
        print("========================================")

    print("\nFinalizado. ¡Ya tienes las correspondencias para cada par de archivos!")


if __name__ == "__main__":
    main()
