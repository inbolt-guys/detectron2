import os
import cv2
import glob
import numpy as np
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor

def get_normal_vector_angles_from_depthmap(depth, imsize = None, camera_matrix=None):
    # Estimate camera matrix if not provided
    if imsize is None:
        imsize = depth.shape
    if camera_matrix is None:
        h, w = depth.shape
        f = 0.5 * max(h, w)  # Focal length estimate
        cx, cy = w / 2, h / 2
        camera_matrix = [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ]

    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]

    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    Z = depth.astype(float)  # Assuming depth_image is already in the correct scale
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    pcl = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    pcl_o3d = o3d.geometry.PointCloud()
    pcl_o3d.points = o3d.utility.Vector3dVector(pcl[pcl[:, 2] > 0].astype(np.float64))

    pcl_o3d.estimate_normals()
    pcl_o3d.remove_non_finite_points()

    pcl_o3d.orient_normals_towards_camera_location()

    normals_map = np.zeros((imsize[0], imsize[1], 3), np.float64)

    camera_axis = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    pcl_normals = np.array(pcl_o3d.normals)
    points = np.asarray(pcl_o3d.points)
    valid_points = points[:, 2] > 0

    u = ((fx * points[valid_points, 0] / points[valid_points, 2]) + cx).astype(int)
    v = ((fy * points[valid_points, 1] / points[valid_points, 2]) + cy).astype(int)
    # filter out points outside of the image
    valid_points[np.where(u >= imsize[1])] = False
    valid_points[np.where(v >= imsize[0])] = False

    u = u[valid_points]
    v = v[valid_points]

    normals_map[v, u] = pcl_normals[valid_points]
    normals_map[v, u] /= np.linalg.norm(normals_map[v, u], axis=1, keepdims=True)

    normal_vector_angles = np.arccos(np.dot(normals_map, camera_axis.T))

    # Normalize to 0-255
    normals_map = (255 * (normals_map + 1) / 2).astype(np.uint8)
    normal_vector_angles = (255 * normal_vector_angles / np.pi).astype(np.uint8)

    normal_vector_angles = smooth_normals(normal_vector_angles, n_iter=5)

    return normal_vector_angles

def smooth_normals(im, n_iter=5, n_iter2=1):
    """
    Apply multiple gaussian blur to smooth the normals while not changing the known values (!= 127)
    except for the last iteration where the known values are also smoothed
    """
    im = np.copy(im)
    mask = im == (127, 127, 127)  # 127 is the value where the normals are not defined
    for _ in range(n_iter):
        im[mask] = cv2.GaussianBlur(im, (5, 5), 0)[mask]
    for _ in range(n_iter2):
        im = cv2.GaussianBlur(im, (5, 5), 0)

    return im

def rgb_to_grayscale(rgb_image):
    weights = np.array([0.299, 0.5870, 0.1140])
    grayscale_image = np.dot(rgb_image, weights)
    return grayscale_image


def process_image(args):
    ip, out_dir = args
    image_name = os.path.splitext(os.path.basename(ip))[0]
    output_path = os.path.join(out_dir, f"{image_name}.png")

    # Check if the output image already exists and can be loaded
    if os.path.exists(output_path) and cv2.imread(output_path) is not None:
        print(f"Skipping {output_path}, already exists and is loadable.")
        return

    print(f"Processing {ip}")

    im = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    rgb = im[:, :, :3]
    depth = im[:, :, 3]
    normals = get_normal_vector_angles_from_depthmap(depth)

    print(output_path)

    greyscale = rgb_to_grayscale(rgb)
    gn = np.dstack((greyscale, normals))
    cv2.imwrite(output_path, gn.astype(np.uint8))

def orient_normals_towards_camera(normals):
    """
    Ensure normals are oriented towards the camera, assuming the camera is at the origin.
    This is done by flipping normals that point away from the camera.
    """
    # Calculate the dot product of the normal with the view direction (assumed to be [0, 0, -1])
    normals = np.copy(normals)
    dot_product = normals[:,:,2]
    # Flip normals that point away from the camera
    normals[dot_product > 0] = -normals[dot_product > 0]
    return normals

coco_path = "/home/inbolt/clara/detectronDocker/dataset_for_detectron/coco2017_depth"
datasets = ["train2017", "test2017", "val2017"]

def process_images_in_directory(dataset, num_processes):
    images = sorted(glob.glob(f"{coco_path}/RGBRD/{dataset}/*.png"))
    out_dir = f"{coco_path}/GN/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    args = [(ip, out_dir) for ip in images]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(process_image, args)

def main():
    num_cores = os.cpu_count()
    datasets = ["train2017", "test2017", "val2017"]
    num_processes = int(num_cores * 0.25)
    print(num_processes)

    for dataset in datasets:
        process_images_in_directory(dataset, num_processes)

if __name__ == '__main__':
    main()
