import numpy as np
from scipy.ndimage import sobel, gaussian_filter
import cv2
import glob
from scipy.ndimage import uniform_filter
import os
from concurrent.futures import ProcessPoolExecutor


def compute_depth_gradients3(depth, diff=1):
    dzdx = (np.roll(depth, -diff, axis=1) - depth) / float(diff)
    dzdy = (np.roll(depth, -diff, axis=0) - depth) / float(diff)
    return dzdx, dzdy


def compute_normals3(depth, diff=1):
    dzdx, dzdy = compute_depth_gradients3(depth, diff)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-6)
    normals = smooth_normals(normals, steps=3)
    normals_255 = ((normals + 1) * 0.5 * 255).astype(np.uint8)
    return normals_255


def smooth_normals(normals, size=5, steps=5):
    for _ in range(steps):
        for i in range(3):
            normals[:, :, i] = uniform_filter(normals[:, :, i], size=size)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-6)
    return normals


def rgb_to_grayscale(rgb_image):
    weights = np.array([0.2989, 0.5870, 0.1140])
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
    normals = compute_normals3(depth)
    greyscale = rgb_to_grayscale(rgb)
    gn = np.dstack((greyscale, normals))
    cv2.imwrite(output_path, gn.astype(np.uint8))


def process_images_in_directory(dataset, num_processes):
    images = sorted(
        glob.glob(
            f"/home/clara/detectronDocker/dataset_for_detectron/coco2017_depth/{dataset}/*.png"
        )
    )
    out_dir = f"/home/clara/detectronDocker/dataset_for_detectron/coco2017_depth/GN/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    args = [(ip, out_dir) for ip in images]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(process_image, args)


def main():
    num_cores = os.cpu_count()
    datasets = ["test2017", "val2017", "train2017"]
    num_processes = num_cores // len(datasets)  # Distribute the cores across datasets

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_images_in_directory, dataset, num_processes)
            for dataset in datasets
        ]
        for future in futures:
            future.result()  # Wait for all datasets to be processed


if __name__ == "__main__":
    main()
