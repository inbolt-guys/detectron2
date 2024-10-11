import numpy as np
from scipy.ndimage import sobel, gaussian_filter
import cv2
import random
import glob
from scipy.ndimage import uniform_filter
import os


def compute_depth_gradients(depth):
    dzdx = sobel(depth, axis=1)  # Gradient in x direction
    dzdy = sobel(depth, axis=0)  # Gradient in y direction
    return dzdx, dzdy


def compute_depth_gradients2(depth, diff=1):
    dzdx = (np.roll(depth, -diff, axis=1) - np.roll(depth, diff, axis=1)) / 2.0
    dzdy = (np.roll(depth, -diff, axis=0) - np.roll(depth, diff, axis=0)) / 2.0
    return dzdx, dzdy


def compute_depth_gradients3(depth, diff=1):
    dzdx = (np.roll(depth, -diff, axis=1) - depth) / float(diff)
    dzdy = (np.roll(depth, -diff, axis=0) - depth) / float(diff)
    return dzdx, dzdy


def compute_normals(depth):
    dzdx, dzdy = compute_depth_gradients(depth)

    # Construct normal vectors
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))

    # Normalize the normal vectors
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-6)  # Add a small value to avoid division by zero
    normals = smooth_normals(normals, steps=3)
    normals_255 = ((normals + 1) * 0.5 * 255).astype(np.uint8)
    return normals_255


def compute_normals2(depth, diff=1):
    dzdx, dzdy = compute_depth_gradients2(depth, diff)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-6)
    normals = smooth_normals(normals, steps=3)
    normals_255 = ((normals + 1) * 0.5 * 255).astype(np.uint8)
    return normals_255


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


def smooth_depth_map(depth, sigma=1, steps=10):
    for _ in range(steps):
        depth = gaussian_filter(depth, sigma=sigma)
    return depth


def rgb_to_grayscale(rgb_image):
    weights = np.array([0.2989, 0.5870, 0.1140])
    grayscale_image = np.dot(rgb_image, weights)
    return grayscale_image


datasets = ["train2017"]
for d in datasets:
    dir = "/home/clara/detectronDocker/dataset_for_detectron/coco2017_depth/GN/" + d
    os.makedirs(dir, exist_ok=True)
    images = sorted(
        glob.glob(
            "/home/clara/detectronDocker/dataset_for_detectron/coco2017_depth/" + d + "/*.png"
        )
    )
    for ip in images:
        image_name = os.path.splitext(os.path.basename(ip))[0]
        out_file = dir + "/" + image_name + ".png"
        if cv2.imread(out_file) is not None:
            print(ip, "skipped")
            continue
        print(ip)

        im = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        rgb = im[:, :, :3]
        depth = im[:, :, 3]

        normals = compute_normals3(depth)
        greyscale = rgb_to_grayscale(rgb)

        gn = np.dstack((greyscale, normals))

        cv2.imwrite(out_file, gn.astype(np.uint8))
