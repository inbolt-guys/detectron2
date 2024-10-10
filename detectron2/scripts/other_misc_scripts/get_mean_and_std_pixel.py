import numpy as np
import cv2
import glob


def normalize_depth(
    depth,
    min_depth=None,
    max_depth=None,
    reverse=True,
    percentileForMin=5,
    percentileForMax=85,
    scaleFactorForMin=1 / 1.5,
    valueWhenLessThanMin=255,
    valueWhenMoreThanMax=255,
    valueWhenNoValue=255,
):
    noValueMask = depth == 0
    if not max_depth:
        max_depth = np.percentile(depth[np.logical_not(noValueMask)], percentileForMax)
    if not min_depth:
        min_depth = (
            np.percentile(depth[np.logical_not(noValueMask)], percentileForMin) * scaleFactorForMin
        )

    depth_normalized = np.copy(depth)

    lessThanMinMask = np.logical_and(depth_normalized < min_depth, np.logical_not(noValueMask))
    moreThanMaxMask = depth_normalized > max_depth

    depth_normalized = (depth_normalized - min_depth) / (max_depth - min_depth) * 255

    depth_normalized[noValueMask] = valueWhenNoValue
    depth_normalized[lessThanMinMask] = valueWhenLessThanMin
    depth_normalized[moreThanMaxMask] = valueWhenMoreThanMax

    if reverse:
        depth_normalized = 255 - depth_normalized

    return depth_normalized.astype(np.uint8)


path = "/home/inbolt/clara/detectronDocker/dataset_for_detectron/coco2017_depth/GN"
image_paths = glob.glob(path + "/*/*.png")
image_paths = sorted(image_paths)
# Initialize sums for mean and squared sums for variance calculation
s = np.zeros(4)
s2 = np.zeros(4)
s3 = 0.0
total_pixels = 0
for ip in image_paths:
    print(ip)
    im = cv2.imread(ip, cv2.IMREAD_UNCHANGED).astype(np.float32)
    pixels = im.shape[0] * im.shape[1]
    total_pixels += pixels / 10000
    s += np.sum(im, axis=(0, 1)) / 10000
    s2 += np.sum(im**2, axis=(0, 1)) / 10000
    # s3 += np.sum(normalize_depth(im[:, :, 3], min_depth=300, max_depth=1500))/10000
    print(s, s2, total_pixels)

# Calculate mean
mean = s / total_pixels
# Calculate variance and standard deviation
variance = (s2 / total_pixels) - (mean**2)
std = np.sqrt(variance)

# meannd = s3/total_pixels

print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", std)
# print(meannd)
