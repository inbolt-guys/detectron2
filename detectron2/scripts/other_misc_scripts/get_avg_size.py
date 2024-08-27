import numpy as np
import cv2
import glob

path = "/home/clara/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD"
image_paths = glob.glob(path + "/*/*.png")
image_paths = sorted(image_paths)
# Initialize sums for mean and squared sums for variance calculation
sh = 0
sw = 0
nb_pixels = 0
i = 0
p = 0
l = len(image_paths)
for ip in image_paths:
    if p < i/l*100:
        print(p)
        p+=1
    im = cv2.imread(ip, cv2.IMREAD_UNCHANGED).astype(np.float32)
    sh += im.shape[0]
    sw += im.shape[1]
    nb_pixels += im.shape[0] * im.shape[1]
    #print(sh, sw, nb_pixels)
    i+=1

# Calculate mean
meanh = sh / l
meanw = sw / l
mean_pixels = nb_pixels / l
print("Mean:", meanh, meanw, mean_pixels, meanh*meanw)