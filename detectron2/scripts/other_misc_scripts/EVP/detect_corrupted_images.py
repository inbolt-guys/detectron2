from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import os
import cv2
import glob
import numpy as np

datasets = ["train2017", "test2017", "val2017"]

for d in datasets:
    images = sorted(glob.glob("/app/detectronDocker/dataset_for_detectron/coco2017_depth/"+d+"/*.png"))
    for ip in images:
        if cv2.imread(ip, cv2.IMREAD_UNCHANGED) is None:
            print(ip)

