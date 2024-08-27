from transformers import AutoModel
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import os
import wget
import cv2
import glob
import numpy as np

def normalize_depth(depth, min_depth = None, max_depth = None, method="percentile", reverse = True, percentileForMin = 5, percentileForMax = 85, 
                    scaleFactorForMin = 1/1.5, valueWhenLessThanMin = 255, valueWhenMoreThanMax = 255, valueWhenNoValue = 255):
        noValueMask = depth == 0
        if not max_depth:
            if method=="percentile":
                max_depth = np.percentile(depth[np.logical_not(noValueMask)], percentileForMax)
            elif method=="real":
                 max_depth = np.max(depth[np.logical_not(noValueMask)])
        if not min_depth:
            if method=="percentile":
                min_depth = np.percentile(depth[np.logical_not(noValueMask)], percentileForMin)*scaleFactorForMin
            elif method=="real":
                 min_depth = np.min(depth[np.logical_not(noValueMask)])

        depth_normalized = np.copy(depth)

        lessThanMinMask = np.logical_and(depth_normalized < min_depth, np.logical_not(noValueMask))
        moreThanMaxMask = depth_normalized > max_depth

        depth_normalized = (depth_normalized-min_depth)/(max_depth-min_depth)*255

        depth_normalized[noValueMask] = valueWhenNoValue
        depth_normalized[lessThanMinMask] = valueWhenLessThanMin
        depth_normalized[moreThanMaxMask] = valueWhenMoreThanMax

        if reverse: depth_normalized = 255 - depth_normalized

        return depth_normalized.astype(np.uint8)

sys.path.append('/app/EVP/stable-diffusion')
sys.path.append('/app/src/taming-transformers')
sys.path.append('/app/src/clip')

os.chdir("/app/EVP/depth")
wget.download("https://huggingface.co/spaces/MykolaL/evp/resolve/main/depth/nyu_class_embeddings_my_captions.pth")
evp = AutoModel.from_pretrained("MykolaL/evp_depth", trust_remote_code=True)

datasets = ["train2017", "test2017", "val2017"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
evp = evp.to(device)
transform = transforms.ToTensor()

corrupted_images = ["train2017/000000071631.png"]

for d in datasets:
    os.makedirs("/app/EVPvolume/out/coco2017_depth/"+d, exist_ok=True) 

for ci in corrupted_images:
    ip = "/app/coco2017/"+ci
    image_name = ci.split("/")[1]
    im = Image.open(ip).convert("RGB")
    image = transform(im).unsqueeze(0).to(device)
    depth = evp(image)
    depth = normalize_depth(depth=depth, method="real")
    im = np.array(im)
    rgbd = np.dstack((im, depth))
    Image.fromarray(rgbd.astype(np.uint8)).save("/app/EVPvolume/out/coco2017_depth/"+ci)
