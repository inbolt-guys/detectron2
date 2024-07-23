import torch
from detectron2.modeling.meta_arch.rcnnDepth import DRCNN

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
import detectron2.model_zoo
from detectron2.engine import RGBDPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pickle as pkl
import json
from copy import deepcopy
from glob import glob
from detectron2.engine import RGBDTrainer
from pipeline import Pipeline

from detectron2.data.datasets import register_coco_instances
from torchviz import make_dot
import matplotlib.pyplot as plt

def register_dataset(dataset_name: str, img_dir: str, annotations_file: str = None):
    """
    Register a dataset to the pipeline

    args:
        dataset_name: name of the dataset
        img_dir: path to the image directory
        annotations_file: path to the annotations file if not in img_dir
    """
    if annotations_file is None:
        annotations_file = os.path.join(img_dir, "annotations.json")
        assert os.path.exists(annotations_file), "Annotations file not found"
    register_coco_instances(dataset_name, {}, annotations_file, img_dir)

    with open(annotations_file) as f:
        categories = json.load(f)["categories"]
        MetadataCatalog.get(dataset_name).set(thing_classes=[cat["name"] for cat in categories])

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file("/app/detectronDocker/detectron2/configs/Base-DRCNN-FPN.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 120.0]
cfg.MODEL.PIXEL_STD =  [1.0, 1.0, 1.0, 1.0]
cfg.INPUT.FORMAT =  "RGBD"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/rgbgrey.pkl"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/model_RGBD1000/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

""" 
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 120.0, 120.0, 120.0]
cfg.MODEL.PIXEL_STD =  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cfg.INPUT.FORMAT =  "RGBN"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/doubleRGB.pkl" """


#cfg.merge_from_file("/home/clara/detectron2/configs/COCO-InstanceSegmentation/mask_Drcnn_R_50_FPN.yaml")
dataset_name = "oil_pan_short_annotated_recording"
dataset_folder = "/app/detectronDocker/dataset_for_detectron/"+dataset_name+"/"+cfg.INPUT.FORMAT
register_dataset(dataset_name, dataset_folder)

predictor = RGBDPredictor(cfg)

# Extract convolutional layers and their weights
conv_weights = []  # List to store convolutional layer weights
conv_layers = []  # List to store convolutional layers
total_conv_layers = 0  # Counter for total convolutional layers
model_weights =[]

# get all the model children as list
model_children = list(predictor.model.children())#counter to keep count of the conv layers
counter = 0#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    print(model_children[i])
    if type(model_children[i]) == torch.nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == torch.nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == torch.nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
    else:
        for j in range(len(model_children[i].children())):
            for child in model_children[i].children()[j].children():
                if type(child) == torch.nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

 
input_image = cv2.imread(random.choice(glob("/app/detectronDocker/dataset_for_detectron/oil_pan_short_annotated_recording/RGBD/*.png")), cv2.IMREAD_UNCHANGED)
# Extract feature maps
feature_maps = []  # List to store feature maps
layer_names = []  # List to store layer names
for layer in conv_layers:
    input_image = layer(input_image)
    feature_maps.append(input_image)
    layer_names.append(str(layer))

print("\nFeature maps shape")
for feature_map in feature_maps:
    print(feature_map.shape)
 
# Process and visualize feature maps
processed_feature_maps = []  # List to store processed feature maps
for feature_map in feature_maps:
    feature_map = feature_map.squeeze(0)  # Remove the batch dimension
    mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
    processed_feature_maps.append(mean_feature_map.data.cpu().numpy())



# Display processed feature maps shapes
print("\n Processed feature maps shape")
for fm in processed_feature_maps:
    print(fm.shape)
 
# Plot the feature maps
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed_feature_maps)):
    ax = fig.add_subplot(5, 4, i + 1)
    ax.imshow(processed_feature_maps[i])
    ax.axis("off")
    ax.set_title(layer_names[i].split('(')[0], fontsize=30)

