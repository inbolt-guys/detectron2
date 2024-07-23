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
from detectron2.utils.analysis import flop_count_operators
import unittest
import torch

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
cfg.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/Base-double_resnetDRCNN-FPN.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model

#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 80.0]
#cfg.MODEL.PIXEL_STD =  [1.0, 1.0, 1.0, 1.0]
cfg.INPUT.FORMAT =  "RGBD"
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/rgbgrey.pkl"
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/model_RGBD1000/model_final.pth"
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/double_pretrained_backbones_RGBD.pkl"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/model_rocket_steel_1_instance_pose1_RGBD2000/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

""" 
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 120.0, 120.0, 120.0]
cfg.MODEL.PIXEL_STD =  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cfg.INPUT.FORMAT =  "RGBN"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/doubleRGB.pkl" """


#cfg.merge_from_file("/home/clara/detectron2/configs/COCO-InstanceSegmentation/mask_Drcnn_R_50_FPN.yaml")
dataset_name = "4_instances_rocket_steel_with_random_objects"
dataset_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/"+dataset_name+"/"+cfg.INPUT.FORMAT
dataset_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/4_instances_rocket_steel_with_random_objects/rgbd/"
#dataset_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_1_instance_pose1/rgbd/"
register_dataset(dataset_name, dataset_folder)

predictor = RGBDPredictor(cfg)
cv2.namedWindow('name', cv2.WINDOW_NORMAL)

while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord("n"):
        im = cv2.imread(random.choice(glob(dataset_folder+"/*.png")), cv2.IMREAD_UNCHANGED)

        rgb = im[:, :, :3]
        depth = im[:, :, 3]

        imRGBD = np.dstack((rgb, depth))
        #imRGBD = np.dstack((np.zeros_like(rgb), depth))
        #imRGBD = np.dstack((rgb, np.zeros_like(depth)))
        inputs2 = [{"image": torch.rand(4, 484, 576)}]
        print(flop_count_operators(predictor.model, inputs2))
        outputs = predictor(imRGBD)

        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        print(outputs["instances"].pred_masks)
        #print(outputs["instances"].pred_boxes.isSecondBackbone)

        v = Visualizer(im[:, :, :3][:, :, ::-1], MetadataCatalog.get("4_instances_rocket_steel_with_random_objects"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("name", out.get_image()[:, :, ::-1])
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

