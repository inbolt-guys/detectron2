from detectron2.modeling.meta_arch.rcnnDepth import DRCNN

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
import detectron2.model_zoo
from detectron2.engine import RGBDPredictor, DefaultPredictor
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
from thop import profile

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


model_name = "early_fusion_new_datasets_normalized_input_no_OCID_FUSE_IN_NOTHING"
model_path = os.path.join("/app/detectronDocker/outputs", model_name)
config_path = os.path.join(model_path, "config.yaml")

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(config_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_0699999.pth")

cfg.DATASETS.TEST = ("OCID_test",)
cfg.DATASETS.TRAIN = ("OCID_test",)
cfg.DATASETS.TRAIN_REPEAT_FACTOR = []
cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

#cfg.merge_from_file("/home/clara/detectron2/configs/COCO-InstanceSegmentation/mask_Drcnn_R_50_FPN.yaml")
dataset_name = "4_instances_rocket_steel_with_random_objects"
dataset_folder = f"/app/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/{dataset_name}/rgbrd/"
#dataset_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_1_instance_pose1/rgbd/"
register_dataset(dataset_name, dataset_folder)

ocid_name = "OCID"
ocid_folder = "/app/detectronDocker/dataset_for_detectron/OCID_COCO/test"
ocid_annotations = "/app/detectronDocker/dataset_for_detectron/OCID_COCO/annotations_test.json"
register_dataset(ocid_name, ocid_folder, ocid_annotations)

predictor = RGBDPredictor(cfg)
#predictor.log_config()
cv2.namedWindow('name', cv2.WINDOW_NORMAL)

while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord("n"):
        im = cv2.imread(random.choice(glob(os.path.join(ocid_folder, "*.png"))), cv2.IMREAD_UNCHANGED)

        rgb = im[:, :, :3]
        depth = im[:, :, 3]

        imRGBD = np.dstack((rgb, depth))
        #imRGBD = np.dstack((np.zeros_like(rgb), depth))
        #imRGBD = np.dstack((rgb, np.zeros_like(depth)))
        #inputs2 = [{"image": torch.rand(4, 484, 576)}]
        #print(flop_count_operators(predictor.model, inputs2))
        #flops, params = profile(predictor.model, inputs=(inputs2, ))
        #print(flops, params)
        outputs = predictor(imRGBD)

        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        print(outputs["instances"].pred_masks.shape)
        #print(outputs["instances"].pred_boxes.isSecondBackbone)

        v = Visualizer(im[:, :, :3][:, :, ::-1], MetadataCatalog.get("OCID"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("name", out.get_image()[:, :, ::-1])
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

