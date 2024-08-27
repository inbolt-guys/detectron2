from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from detectron2.engine import RGBDTrainer
from pipeline import Pipeline
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

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
#cfg.merge_from_file("/app/detectronDocker/detectron2/configs/Base-RCNN-FPNmask_rcnn_R_50_FPN_3x_for_depth.yaml")
cfg.merge_from_file("/app/detectronDocker/detectron2/configs/Base-catDRCNN-FPN.yaml")
#cfg.INPUT.FORMAT =  "D"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/RCNN_Depth_test1/model_final.pth"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/catDRCNN-FPN_pretrained_RGBD_test3/model_final.pth"
cfg.MODEL.DEVICE = "cpu"

cfg.DATASETS.TEST = ("coco_2017_depth_val",)
cfg.DATASETS.TRAIN = ("coco_2017_depth_val",)


#cfg.merge_from_file("/home/clara/detectron2/configs/COCO-InstanceSegmentation/mask_Drcnn_R_50_FPN.yaml")
coco_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/val2017"
coco_train_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/train2017"
register_dataset("coco_2017_depth_val", 
                        coco_val_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_val2017.json")
register_dataset("coco_2017_depth_train", 
                        coco_train_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_train2017.json")



trainer = RGBDTrainer(cfg)
trainer.resume_or_load(resume=False)
evaluator = COCOEvaluator("coco_2017_depth_val", output_dir="./output")
RGBDTrainer.test(cfg=cfg, model=trainer.model, evaluators=evaluator)

