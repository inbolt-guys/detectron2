from detectron2.modeling.meta_arch.rcnnDepth import DRCNN
import argparse
from detectron2.utils.logger import setup_logger

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
from detectron2.engine import RGBDTrainer
from pipeline import Pipeline
import yaml
import torch

if __name__ == "__main__":
    ALL = 1000
    setup_logger()

    cfg_detectron = get_cfg()
    cfg_detectron.set_new_allowed(True)
    cfg_detectron.merge_from_file("/app/detectronDocker/detectron2/configs/Base-catDRCNN-FPN.yaml")
    cfg_detectron.INPUT.FORMAT = "RGBD"
    cfg_detectron.MODEL.WEIGHTS = "/app/detectronDocker/outputs/catDRCNN-FPN_pretrained_RGBD_test3/model_final.pth"
    num_gpu = 1
    bs = (num_gpu * 2)
    cfg_detectron.SOLVER.BASE_LR = 0.01 * bs / 16  # pick a good LR
    cfg_detectron.SOLVER.STEPS = (150_000, 225_000, 275_000)
    cfg_detectron.SOLVER.MAX_ITER = 300_000
    cfg_detectron.SOLVER.MAX_TO_KEEP = 10
    cfg_detectron.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg_detectron.TEST.EVAL_PERIOD = 0

    cfg_detectron.MODEL.FREEZE_BACKBONES = False
    
    output_dir = "/app/detectronDocker/outputs/catDRCNN-FPN_full_training_test1"

    ## train model ##
    coco_train_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/train2017"
    coco_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/val2017"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"

    print("hardware:", hardware, 
          "base lr:",cfg_detectron.SOLVER.BASE_LR, 
          "checkpoint period:", cfg_detectron.SOLVER.CHECKPOINT_PERIOD, 
          "max checkpoints to keep:", cfg_detectron.SOLVER.MAX_TO_KEEP, 
          "evaluation period", cfg_detectron.TEST.EVAL_PERIOD,
          "max iter:", cfg_detectron.SOLVER.MAX_ITER,
          "out dir:", output_dir)
    # paths
    pipeline = Pipeline(
        output_dir=output_dir,
        cfg=cfg_detectron,
        hardware=hardware,
    )

    pipeline.register_dataset("coco_2017_depth_train", 
                              coco_train_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_train2017.json")
    pipeline.register_dataset("coco_2017_depth_val", 
                              coco_val_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_val2017.json")
    pipeline.train()
