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
    setup_logger()

    cfg_detectron = get_cfg()
    cfg_detectron.set_new_allowed(True)
    cfg_detectron.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/RCNN_classic_for_normals.yaml")
    cfg_detectron.SOLVER.CHECKPOINT_PERIOD = 100000
    cfg_detectron.TEST.EVAL_PERIOD = 0
    cfg_detectron.SOLVER.MAX_TO_KEEP = 10
    cfg_detectron.SOLVER.STEPS = (1_600_000, 2_100_000)
    cfg_detectron.SOLVER.MAX_ITER = 2_500_000
    cfg_detectron.SOLVER.IMS_PER_BATCH = 2

    num_gpu = 1
    bs = (num_gpu * 2)
    cfg_detectron.SOLVER.BASE_LR = 0.01 * bs / 16  # pick a good LR

    ## train model ##
    coco_train_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/train2017"
    coco_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/val2017"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    pipeline = Pipeline(
        output_dir="/app/detectronDocker/outputs/RCNN_Normals_1",
        cfg=cfg_detectron,
        hardware=hardware,
    )

    pipeline.register_dataset("coco_2017_normals_train", 
                              coco_train_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/annotations/instances_train2017.json")
    pipeline.register_dataset("coco_2017_normals_val", 
                              coco_val_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/annotations/instances_val2017.json")
    
    pipeline.prepare_training(resume=True)
    trainer = pipeline.trainer
    trainer.train()