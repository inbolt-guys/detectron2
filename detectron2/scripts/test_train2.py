from detectron2.modeling.meta_arch.rcnnDepth import DRCNN
import argparse
from detectron2.utils.logger import setup_logger
from detectron2.utils.analysis import flop_count_operators

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
from detectron2.engine import RGBDTrainer, CopyPasteRGBTrainer
from pipeline import Pipeline

import yaml
import torch

if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=str)

    args = parser.parse_args()

    n_iter = args.n_iter if args.n_iter else 2000 

    cfg_detectron = get_cfg()
    cfg_detectron.set_new_allowed(True)
    cfg_detectron.merge_from_file("/app/detectronDocker/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg_detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    cfg_detectron.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg_detectron.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg_detectron.SOLVER.STEPS = []        # do not decay learning rate
    cfg_detectron.MODEL.ROI_HEADS.NUM_CLASSES = 1

    print(cfg_detectron.INPUT.FORMAT)

    ## train model ##/home/clara/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations
    dataset_folder_train = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/val2017"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    print(hardware, n_iter)

    # paths
    pipeline = Pipeline(
        output_dir="/app/detectronDocker/outputs/test_data_augmentation",
        cfg=cfg_detectron,
        hardware=hardware,
    )
    cfg_detectron.DATASETS.TRAIN = ("coco_2017_depth_train",)
    cfg_detectron.DATASETS.TEST = ()
    pipeline.register_dataset("coco_2017_depth_train", 
                              dataset_folder_train, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_val2017.json")
    beforeTrainParameters = {}
    pipeline.trainer = pipeline._setup_trainer(n_iter)
    pipeline.trainer.resume_or_load(resume=False)
    for name, p in pipeline.trainer.model.named_parameters():
        beforeTrainParameters[name] = p.data
    inputs2 = [{"image": torch.rand(3, 484, 576)}]
    d = flop_count_operators(pipeline.trainer.model, inputs2)
    for i in range(99):
        d2 = flop_count_operators(pipeline.trainer.model, inputs2)
        for k in d2:
            d[k]+=d2[k]
        print(i, d)
    for k in d:
        print(k, d[k]/100)
    #pipeline.train(n_iter)
    for name, p in pipeline.trainer.model.named_parameters():
        pass
        #print(name, (beforeTrainParameters[name] - p.data).abs().sum())
    # train calls "save_config" when training is done so the updated detectron config should be saved in model_output_dir