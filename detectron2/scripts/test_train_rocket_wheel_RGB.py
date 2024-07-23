from detectron2.modeling.meta_arch.rcnnDepth import DRCNN
import argparse
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
import detectron2.model_zoo
from detectron2.engine import RGBDPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pickle as pkl
import json
from copy import deepcopy
from detectron2.engine import DefaultTrainer
from pipeline import Pipeline

import yaml
import torch
from detectron2.evaluation import COCOEvaluator

if __name__ == "__main__":
    setup_logger()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=str)

    args = parser.parse_args()

    n_iter = args.n_iter if args.n_iter else 500

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("/app/detectronDocker/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.INPUT.FORMAT =  "BGR"
    cfg.MODEL.WEIGHTS = "/app/detectronDocker/model_final_f10217.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.MODEL.FREEZE_INCLUDE = []
    cfg.MODEL.UNFREEZE_SCHEDULE = []
    cfg.MODEL.UNFREEZE_ITERS = []
    cfg.WRITER_PERIOD = 20
    
    ## train model ##
    dataset_train_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/rocket_wheel_4_instances_4_poses/rgbd/"
    dataset_test_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/4_instances_rocket_steel_with_random_objects/rgbd/"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    print(hardware, n_iter)

    # paths
    old_old_result = -1
    old_result = -1
    new_result = 0

    cfg.DATASETS.TRAIN = ("rocket_wheel_4_instances_4_poses",) #optimal is 2000 iters with normal loss weights
    cfg.DATASETS.TEST = ("4_instances_rocket_steel_with_random_objects", )
    pipeline = Pipeline(
        output_dir="/app/detectronDocker/outputs/test",
        cfg=cfg,
        hardware=hardware,
            )
    pipeline.register_dataset("rocket_wheel_4_instances_4_poses", dataset_train_folder)
    pipeline.register_dataset("4_instances_rocket_steel_with_random_objects", dataset_test_folder)
    i = 0
    res = []
    while old_result <= new_result or old_old_result <= new_result or old_old_result <= old_result:
        print(old_old_result, old_result, new_result, old_result <= new_result, old_old_result <= new_result)
        del pipeline
        pipeline = Pipeline(
        output_dir="/app/detectronDocker/outputs/test",
        cfg=cfg,
        hardware=hardware,
            )
        pipeline.prepare_training(n_iter, resume=False)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        pipeline.trainer.train()
        del pipeline

        pipeline = Pipeline(
        output_dir="/app/detectronDocker/outputs/test",
        cfg=cfg,
        hardware=hardware,
        )
        pipeline.prepare_training(n_iter, resume=False)

        evaluator = COCOEvaluator("4_instances_rocket_steel_with_random_objects", output_dir="./output")
        
        results = DefaultTrainer.test(cfg=cfg, model=pipeline.trainer.model, evaluators=evaluator)
        if "segm" in results and "AP" in results["segm"]:
            print(results)
            old_old_result = old_result
            old_result = new_result
            new_result = results["segm"]["AP"]
            res.append(new_result)
            print(res)
            i+=1

    print((i-2)*n_iter)
    print(res)