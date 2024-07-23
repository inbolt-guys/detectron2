from detectron2.modeling.meta_arch.rcnnDepth import DRCNN
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from copy import deepcopy
from detectron2.engine import RGBDTrainer
from pipeline import Pipeline
import yaml
import torch

if __name__ == "__main__":
    setup_logger()

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/Base-double_resnetDRCNN-FPN_old.yaml")
    cfg.INPUT.FORMAT = "RGBD"
    cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/early_fusion_DRCNN-FPN_full_training_test2/model_0939999.pth"
    num_gpu = 1
    bs = (num_gpu * 2)    
    output_dir = "/app/detectronDocker/outputs/early_fusion_DRCNN-FPN_retrain_from_unfreezing_resnet1_res2_1"
    cfg.SOLVER.MAX_TO_KEEP = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 20000
    cfg.TEST.EVAL_PERIOD = 0
    cfg.WRITER_PERIOD = 500
    cfg.EVAl_AFTER_TRAIN = False
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    ## train model ##
    coco_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth"
    #coco_folder = "/app/nas/R&D/clara/detectronDocker/dataset_for_detectron/coco2017_depth"
    coco_train_folder = coco_folder + "/RGBD/train2017"
    coco_val_folder = coco_folder + "/RGBD/val2017"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    pipeline = Pipeline(
        output_dir=output_dir,
        cfg=cfg,
        hardware=hardware,
    )

    pipeline.register_dataset("coco_2017_depth_train", 
                              coco_train_folder, annotations_file=coco_folder + "/RGBD/annotations/instances_train2017.json")
    pipeline.register_dataset("coco_2017_depth_val", 
                              coco_val_folder, annotations_file=coco_folder + "/RGBD/annotations/instances_val2017.json")
    
    #Freeze everything in the backbones to learn the fusion steps
    cfg.SOLVER.BASE_LR = 0.002 * bs / 16  # pick a good LR
    cfg.SOLVER.GAMMA = 0.333
    cfg.SOLVER.STEPS = (250_000, 450_000)
    cfg.SOLVER.MAX_ITER = 600_000
    cfg.MODEL.FREEZE_INCLUDE = ["backbone.bottom_up.resnet1.res2", "backbone.bottom_up.resnet1.stem"]

    cfg.MODEL.UNFREEZE_SCHEDULE = [
        ["backbone.bottom_up.resnet1.res2"],
        ["backbone.bottom_up.resnet1.stem"]]
    
    cfg.MODEL.UNFREEZE_ITERS = [
        175_000,
        250_000,
    ]

    pipeline.prepare_training()
    trainer = pipeline.trainer

    trainer.train()
    pipeline.save_config()


