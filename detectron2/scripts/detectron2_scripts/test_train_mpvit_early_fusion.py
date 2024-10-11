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
from detectron2.modeling.backbone.MPViT.config import add_mpvit_config

if __name__ == "__main__":
    setup_logger()

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_mpvit_config(cfg)
    cfg.merge_from_file("/app/detectronDocker/detectron2/configs/MPViT/mpvit_early_fusion.yaml")
    cfg.INPUT.FORMAT = "RGBD"
    num_gpu = 1
    bs = num_gpu * 2
    output_dir = "/app/detectronDocker/outputs/test_mpvit_early_fusion"
    cfg.SOLVER.MAX_TO_KEEP = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 100_000
    cfg.TEST.EVAL_PERIOD = 0
    cfg.WRITER_PERIOD = 1
    cfg.EVAl_AFTER_TRAIN = False

    cfg.DATASETS.TRAIN = "coco_2017_depth_val"

    ## train model ##
    # coco_folder = "/app/nas/R&D/clara/detectronDocker/dataset_for_detectron/coco2017_depth"
    coco_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth"
    coco_train_folder = coco_folder + "/RGBD/train2017"
    coco_val_folder = coco_folder + "/RGBD/val2017"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    pipeline = Pipeline(
        output_dir=output_dir,
        cfg=cfg,
        hardware=hardware,
    )

    pipeline.register_dataset(
        "coco_2017_depth_train",
        coco_train_folder,
        annotations_file=coco_folder + "/RGBD/annotations/instances_train2017.json",
    )
    pipeline.register_dataset(
        "coco_2017_depth_val",
        coco_val_folder,
        annotations_file=coco_folder + "/RGBD/annotations/instances_val2017.json",
    )

    # Freeze everything in the backbones to learn the fusion steps
    cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    cfg.SOLVER.STEPS = (1_000_000, 1_300_000, 1_600_000)
    cfg.SOLVER.MAX_ITER = 1_800_000
    cfg.MODEL.FREEZE_INCLUDE = []
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 155.265]

    cfg.MODEL.UNFREEZE_SCHEDULE = []

    cfg.MODEL.UNFREEZE_ITERS = []

    pipeline.prepare_training(resume=True)
    trainer = pipeline.trainer

    trainer.train()
    pipeline.save_config()
