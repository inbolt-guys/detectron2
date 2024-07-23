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
    cfg.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/Base-double_resnetDRCNN-FPN.yaml")
    cfg.MODEL.WEIGHTS = "/app/detectronDocker/early_fusion_pretrained_RGBRD.pkl"
    num_gpu = 1
    bs = (num_gpu * 2)    
    output_dir = "/app/detectronDocker/outputs/early_fusion_RGBRD_color_not_normalized_2"
    cfg.SOLVER.MAX_TO_KEEP = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 100_000
    cfg.TEST.EVAL_PERIOD = 0
    cfg.WRITER_PERIOD = 100
    cfg.EVAl_AFTER_TRAIN = False

    ## train model ##
    #coco_folder = "/app/nas/R&D/clara/detectronDocker/dataset_for_detectron/coco2017_depth"
    coco_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth"
    coco_train_folder = coco_folder + "/RGBRD/train2017"
    coco_val_folder = coco_folder + "/RGBRD/val2017"
    hardware = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    pipeline = Pipeline(
        output_dir=output_dir,
        cfg=cfg,
        hardware=hardware,
    )

    pipeline.register_dataset("coco_2017_raw_depth_train", 
                              coco_train_folder, annotations_file=coco_folder + "/RGBRD/annotations/instances_train2017.json")
    pipeline.register_dataset("coco_2017_raw_depth_val", 
                              coco_val_folder, annotations_file=coco_folder + "/RGBRD/annotations/instances_val2017.json")
    
    #Freeze everything in the backbones to learn the fusion steps
    cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    cfg.SOLVER.STEPS = (1_800_000, 2_350_000)
    cfg.SOLVER.MAX_ITER = 2_500_000
    cfg.MODEL.FREEZE_INCLUDE = ["resnet"]
    #cfg.MODEL.PIXEL_MEAN = [103.827, 113.927, 119.949, 155.265]
    #cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 155.265]

    cfg.INPUT.FORMAT = "RGBRD"
    cfg.MODEL.PIXEL_MEAN[3] = 2750.11909722
    cfg.MODEL.PIXEL_STD[3] = 1486.31195796
    cfg.DATASETS.TRAIN = ("coco_2017_raw_depth_train",)
    cfg.DATASETS.TEST= ("coco_2017_raw_depth_val",)

    cfg.MODEL.UNFREEZE_SCHEDULE = [
        ["backbone.bottom_up.resnet2.res5"],
        ["backbone.bottom_up.resnet2.res4"],
        ["backbone.bottom_up.resnet2.res3"],
        ["backbone.bottom_up.resnet2.res2"],
        ["backbone.bottom_up.resnet2.stem"],
         
        ["backbone.bottom_up.resnet1.res5"],
        ["backbone.bottom_up.resnet1.res4"],
        ["backbone.bottom_up.resnet1.res3"],
        ["backbone.bottom_up.resnet1.res2"],
        ["backbone.bottom_up.resnet1.stem"], ]
    
    cfg.MODEL.UNFREEZE_ITERS = [
         150_000,
         250_000,
         350_000,
         450_000,
         550_000,

         800_000,
         900_000,
         1_000_000,
         1_100_000,
         1_200_000,
    ]

    pipeline.prepare_training(resume=True)
    trainer = pipeline.trainer

    trainer.train()
    pipeline.save_config()


