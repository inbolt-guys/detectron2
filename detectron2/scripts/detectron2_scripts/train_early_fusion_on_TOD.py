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
    cfg.INPUT.FORMAT = "RGBRD"
    #cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/early_fusion_DRCNN-FPN-attention-relu_full_training_test1/model_0959999.pth"
    #cfg.MODEL.WEIGHTS = "/app/detectronDocker/early_pretrained_backbones_normalized_RGBRD.pth"
    num_gpu = 1
    bs = (num_gpu * 2)    
    output_dir = "/app/detectronDocker/outputs/early_fusion_COLOR_TOD"
    cfg.SOLVER.MAX_TO_KEEP = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 100_000
    cfg.TEST.EVAL_PERIOD = 0
    cfg.WRITER_PERIOD = 1000
    cfg.EVAl_AFTER_TRAIN = False
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("TOD", "TOD_test")
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


    """cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.DATASETS.TRAIN_REPEAT_FACTOR = [
    ]
    cfg.DATASETS.TRAIN = ("coco_2017_depth_val",)"""

    ## train model ##
    #coco_folder = "/app/nas/R&D/clara/detectronDocker/dataset_for_detectron/coco2017_depth"
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

    tod_folder = "/app/datasets/TOD/TOD_COCO/"
    tod_annotations = tod_folder+"annotations_training_set.json"
    tod_images = tod_folder + "training_set"
    pipeline.register_dataset("TOD", 
                              tod_images, annotations_file=tod_annotations)
    tod_annotations = tod_folder+"annotations_test_set.json"
    tod_images = tod_folder + "test_set"
    pipeline.register_dataset("TOD_test", 
                              tod_images, annotations_file=tod_annotations)
    
    cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    cfg.SOLVER.STEPS = (750_000, 875_000, 950_000)
    cfg.SOLVER.MAX_ITER = 1_000_000
    cfg.MODEL.FREEZE_INCLUDE = []
    cfg.MODEL.PIXEL_MEAN = [103.827, 113.927, 119.949, 2750.11909722]
    cfg.MODEL.PIXEL_STD = [73.64243704, 69.95468926, 71.09685732, 1486.31195796]
    cfg.MODEL.BACKBONE.FUSION.FUSE_IN = "COLOR"


    cfg.MODEL.UNFREEZE_SCHEDULE = [

    ]
    
    cfg.MODEL.UNFREEZE_ITERS = [

    ]

    pipeline.prepare_training(resume=True)
    trainer = pipeline.trainer

    trainer.train()
    pipeline.save_config()


