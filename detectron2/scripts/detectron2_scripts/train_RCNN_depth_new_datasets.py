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
    cfg.merge_from_file(
        "/app/detectronDocker/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    num_gpu = 1
    bs = num_gpu * 2
    output_dir = "/app/detectronDocker/outputs/RCNN_DEPTH_New_datasets_no_OCID"
    cfg.MODEL.WEIGHTS = ""
    cfg.SOLVER.MAX_TO_KEEP = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 100_000
    cfg.TEST.EVAL_PERIOD = 0
    cfg.WRITER_PERIOD = 500
    cfg.EVAl_AFTER_TRAIN = False
    cfg.INPUT.FORMAT = "RD"
    cfg.DATASETS.TRAIN_REPEAT_FACTOR = [
        ("YCB_syn", 0.02),  # 64k*0.02 = 1280
        ("YCB_real", 0.1),  # 130k*0.1 = 13000
        ("TOD", 0.1),  # 280k*0.01 = 28000
    ]
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("YCB_real", "YCB_syn", "TOD")
    cfg.DATALOADER.SAMPLER_TRAIN = "WeightedTrainingSampler"
    # paths
    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    print(hardware)

    # Freeze everything in the backbones to learn the fusion steps
    cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.SOLVER.STEPS = (750_000, 875_000, 950_000)
    cfg.SOLVER.MAX_ITER = 1_000_000
    cfg.MODEL.FREEZE_INCLUDE = []
    cfg.MODEL.PIXEL_MEAN = [2750.11909722]
    cfg.MODEL.PIXEL_STD = [1486.31195796]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.UNFREEZE_SCHEDULE = []

    cfg.MODEL.UNFREEZE_ITERS = []

    pipeline = Pipeline(
        output_dir=output_dir,
        cfg=cfg,
        hardware=hardware,
    )

    ## train model ##
    ocid_folder = "/app/datasets/OCID-dataset/OCID_COCO/"
    ocid_annotations = ocid_folder + "annotations_train.json"
    ocid_images = ocid_folder + "train"
    pipeline.register_dataset("OCID", ocid_images, annotations_file=ocid_annotations)

    ycb_real_folder = "/app/datasets/YCB/YCB_COCO/"
    ycb_annotations = ycb_real_folder + "annotations_data_train.json"
    ycb_images = ycb_real_folder + "data_train"
    pipeline.register_dataset("YCB_real", ycb_images, annotations_file=ycb_annotations)

    ycb_syn_folder = "/app/datasets/YCB/YCB_COCO/"
    ycb_syn_annotations = ycb_syn_folder + "annotations_data_syn_train.json"
    ycb_syn_images = ycb_syn_folder + "data_syn_train"
    pipeline.register_dataset("YCB_syn", ycb_syn_images, annotations_file=ycb_syn_annotations)

    tod_folder = "/app/datasets/TOD/TOD_COCO/"
    tod_annotations = tod_folder + "annotations_training_set.json"
    tod_images = tod_folder + "training_set"
    pipeline.register_dataset("TOD", tod_images, annotations_file=tod_annotations)

    pipeline.prepare_training(resume=True)
    trainer = pipeline.trainer

    trainer.train()
    pipeline.save_config()
