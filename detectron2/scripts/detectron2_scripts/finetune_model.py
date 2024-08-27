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
import os

if __name__ == "__main__":
    setup_logger()
    model_name = ""
    model_path = os.path.join("/app/detectronDocker/outputs", model_name)
    config_path = os.path.join(model_path, "config.yaml")
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)

    cfg.SOLVER.BASE_LR = 0.0025 * 2 / 16
    cfg.SOLVER.MAX_ITER = 2000

    dataset_name = "4_instances_rocket_steel"
    dataset_folder = "/app/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/4_instances_rocket_steel/rgbrd/"
    dataset_annotations = dataset_folder+"annotations.json"
    dataset_images = dataset_folder
    output_dir = cfg.OUTPUT_DIR + "_finetuned_on_" + dataset_name
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.DATASETS.TRAIN_REPEAT_FACTOR = [
    ]
    cfg.DATASETS.TRAIN = (dataset_name,)

    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Pipeline(
        output_dir=output_dir,
        cfg=cfg,
        hardware=hardware,
    )

    pipeline.register_dataset(dataset_name, 
                              dataset_images, annotations_file=dataset_annotations)

    pipeline.prepare_training(resume=True)
    trainer = pipeline.trainer

    trainer.train()
    pipeline.save_config()


