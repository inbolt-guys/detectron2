import json
import os

import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, RGBDTrainer, RGBDPredictor, DefaultPredictor, DepthTrainer, CopyPasteRGBTrainer
from detectron2.utils.logger import setup_logger

class Pipeline:
    """
    Pipeline class for detectron training and inference
    """

    def __init__(
        self,
        output_dir: str = None,
        cfg = None,
        hardware: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize the pipeline

        args:
            output_dir: path to output directory
            source: path to a directory containing a pretrained model config file
            mode: rgb | depth | normals
        """

        self.output_dir = output_dir
        self.cfg = cfg
        self.trainer = None
        self.predictor = None
        if hardware is None:
            self.hardware = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.hardware = hardware
        self.setup()

    def setup(self):
        """
        Setup the pipeline
        """
        self.cfg.MODEL.DEVICE = self.hardware  # default is cuda

        # if output_dir is specified, use it, otherwise use config file's default
        if self.output_dir:
            self.cfg.OUTPUT_DIR = self.output_dir

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def _setup_trainer(self, n_iter=None):
        """
        Setup the trainer. This function cannot be called if no dataset has been registered or the creation of the trainer will fail
        """
        self.cfg.MODEL.DEVICE = self.hardware  # default is cuda
        if n_iter is not None:
            self.cfg.SOLVER.MAX_ITER = int(n_iter) # needs to be given before build_predictor

        """# This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025  # smaller learning rate for fine tuning
        self.cfg.SOLVER.STEPS = []  # no learning rate decay"""
        if self.cfg.INPUT.FORMAT == "RGBD" or self.cfg.INPUT.FORMAT == "RGBRD":
            return RGBDTrainer(self.cfg)
        if self.cfg.INPUT.FORMAT == "D" or self.cfg.INPUT.FORMAT == "N" or self.cfg.INPUT.FORMAT == "RD":
            return DepthTrainer(self.cfg)
        if self.cfg.INPUT.FORMAT == "RGB" or self.cfg.INPUT.FORMAT == "BGR":
            return DefaultTrainer(self.cfg)
    def setup_predictor(self):
        """
        Setup the predictor
        """
        return RGBDPredictor(self.cfg)

    def train(self, n_iter=None):
        """
        Train the model
        """
        setup_logger()
        self.trainer = self._setup_trainer(n_iter)

        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.save_config()
    
    def prepare_training(self, n_iter=None, resume = False):
        """
        Train the model
        """
        setup_logger()
        self.trainer = self._setup_trainer(n_iter)
        self.trainer.resume_or_load(resume=resume)

    def register_dataset(self, dataset_name: str, img_dir: str, annotations_file: str = None):
        """
        Register a dataset to the pipeline

        args:
            dataset_name: name of the dataset
            img_dir: path to the image directory
            annotations_file: path to the annotations file if not in img_dir
        """
        if annotations_file is None:
            annotations_file = os.path.join(img_dir, "annotations.json")
            print(annotations_file)
            assert os.path.exists(annotations_file), "Annotations file not found"
        register_coco_instances(dataset_name, {}, annotations_file, img_dir)

        """self.cfg.DATASETS.TRAIN = (dataset_name,)
        self.cfg.DATASETS.TEST = ()"""
        with open(annotations_file) as f:
            categories = json.load(f)["categories"]
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
            MetadataCatalog.get(dataset_name).set(thing_classes=[cat["name"] for cat in categories])

    def evaluate(self):
        """
        Evaluate the model
        """
        self.trainer.resume_or_load(resume=False)
        self.trainer.test(self.cfg, self.trainer.model, self.predictor)

    def predict_from_array(self, img: np.ndarray):
        """
        Predict on a single image
        """
        return self.predictor(img)

    def save_config(self):
        """
        Save the config
        """
        config_yaml = self.cfg.dump()
        with open(os.path.join(self.cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
            f.write(config_yaml)

    def get_config(self):
        """
        Get the config
        """
        return self.cfg
