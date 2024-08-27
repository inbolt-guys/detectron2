from detectron2.utils.logger import setup_logger
import logging
setup_logger(level=logging.INFO)

# import some common libraries
import os, json

# import some common detectron2 utilities
from detectron2.engine import RGBDPredictor, DefaultPredictor, DepthPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import json
from detectron2.engine import RGBDTrainer, DepthTrainer, DefaultTrainer
from detectron2.data.datasets import register_coco_instances
import torch

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

def register_dataset(dataset_name: str, img_dir: str, annotations_file: str = None):
    """
    Register a dataset to the pipeline

    args:
        dataset_name: name of the dataset
        img_dir: path to the image directory
        annotations_file: path to the annotations file if not in img_dir
    """
    if annotations_file is None:
        annotations_file = os.path.join(img_dir, "annotations.json")
        assert os.path.exists(annotations_file), "Annotations file not found"
    register_coco_instances(dataset_name, {}, annotations_file, img_dir)

    with open(annotations_file) as f:
        categories = json.load(f)["categories"]
        MetadataCatalog.get(dataset_name).set(thing_classes=[cat["name"] for cat in categories])

model_names = ["RCNN_Raw_Depth_1",
                 ]
coco_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/val2017"
coco_raw_depth_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBRD/val2017"
coco_normals_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/val2017"
register_dataset("coco_2017_depth_val", 
                    coco_val_folder, 
                    annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_val2017.json")
register_dataset("coco_2017_raw_depth_val", 
                    coco_raw_depth_val_folder, 
                    annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBRD/annotations/instances_val2017.json")
register_dataset("coco_2017_normals_val", 
                    coco_normals_val_folder, 
                    annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/annotations/instances_val2017.json")



for model_name in model_names:
    model_folder = "/app/detectronDocker/outputs/" + model_name
    config_path = model_folder + "/config.yaml"
    model_path = model_folder + "/model_final.pth"
    if model_name == "base_detectron_rgb":
        model_path = model_folder + "/model_final.pkl"
    print(model_name)
    if not os.path.exists(model_path):
        print("model not found")
        continue
    if not os.path.exists(config_path):
        print("config not found")
        continue
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.INPUT.FORMAT == "RGBD" or cfg.INPUT.FORMAT == "RGBRD":
        predictor =  RGBDPredictor(cfg)
    elif cfg.INPUT.FORMAT == "D" or cfg.INPUT.FORMAT == "N" or cfg.INPUT.FORMAT == "RD":
        predictor =  DepthPredictor(cfg)
    elif cfg.INPUT.FORMAT == "RGB" or cfg.INPUT.FORMAT == "BGR":
        predictor = DefaultPredictor(cfg)
    else:
        print("UNKNOWN INPUT FORMAT")
        continue
    
    if cfg.INPUT.FORMAT == "N":
        evaluator = COCOEvaluator("coco_2017_normals_val", output_dir="./output")
        val_loader = DepthTrainer.build_test_loader(cfg, "coco_2017_normals_val")
    elif cfg.INPUT.FORMAT == "D":
        evaluator = COCOEvaluator("coco_2017_depth_val", output_dir="./output")
        val_loader = DepthTrainer.build_test_loader(cfg, "coco_2017_raw_depth_val")
    elif cfg.INPUT.FORMAT == "RD":
        evaluator = COCOEvaluator("coco_2017_raw_depth_val", output_dir="./output")
        val_loader = DepthTrainer.build_test_loader(cfg, "coco_2017_raw_depth_val")
    elif cfg.INPUT.FORMAT == "RGBD":
        evaluator = COCOEvaluator("coco_2017_depth_val", output_dir="./output")
        val_loader = RGBDTrainer.build_test_loader(cfg, "coco_2017_depth_val")
    elif cfg.INPUT.FORMAT == "RGBRD":
        evaluator = COCOEvaluator("coco_2017_raw_depth_val", output_dir="./output")
        val_loader = RGBDTrainer.build_test_loader(cfg, "coco_2017_raw_depth_val")
    else:
        evaluator = COCOEvaluator("coco_2017_depth_val", output_dir="./output")
        val_loader = DefaultTrainer.build_test_loader(cfg, "coco_2017_depth_val")
    print("hardware:", cfg.MODEL.DEVICE, 
        "\ninput format", cfg.INPUT.FORMAT,
        "\neval after train:", cfg.EVAl_AFTER_TRAIN,
        "\nevaluation period", cfg.TEST.EVAL_PERIOD,
        "\n",
        "\ncheckpoint period:", cfg.SOLVER.CHECKPOINT_PERIOD, 
        "\nmax checkpoints to keep:", cfg.SOLVER.MAX_TO_KEEP, 
        "\n",
        "\nout dir:", cfg.OUTPUT_DIR,
        "\nwriter period:", cfg.WRITER_PERIOD,
        "\npixel mean:", cfg.MODEL.PIXEL_MEAN,
        "\npixel_std:", cfg.MODEL.PIXEL_STD,
        "\n",
        "\nbase lr:", cfg.SOLVER.BASE_LR, 
        "\nmax iter:", cfg.SOLVER.MAX_ITER,
        "\nlr steps:", cfg.SOLVER.STEPS,
        "\nlr gamma:", cfg.SOLVER.GAMMA,
        "\n",
        "\nfreeze at:", cfg.MODEL.BACKBONE.FREEZE_AT,
        "\nunfreeze schedule:", cfg.MODEL.UNFREEZE_SCHEDULE,
        "\nunfreeze iters:", cfg.MODEL.UNFREEZE_ITERS,
        "\nfreeze include:", cfg.MODEL.FREEZE_INCLUDE,
        "\nfreeze_all_exclude:", cfg.MODEL.FREEZE_ALL_EXCLUDE,
        "\n",
        "\nbackbone:", cfg.MODEL.BACKBONE
    )
    print(inference_on_dataset(predictor.model, val_loader, evaluator))