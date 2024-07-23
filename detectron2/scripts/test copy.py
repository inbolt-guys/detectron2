from detectron2.modeling.meta_arch.rcnnDepth import DRCNN

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
import detectron2.model_zoo
from detectron2.engine import RGBDPredictor, DefaultPredictor, DepthPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pickle as pkl
import json
from copy import deepcopy
from glob import glob
from detectron2.engine import RGBDTrainer
from pipeline import Pipeline
from detectron2.data.datasets import register_coco_instances
from torchviz import make_dot

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

cfg = get_cfg()
cfg.set_new_allowed(True)
#cfg.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/Base-RCNN-FPNmask_rcnn_R_50_FPN_3x_for_depth.yaml")
#cfg.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/Base-catDRCNN-FPN.yaml")
cfg.merge_from_file("/app/detectronDocker/detectron2/configs/double_backbones_configs/Base-double_resnetDRCNN-FPN.yaml")
#cfg.INPUT.FORMAT =  "D"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/RCNN_Depth_test2/model_final.pth"
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/catDRCNN-FPN_pretrained_RGBD_test3/model_final.pth"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/early_fusion_DRCNN-FPN_full_training_test2/model_1139999.pth"
cfg.MODEL.DEVICE = "cpu"



#cfg.merge_from_file("/home/clara/detectron2/configs/COCO-InstanceSegmentation/mask_Drcnn_R_50_FPN.yaml")
coco_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/val2017"
register_dataset("coco_2017_depth_val", 
                        coco_val_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_val2017.json")

predictor = RGBDPredictor(cfg)
#predictor2 = DepthPredictor(cfg)
#cfg.MODEL.WEIGHTS = "/app/detectronDocker/outputs/RCNN_Depth_test1/model_final.pth"
#predictor = DepthPredictor(cfg)
cv2.namedWindow('name', cv2.WINDOW_NORMAL)
image_paths = glob("/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/test2017/*.png")
image_paths = sorted(image_paths)
i=0
while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord("n"):
        im = cv2.imread(random.choice(image_paths), cv2.IMREAD_UNCHANGED)
        #im = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
        i+=1

        rgb = im[:, :, :3]
        depth = im[:, :, 3]

        #outputs = predictor(depth[:, :, np.newaxis])
        #outputs2 = predictor2(depth[:, :, np.newaxis])
        outputs = predictor(im)

        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        #print(outputs["instances"].pred_boxes.isSecondBackbone)

        v = Visualizer(rgb[..., [2, 1, 0]], MetadataCatalog.get("coco_2017_depth_val"), scale=1.0)
        #v2 = Visualizer(rgb[..., [2, 1, 0]], MetadataCatalog.get("coco_2017_depth_val"), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #out2 = v2.draw_instance_predictions(outputs2["instances"].to("cpu"))
        triple_depth = np.dstack((depth, depth, depth))
        cv2.imshow("name", np.hstack((triple_depth, out.get_image()[..., [2, 1, 0]])))
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

