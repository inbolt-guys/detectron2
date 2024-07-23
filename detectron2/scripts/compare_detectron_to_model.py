from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, RGBDPredictor, DepthPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from glob import glob
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

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file("/app/detectronDocker/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD =  [1.0, 1.0, 1.0]
cfg.INPUT.FORMAT =  "RGB"
cfg.MODEL.WEIGHTS = "/app/detectronDocker/model_final_f10217.pkl"
cfg.MODEL.DEVICE = "cpu"


model2_name = "early_fusion_DRCNN-FPN-attention-relu_full_training_from_scratch"
cfg2 = get_cfg()
cfg2.set_new_allowed(True)
cfg2.merge_from_file(f"/app/detectronDocker/outputs/{model2_name}/config.yaml")
cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # set threshold for this model
cfg2.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

cfg2.MODEL.WEIGHTS = f"/app/detectronDocker/outputs/{model2_name}/model_final.pth"
cfg2.MODEL.DEVICE = "cpu"

model3_name = "RCNN_Depth_good_pixel_mean"
model3_name = "RCNN_Depth_test2"

cfg3 = get_cfg()
cfg3.set_new_allowed(True)
cfg3.merge_from_file(f"/app/detectronDocker/outputs/{model3_name}/config.yaml")
cfg3.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # set threshold for this model
cfg3.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

cfg3.MODEL.WEIGHTS = f"/app/detectronDocker/outputs/{model3_name}/model_final.pth"
cfg3.MODEL.DEVICE = "cpu"

model4_name = "RCNN_Normals_1"

cfg4 = get_cfg()
cfg4.set_new_allowed(True)
cfg4.merge_from_file(f"/app/detectronDocker/outputs/{model4_name}/config.yaml")
cfg4.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # set threshold for this model
cfg4.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

cfg4.MODEL.WEIGHTS = f"/app/detectronDocker/outputs/{model4_name}/model_final.pth"
cfg4.MODEL.DEVICE = "cpu"

coco_val_folder = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/val2017"
register_dataset("coco_2017_depth_val", 
                        coco_val_folder, annotations_file="/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/annotations/instances_val2017.json")

predictor = DefaultPredictor(cfg)
predictor2 = RGBDPredictor(cfg2)
predictor3 = DepthPredictor(cfg3)
predictor4 = DepthPredictor(cfg4)

cv2.namedWindow('name', cv2.WINDOW_NORMAL)
image_folder_RGBD = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/RGBD/test2017/"
image_folder_GN = "/app/detectronDocker/dataset_for_detectron/coco2017_depth/GN/test2017/"
image_paths = glob(image_folder_RGBD+"*.png")
image_paths = sorted(image_paths)
i = 0
while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord("n"):
        im_name = os.path.basename(random.choice(image_paths))
        imRGBD = cv2.imread(image_folder_RGBD + im_name, cv2.IMREAD_UNCHANGED)
        imGN = cv2.imread(image_folder_GN + im_name, cv2.IMREAD_UNCHANGED)
        
        #im = np.rot90(im, 2)

        rgb = imRGBD[:, :, :3]
        depth = imRGBD[:, :, 3:]
        grey = imGN[:, :, :1]
        normals = imGN[:, :, 1:]

        outputs = predictor(rgb)
        #outputs2 = predictor2(imRGBD)
        #outputs3 = predictor3(imRGBD)
        outputs4 = predictor4(imGN)

        v = Visualizer(rgb[:, :, ::-1], MetadataCatalog.get("coco_2017_depth_val"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        v2 = Visualizer(normals, MetadataCatalog.get("coco_2017_depth_val"), scale=1.2)
        out2 = v2.draw_instance_predictions(outputs4["instances"].to("cpu"))
        cv2.imshow("name", np.hstack((out.get_image()[:, :, ::-1], out2.get_image()[:, :, ::-1])))
        i+=1
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

