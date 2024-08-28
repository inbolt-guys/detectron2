import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from detectron2.engine import RGBDTrainer, DepthTrainer, DefaultTrainer
from pipeline import Pipeline
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
import glob
from detectron2.evaluation import DatasetEvaluator
import numpy as np
import cv2
import pycocotools.mask as mask_util

# My libraries
from detectron2.scripts.detectron2_scripts.util import munkres as munkres
from detectron2.scripts.detectron2_scripts.util import utilities as util_

import logging
import numpy as np
import torch
import itertools
from collections import OrderedDict
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO

BACKGROUND_LABEL = 0

# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def boundary_overlap(predicted_mask, gt_mask, bound_th=0.003):
    """
    Compute true positives of overlapped masks, using dilated boundaries

    Arguments:
        predicted_mask  (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        overlap (float): IoU overlap of boundaries
    """
    assert np.atleast_3d(predicted_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(predicted_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = util_.seg2bmap(predicted_mask)
    gt_boundary = util_.seg2bmap(gt_mask)

    from skimage.morphology import disk

    # Dilate segmentation boundaries
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix), iterations=1)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix), iterations=1)

    # Get the intersection (true positives). Calculate true positives differently for
    #   precision and recall since we have to dilated the boundaries
    fg_match = np.logical_and(fg_boundary, gt_dil)
    gt_match = np.logical_and(gt_boundary, fg_dil)

    # Return precision_tps, recall_tps (tps = true positives)
    return np.sum(fg_match), np.sum(gt_match)

# This function is modeled off of P/R/F measure as described by Dave et al. (arXiv19)
def multilabel_metrics(prediction, gt, obj_detect_threshold=0.70):
    """ Compute Overlap and Boundary Precision, Recall, F-measure
        Also compute #objects detected, #confident objects detected, #GT objects.

        It computes these measures only of objects (2+), not background (0) / table (1).
        Uses the Hungarian algorithm to match predicted masks with ground truth masks.

        A "confident object" is an object that is predicted with more than 0.75 F-measure

        @param gt: a [H x W] numpy.ndarray with ground truth masks
        @param prediction: a [H x W] numpy.ndarray with predicted masks

        @return: a dictionary with the metrics
    """

    ### Compute F-measure, True Positive matrices ###

    # Get unique OBJECT labels from GT and prediction
    labels_gt = np.unique(gt)
    labels_gt = labels_gt[~np.isin(labels_gt, [BACKGROUND_LABEL])]
    num_labels_gt = labels_gt.shape[0]

    labels_pred = np.unique(prediction)
    labels_pred = labels_pred[~np.isin(labels_pred, [BACKGROUND_LABEL])]
    num_labels_pred = labels_pred.shape[0]

    # F-measure, True Positives, Boundary stuff
    F = np.zeros((num_labels_gt, num_labels_pred))
    true_positives = np.zeros((num_labels_gt, num_labels_pred))
    boundary_stuff = np.zeros((num_labels_gt, num_labels_pred, 2)) 
    # Each item of "boundary_stuff" contains: precision true positives, recall true positives

    # Edge cases
    if (num_labels_pred == 0 and num_labels_gt > 0 ): # all false negatives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 1.,
                'Objects Recall' : 0.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 0.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                }
    elif (num_labels_pred > 0 and num_labels_gt == 0 ): # all false positives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 0.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 0.,
                'Boundary Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                }
    elif (num_labels_pred == 0 and num_labels_gt == 0 ): # correctly predicted nothing
        return {'Objects F-measure' : 1.,
                'Objects Precision' : 1.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 1.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 1.,
                }

    # For every pair of GT label vs. predicted label, calculate stuff
    for i, gt_i in enumerate(labels_gt):

        gt_i_mask = (gt == gt_i)

        for j, pred_j in enumerate(labels_pred):
            
            pred_j_mask = (prediction == pred_j)
            
            ### Overlap Stuff ###

            # true positive
            A = np.logical_and(pred_j_mask, gt_i_mask)
            tp = np.int64(np.count_nonzero(A)) # Cast this to numpy.int64 so 0/0 = nan
            true_positives[i,j] = tp 
            
            # precision
            prec = tp/np.count_nonzero(pred_j_mask)
            
            # recall
            rec = tp/np.count_nonzero(gt_i_mask)
            
            # F-measure
            F[i,j] = (2 * prec * rec) / (prec + rec)

            ### Boundary Stuff ###
            boundary_stuff[i,j] = boundary_overlap(pred_j_mask, gt_i_mask)

    ### More Boundary Stuff ###
    boundary_prec_denom = 0. # precision_tps + precision_fps
    for pred_j in labels_pred:
        pred_mask = (prediction == pred_j)
        boundary_prec_denom += np.sum(util_.seg2bmap(pred_mask))
    boundary_rec_denom = 0. # recall_tps + recall_fns
    for gt_i in labels_gt:
        gt_mask = (gt == gt_i)
        boundary_rec_denom += np.sum(util_.seg2bmap(gt_mask))


    ### Compute the Hungarian assignment ###
    F[np.isnan(F)] = 0
    m = munkres.Munkres()
    assignments = m.compute(F.max() - F.copy()) # list of (y,x) indices into F (these are the matchings)
    idx = tuple(np.array(assignments).T)

    ### Compute the number of "detected objects" ###
    num_obj_detected = 0
    for a in assignments:
        if F[a] > obj_detect_threshold:
            num_obj_detected += 1

    # Overlap measures
    precision = np.sum(true_positives[idx]) / np.sum(prediction > 0)
    recall = np.sum(true_positives[idx]) / np.sum(gt > 0)
    F_measure = (2 * precision * recall) / (precision + recall)
    if np.isnan(F_measure): # b/c precision = recall = 0
        F_measure = 0

    # Boundary measures
    boundary_precision = np.sum(boundary_stuff[idx][:,0]) / boundary_prec_denom
    boundary_recall = np.sum(boundary_stuff[idx][:,1]) / boundary_rec_denom
    boundary_F_measure = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
    if np.isnan(boundary_F_measure): # b/c/ precision = recall = 0
        boundary_F_measure = 0


    return {'Objects F-measure' : F_measure,
            'Objects Precision' : precision,
            'Objects Recall' : recall,
            'Boundary F-measure' : boundary_F_measure,
            'Boundary Precision' : boundary_precision,
            'Boundary Recall' : boundary_recall,
            'obj_detected' : num_labels_pred,
            'obj_detected_075' : num_obj_detected,
            'obj_gt' : num_labels_gt,
            'obj_detected_075_percentage' : num_obj_detected / num_labels_gt,
            }

def create_labeled_mask(instance_masks, height, width):
    """
    Create a labeled mask image from individual object masks.

    Args:
        instance_masks (list of np.ndarray): List of binary masks for each object.

    Returns:
        np.ndarray: A single mask image where each object is labeled with a unique integer.
    """
    labeled_mask = np.zeros((height, width), dtype=np.int32)
    for i, mask in enumerate(instance_masks):
        labeled_mask[mask] = i + 1  # Labels objects starting from 1
    return labeled_mask
def rle_to_mask(rle, height, width):
    """
    Convert RLE (Run-Length Encoding) to a binary mask.
    
    Args:
        rle (dict): RLE dictionary containing 'counts' and 'size' or directly the RLE string.
        height (int): The height of the mask.
        width (int): The width of the mask.
    
    Returns:
        np.ndarray: Binary mask with shape (height, width).
    """
    mask = mask_util.decode(rle)
    if len(mask.shape) > 2:
        mask = np.sum(mask, axis=2)  # Handle multi-part objects if present
    return mask.astype(np.uint8)  # Convert to a binary mask (0s and 1s)
class CustomMultilabelEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir=None):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.metadata = MetadataCatalog.get(dataset_name)
        self.predictions = []
        self.ground_truths = []
        json_file = self.metadata.json_file
        self._coco_api = COCO(json_file)
    def reset(self):
        self.predictions = []
        self.ground_truths = []
        
    def process(self, inputs, outputs):
        """
        Process a single batch of inputs and outputs.
        """
        for input, output in zip(inputs, outputs):
            # Ground truth masks
            ann_ids = self._coco_api.getAnnIds(imgIds=input["image_id"])
            anns = self._coco_api.loadAnns(ann_ids)
            gt_masks = []
            for ann in anns:
                rle = ann["segmentation"]  # This is the RLE encoding of the mask
                binary_mask = rle_to_mask(rle, input["height"], input["width"])
                gt_masks.append(binary_mask != 0)
            gt_labeled_mask = create_labeled_mask(gt_masks, input["height"], input["width"])

            """gt_instance_masks = input["instances"].gt_masks.tensor.cpu().numpy()
            gt_labeled_mask = create_labeled_mask(gt_instance_masks)"""
            
            # Predicted masks
            pred_instance_masks = output["instances"].pred_masks.cpu().numpy()
            pred_labeled_mask = create_labeled_mask(pred_instance_masks, input["height"], input["width"])
            
            self.ground_truths.append(gt_labeled_mask)
            self.predictions.append(pred_labeled_mask)
    
    def evaluate(self):
        """
        Evaluate all predictions using the multilabel_metrics function.
        """
        results = {
            "Objects F-measure": [],
            "Objects Precision": [],
            "Objects Recall": [],
            "Boundary F-measure": [],
            "Boundary Precision": [],
            "Boundary Recall": [],
            "obj_detected": [],
            "obj_detected_075": [],
            "obj_gt": [],
            "obj_detected_075_percentage": []
        }
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            metrics = multilabel_metrics(prediction=pred, gt=gt)
            for k in results.keys():
                results[k].append(metrics[k])
        
        # Compute the average across all images
        averaged_results = {k: np.mean(v) for k, v in results.items()}
        return averaged_results

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

ocid_folder = "/app/detectronDocker/dataset_for_detectron/OCID_COCO"
ocid_train_annos = os.path.join(ocid_folder, "annotations_all.json")
ocid_train_folder = os.path.join(ocid_folder, "all")
register_dataset("OCID", 
                        ocid_train_folder, annotations_file=ocid_train_annos)
all_results = {}
models = glob.glob("/app/detectronDocker/outputs/*")
models = [item for item in models if os.path.isdir(item)]
for model_name in models:
    model_path = os.path.join("/app/detectronDocker/outputs", model_name)
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.isfile(config_path):
        continue
    m = os.path.join(model_path, "model_final.pth")
    if not os.path.isfile(m):
        continue
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.DATASETS.TEST = ("OCID",)
    cfg.DATASETS.TRAIN = ("OCID",)
    cfg.DATASETS.TRAIN_REPEAT_FACTOR = []
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    # Iterate through models and collect metrics
    # Update model configuration
    cfg.MODEL.WEIGHTS = m
    if cfg.INPUT.FORMAT == "N":
        trainer = DepthTrainer(cfg)
    elif cfg.INPUT.FORMAT == "D":
        trainer = DepthTrainer(cfg)
    elif cfg.INPUT.FORMAT == "RD":
        trainer = DepthTrainer(cfg)
    elif cfg.INPUT.FORMAT == "RGBD":
        trainer = RGBDTrainer(cfg)
    elif cfg.INPUT.FORMAT == "RGBRD":
        trainer = RGBDTrainer(cfg)
    else:
        trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Evaluate the model
    evaluator = COCOEvaluator("OCID", output_dir="./output")
    evaluator = CustomMultilabelEvaluator("OCID")
    res, inference_time = trainer.test(cfg=cfg, model=trainer.model, evaluators=evaluator, return_inference_time=True)
    print(res)
    res["inference_time"] = inference_time
    all_results[model_name] = res

with open(os.path.join("/app/detectronDocker/outputs", f"all_results_OCID.json"), "w") as out_file:
    json.dump(all_results, out_file)   
    print("json saved")