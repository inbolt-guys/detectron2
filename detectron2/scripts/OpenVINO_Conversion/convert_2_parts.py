import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List, Optional, Tuple


import detectron2.data.transforms as T

from detectron2.checkpoint import DetectionCheckpointer

from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import DRCNN

from detectron2.config import configurable

from detectron2.config import configurable
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances


from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.config import get_cfg
from detectron2.export import TracingAdapter
from detectron2.utils.visualizer import Visualizer

from typing import List, Dict

import time
import os
import numpy as np
import cv2

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import openvino as ov  # OpenVINO has to be installed. The latest version of OpenVINO 2024.3.0 (8/10/2024) is not compiling the model properly so please use version=2024.2.0

import argparse
import warnings

# Print a custom warning message at the start of the script
warnings.warn(
    "Please be sure that you use the detectron2 library installed with the inbolt detectron2 repo.",
    category=UserWarning,
)


parser = argparse.ArgumentParser(
    description="""This script can be used to convert a detectron2 trained model to a split OpenVINO model. The outputs of this script 
                                 are 2 OpenVINO models: backbone file and RPN&ROI file. This can be further use to map the backbone to the NPU for example to run a multiprocessing or multithreading inference. To use this script 
                                 please provide the folder containing the weights of the model (.pth file) and the config file (.yaml) file. Please also provide the 
                                 input size that you desired for the OpenVINO model. Please check that the model has been trained on the input size you provide. If there are several .pth files please rename the one that you
                                 want to use : model.pth and the config file: config.yaml. Adapt the config file in script to match your criteria, ex: SCORE_THRESH_TEST or filename of .pth file"""
)

parser.add_argument(
    "--folder_path",
    type=str,
    required=True,
    help="Folder containing the .pth file and the .yaml file",
)
parser.add_argument(
    "--input_width",
    type=int,
    required=True,
    help="Input width of the OpenVINO model. Make sure this input size has been trained",
)
parser.add_argument(
    "--input_height",
    type=int,
    required=True,
    help="Input height of the OpenVINO model. Make sure this input size has been trained",
)
parser.add_argument(
    "--image_path",
    type=str,
    required=True,
    help="Path of an image to do the tracing conversion to OpenVINO. You can take one picture of the training or validation dataset",
)
args = parser.parse_args()


config_path = os.path.join(args.folder_path, "config.yaml")

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(config_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

cfg.MODEL.WEIGHTS = os.path.join(args.folder_path, "model_0003999.pth")

cfg.DATASETS.TEST = ("OCID_test",)
cfg.DATASETS.TRAIN = ("OCID_test",)
cfg.DATASETS.TRAIN_REPEAT_FACTOR = []
cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
cfg.INPUT.IMAGE_SIZE_AFTER_RESIZE = [
    (args.input_width, args.input_height)
]  # [(640,853)] # New parameter of the config file added in Maxime's commit


def get_sample_inputs(original_image):

    assert original_image.shape[2] == 4, "image must be RGBD or GN"

    image = np.copy(original_image)
    if cfg.INPUT.FORMAT == "RGBD":
        image[..., [0, 2]] = image[..., [2, 0]]
    elif cfg.INPUT.FORMAT == "RGBRD":
        image[..., [0, 2]] = image[..., [2, 0]]
        image = image.astype(np.float32)

    height, width = image.shape[:2]

    # Resize image
    image = resize(image, width, height)

    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image.to(cfg.MODEL.DEVICE)

    inputs = {"image": image, "height": height, "width": width}

    # Sample ready
    sample_inputs = [inputs]
    return sample_inputs


def resize(image, width, height):
    if width == args.input_width and height == args.input_height:  # No resize needs to be done
        return image
    else:
        aug = T.Resize((args.input_width, args.input_height))
        return aug.get_transform(image).apply_image(image)


sample_image = args.image_path

im = cv2.imread(sample_image, cv2.IMREAD_UNCHANGED)

rgb = im[:, :, :3]  # image of shape (H, W, C) (in BGR order)
depth = im[:, :, 3]

imRGBD = np.dstack((rgb, depth))
sample_inputs = get_sample_inputs(imRGBD)


class PreproAndBakbone(nn.Module):
    """
    This custom class represents the backbone of a RCNN architecture.
    This module is doing the preprocessing of the input : removing mean and dividing by std AND also computing the features maps.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, images: torch.Tensor):
        """
        Here it will only be for inference
        """
        if not self.training:
            return self.inference(images)
        else:
            raise ValueError(f"The model is not in inference mode. Try model.eval()")

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        imagesRGB, imagesDepth = self.preprocess_image(batched_inputs)
        features = self.backbone(imagesRGB.tensor, imagesDepth.tensor)

        return features

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        if self.input_format == "RGBD" or self.input_format == "RGBRD":
            imagesRGB, imagesDepth = zip(*[torch.split(x, [3, 1], dim=0) for x in images])
        elif self.input_format == "GN":
            imagesRGB, imagesDepth = zip(*[torch.split(x, [1, 3], dim=0) for x in images])
        imagesRGB = list(imagesRGB)
        imagesDepth = list(imagesDepth)
        imagesRGB = ImageList.from_tensors(
            imagesRGB,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        imagesDepth = ImageList.from_tensors(
            imagesDepth,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return imagesRGB, imagesDepth


class RPNAndROI(nn.Module):
    """
    Second part of the RCNN architecture. That is to say RPN proposal and ROI Heads. This class is used to do the tracing when the model is splitted in 2.
    """

    @configurable
    def __init__(
        self,
        *,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        mode,
    ):
        """
        Args:
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.training = False
        self.mode = mode

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
        }

    def forward(self, images: torch.Tensor):
        """
        Here it will only be for inference
        """
        if not self.training:
            return self.inference(images)
        else:
            raise ValueError(f"The model is not in inference mode. Try model.eval()")

    def inference(
        self,
        features,
        detected_instances: Optional[List[Instances]] = None,
    ):
        """
        Run inference on the given feature maps.

        """
        assert not self.training

        proposals, _ = self.proposal_generator(features, None, self.mode == "simple")

        results, _ = self.roi_heads(None, features, proposals, None)
        # The post-processing is not done to do the tracing conversion to OpenVINO
        return results


# Create a torch model
torch_model = build_model(cfg)
DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
torch_model.eval()  # To do inference on it


backbone = PreproAndBakbone(
    backbone=torch_model.backbone,
    pixel_mean=torch_model.pixel_mean,
    pixel_std=torch_model.pixel_std,
    input_format=torch_model.input_format,
    vis_period=torch_model.vis_period,
)
backbone.eval()

backbone_results = backbone(sample_inputs)


rpn_roi = RPNAndROI(
    proposal_generator=torch_model.proposal_generator,
    roi_heads=torch_model.roi_heads,
    input_format=torch_model.input_format,
    vis_period=torch_model.vis_period,
    mode=torch_model.mode,
)
rpn_roi.eval()

rpn_roi_results = rpn_roi(backbone_results)

# print(rpn_roi_results)


def convert_backbone(model: torch.nn.Module, sample_input: List[Dict[str, torch.Tensor]]):
    """
    Function for converting Detectron2 models, creates TracingAdapter for making model tracing-friendly,
    prepares inputs and converts model to OpenVINO Model

    Parameters:
      model (torch.nn.Module): Model object for conversion
      sample_input (List[Dict[str, torch.Tensor]]): sample input for tracing
    Returns:
      ov_model (ov.Model): OpenVINO Model
    """
    # prepare input for tracing adapter
    tracing_input = [{"image": sample_input[0]["image"]}]

    # override model forward
    def inference(model, inputs):
        features = model.inference(inputs)
        return [{"features": features}]

    # create traceable model
    traceable_model = TracingAdapter(model, tracing_input, inference)
    warnings.filterwarnings("ignore")
    # convert PyTorch model to OpenVINO model
    ov_model = ov.convert_model(traceable_model, example_input=sample_input[0]["image"])
    return ov_model


def convert_rpn_roi(model: torch.nn.Module, features: Dict[str, torch.Tensor]):
    """
    Function for converting Detectron2 models, creates TracingAdapter for making model tracing-friendly,
    prepares inputs and converts model to OpenVINO Model

    Parameters:
      model (torch.nn.Module): Model object for conversion
      sample_input (List[Dict[str, torch.Tensor]]): sample input for tracing
    Returns:
      ov_model (ov.Model): OpenVINO Model
    """
    # prepare input for tracing adapter
    tracing_input = features

    # override model forward
    def inference(model, inputs):
        inst = model.inference(inputs)[0]
        return [{"inst": inst}]

    # create traceable model
    traceable_model = TracingAdapter(model, tracing_input, inference)
    warnings.filterwarnings("ignore")

    # First version
    # traced = torch.jit.trace(traceable_model, traceable_model.flattened_inputs)
    # ov_model = ov.convert_model(traced)

    # Second version
    ov_model = ov.convert_model(
        traceable_model,
        example_input=(
            features["p2"],
            features["p3"],
            features["p4"],
            features["p5"],
            features["p6"],
        ),
    )

    return ov_model


backbone_ov = convert_backbone(backbone, sample_inputs)
# TupleSchema(schemas=[ListSchema(schemas=[DictSchema(schemas=[IdentitySchema()], sizes=[1], keys=['image'])], sizes=[1])], sizes=[1])

rpn_roi_ov = convert_rpn_roi(rpn_roi, backbone_results)


backbone_ov.reshape([4, args.input_width, args.input_height])  # [4, 640, 853]

input_layer = rpn_roi_ov.inputs

# Define new shapes for RPN&ROI
new_shapes = {
    input_layer[0]: eval(backbone_ov.outputs[0].shape.to_string()),
    input_layer[1]: eval(backbone_ov.outputs[1].shape.to_string()),
    input_layer[2]: eval(backbone_ov.outputs[2].shape.to_string()),
    input_layer[3]: eval(backbone_ov.outputs[3].shape.to_string()),
    input_layer[4]: eval(backbone_ov.outputs[4].shape.to_string()),
}

for layer, shape in new_shapes.items():
    rpn_roi_ov.reshape({layer: shape})


# ov.save_model(backbone_ov, "OpenVINO/Smaller_models/Ubuntu24/Split_in_2/backbone_w_preprocess_640_853.xml")
# ov.save_model(rpn_roi_ov, "OpenVINO/Smaller_models/Ubuntu24/Split_in_2/rpn_roi_ov_640_853.xml")

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

output_file_backbone = (
    script_dir
    + "/"
    + "backbone_rgbrd_"
    + str(args.input_width)
    + "_"
    + str(args.input_height)
    + ".xml"
)
ov.save_model(backbone_ov, output_file_backbone)
print("The backbone model has been saved to:", output_file_backbone)

output_file_rpn_roi = (
    script_dir
    + "/"
    + "rpn_roi_rgbrd_"
    + str(args.input_width)
    + "_"
    + str(args.input_height)
    + ".xml"
)
ov.save_model(rpn_roi_ov, output_file_rpn_roi)
print("The RPN&ROI model has been saved to:", output_file_rpn_roi)
