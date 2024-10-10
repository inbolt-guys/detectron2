import torch
import torch.nn.functional as F

import detectron2.data.transforms as T

from detectron2.checkpoint import DetectionCheckpointer

from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import DRCNN


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
    description="""This script can be used to convert a detectron2 trained model to OpenVINO model for deployement. To use this script 
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

# Create a torch model
torch_model = build_model(cfg)
DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
torch_model.eval()  # To do inference on it

start_time = time.perf_counter()
results = torch_model(sample_inputs)
end_time = time.perf_counter()

print("Elapsed time for the pytorch model: ", round(end_time - start_time, 2), "s")

v = Visualizer(rgb[:, :, ::-1])  #  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
v = v.draw_instance_predictions(results[0]["instances"])

plt.figure()
plt.imshow(v.get_image()[:, :, ::-1])
plt.title("Result for the Pytorch model ")
plt.show()


def convert_detectron2_model(model: torch.nn.Module, sample_input: List[Dict[str, torch.Tensor]]):
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

    # override model forward and disable postprocessing if required
    if isinstance(model, DRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    # create traceable model
    traceable_model = TracingAdapter(model, tracing_input, inference)
    warnings.filterwarnings("ignore")
    # convert PyTorch model to OpenVINO model
    ov_model = ov.convert_model(traceable_model, example_input=sample_input[0]["image"])
    return ov_model


ov_model = convert_detectron2_model(torch_model, sample_inputs)
print("Conversion has been successfull")
ov_model.reshape([4, args.input_width, args.input_height])

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

output_file = (
    script_dir
    + "/"
    + "detectron2_rgbrd_"
    + str(args.input_width)
    + "_"
    + str(args.input_height)
    + ".xml"
)
ov.save_model(ov_model, output_file)

print("The model has been saved to:", output_file)
