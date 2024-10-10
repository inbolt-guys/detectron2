import torch
import torch.nn.functional as F

import detectron2.data.transforms as T

import openvino as ov
import nncf

import time
import os
import numpy as np
import cv2

from torch.utils.data import Dataset

import argparse

import warnings

# Print a custom warning message at the start of the script
warnings.warn(
    "Please be sure that you use the detectron2 library installed with the inbolt detectron2 repo.",
    category=UserWarning,
)


parser = argparse.ArgumentParser(
    description="""This script is designed to quantize the RPN&ROI OpenVINO model (IR Representation = .xml file + .bin file)
                                 using a calibration dataset. To have the inputs of the RPN&ROI we need to compute first the feature maps to feed then to the quantization algorithm. 
                                 That's why the corresponding backbone is needed, to do inference. The calibration dataset should be around of 300 images that can be taken from training and validatation 
                                 datasets. The best is to do a 50/50 mix of training and validation datasets. The quantized model will be named as the 
                                 OpenVINO model with prefix "quant_" in the same folder. """
)

parser.add_argument(
    "--backbone_path",
    type=str,
    required=True,
    help="File path of the backbone .xml file (in the same folder as the .bin file). Please provide the corresponding backbone with the correct RPN&ROI model file",
)
parser.add_argument(
    "--rpn_roi_path",
    type=str,
    required=True,
    help="File path of the RPN&ROI .xml file (in the same folder as the .bin file). Please provide the corresponding RPN&ROI with the correct backbone model file",
)
parser.add_argument("--folder_path", type=str, required=True, help="Folder containing the images")
parser.add_argument(
    "--input_width", type=int, required=True, help="Input width of the OpenVINO model."
)
parser.add_argument(
    "--input_height", type=int, required=True, help="Input height of the OpenVINO model."
)
parser.add_argument(
    "--backbone_device",
    type=str,
    default="NPU",
    help="Device on which the backbone will be compiled on to do compute feature maps",
)
parser.add_argument(
    "--input_format",
    type=str,
    default="RGBRD",
    help="Input format of the model, can be:  RGBD or RGBRD. RGBD = RGB + normalized depth / RGBRD = RGB + Raw Depth",
)
args = parser.parse_args()


def get_image(original_image):

    assert original_image.shape[2] == 4, "image must be RGBD or GN"

    image = np.copy(original_image)
    if args.input_format == "RGBD":
        image[..., [0, 2]] = image[..., [2, 0]]
    elif args.input_format == "RGBRD":
        image[..., [0, 2]] = image[..., [2, 0]]
        image = image.astype(np.float32)

    height, width = image.shape[:2]

    # Resize image
    image = resize(image)

    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image.to("cpu")
    return image


def resize(image):

    aug = T.Resize((args.input_width, args.input_height))
    return aug.get_transform(image).apply_image(image)


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        rgb = im[:, :, :3]  # image of shape (H, W, C) (in BGR order)
        depth = im[:, :, 3]

        imRGBD = np.dstack((rgb, depth))
        image = get_image(imRGBD)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label


dataset = ImageFolderDataset(args.folder_path)
calibration_loader = torch.utils.data.DataLoader(dataset)


def transform_fn(data_item):
    images, _ = data_item
    image = images.squeeze(0)
    res = compiled_backbone(image)
    return res


calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

core = ov.Core()
model = core.read_model(model=args.rpn_roi_path)

core = ov.Core()
backbone_ov = core.read_model(model=args.backbone_path)
compiled_backbone = core.compile_model(
    backbone_ov, device_name=args.backbone_device, config={"PERFORMANCE_HINT": "LATENCY"}
)

rpn_roi = core.read_model(model=args.rpn_roi_path)


# Tries to remove the problem of GPU compilation for quant model. The CPU compilation is working fine.
# patterns = ['.*__module\.model\.roi_heads\.box_pooler.*']#  ['.*__module\.model\.proposal_generator.*','.*__module\.model\.roi_heads.*']
# ignored_scope = nncf.IgnoredScope(patterns=patterns)

# operation_types = ['Broadcast']
# ignored_scope = nncf.IgnoredScope(types = operation_types)


start_time = time.perf_counter()

quantized_model = nncf.quantize(model, calibration_dataset)  # ignored_scope = ignored_scope
elapsed_time = round(time.perf_counter() - start_time)

print("The quantization took :", elapsed_time, "s")

output_folder = os.path.dirname(args.rpn_roi_path) + "/quant_" + os.path.basename(args.rpn_roi_path)


ov.save_model(quantized_model, output_folder)

print("The model have been saved to location:", output_folder)
