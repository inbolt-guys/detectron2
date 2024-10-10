# import some common libraries
import numpy as np
import pickle as pkl
import torch
from copy import deepcopy

image_weights_file = open(
    "/app/detectronDocker/outputs/RCNN_RGB_class_agnostic_and_normalized_1class_freeze_all_except_stem/model_final.pth",
    "rb",
)
depth_weights_file = open("/app/detectronDocker/outputs/RCNN_Raw_Depth_1/model_final.pth", "rb")

image_weights = torch.load(image_weights_file, map_location=torch.device("cpu"))
depth_weights = torch.load(depth_weights_file, map_location=torch.device("cpu"))

full_model = deepcopy(image_weights)

for k in image_weights["model"].keys():
    ks = k.split(".")
    full_model["model"].pop(k)
    if "backbone" in k:
        v_image = image_weights["model"][k]
        v_depth = depth_weights["model"][k]
        full_model["model"]["backbone.fpnRGB." + ".".join(ks[1:])] = v_image
        full_model["model"]["backbone.fpnDepth." + ".".join(ks[1:])] = v_depth

for k in full_model["model"].keys():
    v = full_model["model"][k]
    if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
        print(k, v)

with open("/app/detectronDocker/late_pretrained_backbones_normalized_RGBRD.pkl", "wb") as handle:
    pkl.dump(full_model, handle, protocol=pkl.HIGHEST_PROTOCOL)
    torch.save(
        full_model,
        "/app/detectronDocker/late_pretrained_backbones_normalized_RGBRD.pth",
        pickle_protocol=pkl.HIGHEST_PROTOCOL,
    )
