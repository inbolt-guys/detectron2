import openvino as ov
import openvino.properties.hint as hints


from AsyncPipeline.steps import run_pipeline, ModelStepInfo
from AsyncPipeline.models import IEModel
from AsyncPipeline.shared import SharedLocation
import multiprocessing as mp

import os
import re
import gc

import argparse

parser = argparse.ArgumentParser(
    description="""This script is designed to do inference using the model splitted in 2 parts. The first part being the backbone and the second part being 
                                 the RPN&ROI The project has been designed to work with RGB-RD (Raw Depth) models, if you to use a different input format please change the preprocessing in the DataSteppart of the Mask-RCNN Architecture. The structure is based on the Multithreading method, see the folder Split_model_Multithreading. 
                                 The project has been designed to work with RGB-RD (Raw Depth) models, if you to use a different input format please change the preprocessing. 
                                 To use please provide a folder containing images you want to do inference on, the model backbone path and the corresponding RPN&ROI path.
                                 You can select also the inference device for each model and also the number of request for the AsyncInferQueue of OpenVINO (see https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html).
                                 It is usually better to put the req_rpn_roi higher than req_backbone. Each model is doing inference in a different process, the communication between processes (meaning the results of the backbone inference) is handled usign a SharedLocation class that is based on 
                                 the multiprocessing library. You can use a quantized backbone with a non quantized RPN&ROI  """
)

parser.add_argument(
    "--folder_path",
    type=str,
    required=True,
    help="Folder containing the picture you want to make inference on",
)
parser.add_argument(
    "--backbone_path",
    type=str,
    required=True,
    help="Path of the backbone .xml file. Make sure the filename contains the input size of the model (for example 480_640)",
)
parser.add_argument(
    "--rpn_roi_path",
    type=str,
    required=True,
    help="Path of the RPN&ROI .xml file. Make sure the filename contains the input size of the model (for example 480_640) and that the file corresponds with the backbone file",
)
parser.add_argument(
    "--device_backbone", type=str, default="NPU", help="Device on which the backbone is compiled"
)
parser.add_argument(
    "--device_rpn_roi",
    type=str,
    default="HETERO:GPU,CPU",
    help="Device on which the RPN&ROI is compiled",
)
parser.add_argument(
    "--req_backbone", type=int, default=1, help="Device on which the backbone is compiled"
)
parser.add_argument(
    "--req_rpn_roi", type=int, default=1, help="Device on which the RPN&ROI is compiled"
)
parser.add_argument(
    "--num_images", type=int, default=100, help="Number of images to do the inference on"
)

args = parser.parse_args()


# region inference dataset

images_folder = [
    os.path.join(args.folder_path, img)
    for img in os.listdir(args.folder_path)
    if img.lower().endswith(("png", "jpg", "jpeg", "gif", "bmp"))
]

# To sort the images to have them in chronological order
# def extract_number_from_path(path):
#     # Extract the filename from the path
#     filename = os.path.basename(path)
#     # Extract the numerical part from the filename (before '.png')
#     match = re.match(r'(\d+(\.\d+)?)\.png$', filename)
#     if match:
#         # Convert the extracted number to a float
#         return float(match.group(1))
#     else:
#         # Return a default value or raise an error if needed
#         return float('inf')

# # Sort the list based on the extracted number
# images_folder = sorted(images_folder, key=extract_number_from_path)

images_folder = images_folder[: args.num_images]


core = ov.Core()


# backbone_path = 'OpenVINO_Compilation_CLI/quant_backbone_rgbrd_480_640.xml'
# rpn_roi_path = 'OpenVINO_Compilation_CLI/rpn_roi_rgbrd_480_640.xml'


backbone = core.read_model(args.backbone_path)
output_shape = [eval(backbone.outputs[i].shape.to_string()) for i in range(len(backbone.outputs))]
del backbone
gc.collect()


# # To run the pipeline
# best config for (240,320) is num_req_backbone = 1 and num_req_rpnroi = 3; best FPS : 12.213
# best config for (360,480) is num_req_backbone = 2 and num_req_rpnroi = 4, best FPS : 8.892
# best config for (480,640) is num_req_backbone = 2 and num_req_rpnroi = 6, best FPS : 8.220
# best config for (640,853) is num_req_backbone = 1 and num_req_rpnroi = 3, best FPS : 5.995 or (3,5) 5.889


# best config for (240,320) QUANT is num_req_backbone = 1 and num_req_rpnroi = 4; best FPS : 12.649
# best config for (360,480) QUANT is num_req_backbone = 1 and num_req_rpnroi = 5; best FPS : 10.427 500 images
# best config for (480,640) QUANT is num_req_backbone = 1 and num_req_rpnroi = 5; best FPS : 8.444 with 500 images I reached 9.155
# best config for (640,853) QUANT is num_req_backbone = 3 and num_req_rpnroi = 5; best FPS : 6.982 500 images

# With GuideNOW running
# best config for (240,320) is num_req_backbone = 2 and num_req_rpnroi = 4; best FPS : 8.256
# best config for (360,480) is num_req_backbone = 3 and num_req_rpnroi = 5; best FPS : 6.858


shared_location = SharedLocation(output_shape)

BackboneInfo = ModelStepInfo(
    args.backbone_path,
    core,
    args.device_backbone,
    num_requests=args.req_backbone,
    step_type="Backbone",
    shared_loc=shared_location,
)
RpnRoiInfo = ModelStepInfo(
    args.rpn_roi_path,
    core,
    args.device_rpn_roi,
    num_requests=args.req_rpn_roi,
    step_type="RPN & ROI",
    shared_loc=shared_location,
)
models = [BackboneInfo, RpnRoiInfo]

fps = run_pipeline(images_folder, shared_location, models=models)
