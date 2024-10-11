import openvino as ov
import openvino.properties.hint as hints


from AsyncPipeline.steps import run_pipeline, ModelStepInfo
from AsyncPipeline.models import IEModel
from AsyncPipeline.shared import SharedLocation
import multiprocessing as mp

import os
import re
import gc

import psutil

import argparse

# Initialize the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--size", type=str, help="Input size of the model", default="240_320")
parser.add_argument(
    "--quant", type=str, help="If use quant model or not, values : y or n ", default="n"
)
parser.add_argument(
    "--req_backbone",
    type=int,
    help="Number of requests in the AsyncInferQueue of the backbone ",
    default=-1,
)
parser.add_argument(
    "--req_rpn_roi",
    type=int,
    help="Number of requests in the AsyncInferQueue of the RPN & ROI ",
    default=-1,
)
parser.add_argument(
    "--mean", type=int, help="Mean of fps when both num_req are given. Values : y or n ", default=-1
)
parser.add_argument(
    "--num_images", type=int, help="Number of images to use for inference", default=500
)


# Parse arguments
args = parser.parse_args()

if (args.req_backbone == -1 and args.req_rpn_roi != -1) or (
    args.req_rpn_roi == -1 and args.req_backbone != -1
):
    raise ValueError("Both num_req_backbone and req_rpn_roi must be defined")

if args.req_backbone == -1 and args.req_rpn_roi == -1:
    flag_search = True
else:
    flag_search = False

print("Percentage of RAM usage at the beggining:", psutil.virtual_memory().percent, " %")


# region inference dataset
calibraset_folder = "datasets/4_instances_rocket_steel_with_random_objects/rgbd"
# calibraset_folder = 'OpenVINO/test'

images_folder = [
    os.path.join(calibraset_folder, img)
    for img in os.listdir(calibraset_folder)
    if img.lower().endswith(("png", "jpg", "jpeg", "gif", "bmp"))
]


def extract_number_from_path(path):
    # Extract the filename from the path
    filename = os.path.basename(path)
    # Extract the numerical part from the filename (before '.png')
    match = re.match(r"(\d+(\.\d+)?)\.png$", filename)
    if match:
        # Convert the extracted number to a float
        return float(match.group(1))
    else:
        # Return a default value or raise an error if needed
        return float("inf")


# Sort the list based on the extracted number
num_im = args.num_images
images_folder = sorted(images_folder, key=extract_number_from_path)
images_folder = images_folder[:num_im]


core = ov.Core()

suffix = "OpenVINO/Smaller_models/Ubuntu24/Split_in_2/"

if args.quant == "y":
    suffix = suffix + "quant_"

backbone_path = suffix + "backbone_w_preprocess_" + args.size + ".xml"
rpn_roi_path = suffix + "rpn_roi_ov_" + args.size + ".xml"

# backbone_path = 'OpenVINO/Smaller_models/Ubuntu24/Split_in_2/backbone_w_preprocess_240_320.xml'
# rpn_roi_path = 'OpenVINO/Smaller_models/Ubuntu24/Split_in_2/rpn_roi_ov_240_320.xml'


device_name_backbone = "NPU"  #'NPU'
device_name_rpn_roi = "HETERO:GPU,CPU"

backbone = core.read_model(backbone_path)
output_shape = [eval(backbone.outputs[i].shape.to_string()) for i in range(len(backbone.outputs))]
del backbone
gc.collect()


# output_shape = [[1, 256, 160, 216], [1, 256, 80, 108], [1, 256, 40, 54], [1, 256, 20, 27], [1, 256, 10, 14]]


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


if flag_search:
    size_list = []
    fps_list = []

    for num_req_backbone in range(1, 4):
        for num_req_rpnroi in range(1, 6):
            if num_req_backbone > num_req_rpnroi:
                continue
            size_list.append((num_req_backbone, num_req_rpnroi))
            print("--------------------------------")
            print(
                "Num req for backbone :",
                num_req_backbone,
                "and num req for RPN&ROI :",
                num_req_rpnroi,
            )
            shared_location = SharedLocation(output_shape)

            BackboneInfo = ModelStepInfo(
                backbone_path,
                core,
                device_name_backbone,
                num_requests=num_req_backbone,
                step_type="Backbone",
                shared_loc=shared_location,
            )
            RpnRoiInfo = ModelStepInfo(
                rpn_roi_path,
                core,
                device_name_rpn_roi,
                num_requests=num_req_rpnroi,
                step_type="RPN & ROI",
                shared_loc=shared_location,
            )
            models = [BackboneInfo, RpnRoiInfo]
            print(
                "Percentage of RAM usage after loading models :",
                psutil.virtual_memory().percent,
                " %",
            )
            fps = run_pipeline(images_folder, shared_location, models=models)
            print(
                "Percentage of RAM usage after running pipeline :",
                psutil.virtual_memory().percent,
                " %",
            )

            fps_list.append(fps)

    index_max = max(range(len(fps_list)), key=fps_list.__getitem__)
    print("The best config is : ", size_list[index_max], "with fps :", fps_list[index_max])

else:
    if args.mean == -1:
        shared_location = SharedLocation(output_shape)
        BackboneInfo = ModelStepInfo(
            backbone_path,
            core,
            device_name_backbone,
            num_requests=args.req_backbone,
            step_type="Backbone",
            shared_loc=shared_location,
        )
        RpnRoiInfo = ModelStepInfo(
            rpn_roi_path,
            core,
            device_name_rpn_roi,
            num_requests=args.req_rpn_roi,
            step_type="RPN & ROI",
            shared_loc=shared_location,
        )
        models = [BackboneInfo, RpnRoiInfo]
        print("--------------------------------")
        print(
            "Percentage of RAM usage after loading models :", psutil.virtual_memory().percent, " %"
        )
        fps = run_pipeline(images_folder, shared_location, models=models)
        print(
            "Percentage of RAM usage after running pipeline :",
            psutil.virtual_memory().percent,
            " %",
        )
    else:
        fps_list = []

        # shared_location = SharedLocation(output_shape)
        # BackboneInfo = ModelStepInfo(backbone_path, core, device_name_backbone, num_requests = args.req_backbone, step_type = "Backbone",shared_loc = shared_location)
        # RpnRoiInfo = ModelStepInfo(rpn_roi_path, core, device_name_rpn_roi, num_requests= args.req_rpn_roi , step_type = "RPN & ROI",shared_loc = shared_location)
        # models = [BackboneInfo,RpnRoiInfo]

        for i in range(args.mean):
            shared_location = SharedLocation(output_shape)
            BackboneInfo = ModelStepInfo(
                backbone_path,
                core,
                device_name_backbone,
                num_requests=args.req_backbone,
                step_type="Backbone",
                shared_loc=shared_location,
            )
            RpnRoiInfo = ModelStepInfo(
                rpn_roi_path,
                core,
                device_name_rpn_roi,
                num_requests=args.req_rpn_roi,
                step_type="RPN & ROI",
                shared_loc=shared_location,
            )
            models = [BackboneInfo, RpnRoiInfo]
            print("--------------------------------")
            print(
                "Percentage of RAM usage after loading models :",
                psutil.virtual_memory().percent,
                " %",
            )
            fps = run_pipeline(images_folder, shared_location, models=models)
            print(
                "Percentage of RAM usage after running pipeline :",
                psutil.virtual_memory().percent,
                " %",
            )
            fps_list.append(fps)

        mean = sum(fps_list) / len(fps_list)

        print(
            "The mean FPS for req_backbone =",
            args.req_backbone,
            "and for req_rpn_roi =",
            args.req_rpn_roi,
            "is :",
            mean,
        )
