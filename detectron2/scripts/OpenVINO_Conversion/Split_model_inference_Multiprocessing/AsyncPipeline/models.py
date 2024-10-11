from collections import deque
from queue import Queue

import multiprocessing as mp
from itertools import cycle
import sys
import torch
import os

import logging as log
import cv2
import numpy as np
import time
from openvino import AsyncInferQueue

from detectron2.modeling import detector_postprocess
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer


class AsyncWrapper:
    def __init__(self, ie_model, num_requests):
        self.model = ie_model
        self.num_requests = num_requests
        self._result_ready = False
        self._req_ids = cycle(range(num_requests))
        self._result_ids = cycle(range(num_requests))
        self._frames = deque(maxlen=num_requests)

    # def infer(self, model_input, frame=None):
    #     """Schedule current model input to infer, return last result"""
    #     next_req_id = next(self._req_ids)
    #     self.model.async_infer(model_input, next_req_id)

    #     if next_req_id == self.num_requests - 1: # Quand on a remplit toute la liste d'attente, on attend les premiers resultats
    #         self._result_ready = True

    #     if self._result_ready:
    #         result_req_id = next(self._result_ids)
    #         result = self.model.wait_request(result_req_id)
    #         return result
    #     else:
    #         return None

    def infer(self, model_input, frame=None):
        """Schedule current model input to infer, return last result"""
        self.model.async_infer(model_input)


class IEModel:
    def __init__(self, model_path, core, target_device, num_requests, model_type, shared_loc):
        log.info("Reading {} model {}".format(model_type, model_path))
        self.model_path = model_path
        self.model = core.read_model(model_path)

        self.input_name = [
            self.model.inputs[i].get_any_name() for i in range(len(self.model.inputs))
        ]
        self.input_shape = [self.model.inputs[i].shape for i in range(len(self.model.inputs))]

        # self.time_queue = Queue(maxsize = 150)
        self.target_device = target_device

        self.config_file = self.create_config_file(target_device)

        compiled_model = core.compile_model(self.model, target_device, config=self.config_file)

        self.output_tensor = [compiled_model.outputs[i] for i in range(len(self.model.outputs))]

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info("The {} model {} is loaded to {}".format(model_type, model_path, target_device))

        self.shared_loc = shared_loc
        self.idx = 0
        self.image_folder = None
        self.output_folder = "OpenVINO/Smaller_models/Ubuntu24/Split_in_2/results"

    def create_config_file(self, target_device):
        config = {}
        # if 'CPU' in target_device:
        # config['CPU_THREADS_NUM'] = str(0)
        # config['ENABLE_CPU_PINNING'] = 'NO'
        # config['PERFORMANCE_HINT'] = 'LATENCY'
        # if 'GPU' in target_device:
        # config['GPU_PLUGIN_THROTTLE'] = '1' # Use when doing HETERO:GPU,CPU
        # config['PERFORMANCE_HINT'] = 'THROUGHPUT'
        # config["NUM_STREAMS"] = str(6)
        # config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "YES"

        # elif 'NPU' in target_device:
        # config['PERFORMANCE_HINT'] = 'LATENCY'
        # config["NUM_STREAMS"] = str(0)
        # config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "YES"

        # config={"PERFORMANCE_HINT": "LATENCY"}

        return config

    def completion_callback(self, infer_request, id):
        res = []
        for i in range(len(self.output_tensor)):
            res.append(infer_request.get_output_tensor(i).data)
        # start_time = self.time_queue.get()
        # end_time = time.perf_counter()
        # elapsed_time = round((end_time - start_time)*1000,4) # in ms
        if "backbone" in self.model_path:
            # print("\033[91mOne backbone inference finish!\033[0m")
            self.shared_loc.set(res)

        if "rpn_roi" in self.model_path:
            # print("\033[92mOne RPN&ROI inference finish at :\033[0m")
            instances = self.to_instance(res)
            res = detector_postprocess(instances, 480, 640)
            # print("\033[92mOne RPN&ROI full inference finished at :\033[0m", time.perf_counter())
            # frame = cv2.imread(self.image_folder[self.idx],cv2.IMREAD_UNCHANGED)[:,:,0:3]
            # v = Visualizer(frame)
            # v = v.draw_instance_predictions(res)

            # image_path = os.path.join(self.output_folder, str(self.idx) +'.jpg')
            # cv2.imwrite(image_path, v.get_image()[:, :, ::-1])
            # self.idx += 1

    def async_infer(self, input, req_id=None):
        input_data = {}
        for i in range(len(self.input_name)):
            input_data[self.input_name[i]] = input[i]

        # To measure inference time
        # self.time_queue.put(time.perf_counter())

        self.infer_queue.start_async(input_data, req_id)

    def wait_all(self):  # Wait for all the reqs to finish
        self.infer_queue.wait_all()

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()

    def to_instance(self, result):
        if "GPU" in self.target_device:
            return self.to_instance_gpu(result)
        else:
            return self.to_instance_cpu(result)

    def to_instance_gpu(self, result):
        results = Instances((result[0][0], result[0][1]))

        bounding_boxes = Boxes(torch.tensor(result[1]))

        results.set("pred_boxes", bounding_boxes)
        results.set("scores", torch.tensor(result[4]))
        results.set("pred_masks", torch.tensor(result[3]))
        results.set("pred_classes", torch.tensor(result[2]))
        return results

    def to_instance_cpu(self, result):
        results = Instances((result[4][0], result[4][1]))

        bounding_boxes = Boxes(torch.tensor(result[0]))

        results.set("pred_boxes", bounding_boxes)
        results.set("scores", torch.tensor(result[3]))
        results.set("pred_masks", torch.tensor(result[2]))
        results.set("pred_classes", torch.tensor(result[1]))
        return results
