from collections import deque
from queue import Queue

from itertools import cycle
import sys
import torch

import logging as log
import cv2
import numpy as np
import time
from openvino import AsyncInferQueue

from detectron2.modeling import detector_postprocess
from detectron2.structures import Instances, Boxes


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
        if not self.model.output_queue.empty():
            result = self.model.output_queue.get()
            return result
        else:
            return None


class IEModel:
    def __init__(self, model_path, core, target_device, num_requests, model_type):
        log.info("Reading {} model {}".format(model_type, model_path))
        self.model_path = model_path
        self.model = core.read_model(model_path)

        self.input_name = [
            self.model.inputs[i].get_any_name() for i in range(len(self.model.inputs))
        ]
        self.input_shape = [self.model.inputs[i].shape for i in range(len(self.model.inputs))]

        self.outputs = {}
        self.output_queue = Queue(maxsize=150)

        self.time_queue = Queue(maxsize=150)

        compiled_model = core.compile_model(
            self.model, target_device, config={"PERFORMANCE_HINT": "THROUGHPUT"}
        )  # config={"PERFORMANCE_HINT": "LATENCY"}
        self.output_tensor = [compiled_model.outputs[i] for i in range(len(self.model.outputs))]

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info("The {} model {} is loaded to {}".format(model_type, model_path, target_device))

        self.result_ready_2 = False

    def completion_callback(self, infer_request, id):
        res = {}
        for i in range(len(self.output_tensor)):
            res[str(i)] = infer_request.get_output_tensor(i).data

        if "rpn_roi" in self.model_path:  # Do postprocessing here
            instances = self.to_instance(res)
            res = detector_postprocess(instances, 480, 640)

        if id is None:
            self.output_queue.put(res)
        else:
            self.outputs[id] = res
        end_time = time.perf_counter()
        start_time = self.time_queue.get()

        # Display inference time for backbone and rpn_roi
        elapsed_time = (end_time - start_time) * 1000  # in ms

        if "backbone" in self.model_path:
            print("Elapsed time for backbone inference:", elapsed_time)
        if "rpn_roi" in self.model_path:
            print("Elapsed time for RPN & ROI inference:", elapsed_time)

    def async_infer(self, input, req_id=None):
        input_data = {}
        for i in range(len(self.input_name)):
            input_data[self.input_name[i]] = input[str(i)]
        self.time_queue.put(time.perf_counter())
        self.infer_queue.start_async(input_data, req_id)

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()

    def to_instance(self, result):
        results = Instances((result["0"][0], result["0"][1]))

        bounding_boxes = Boxes(torch.tensor(result["1"]))

        results.set("pred_boxes", bounding_boxes)
        results.set("scores", torch.tensor(result["4"]))
        results.set("pred_masks", torch.tensor(result["3"]))
        results.set("pred_classes", torch.tensor(result["2"]))
        return results
