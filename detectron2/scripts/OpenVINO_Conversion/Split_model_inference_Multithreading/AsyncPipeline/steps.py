import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detectron2.data import detection_utils
import detectron2.data.transforms as T
import torch
from detectron2.structures import Instances, Boxes
from detectron2.modeling import detector_postprocess
from detectron2.utils.visualizer import Visualizer


import os


# from .meters import MovingAverageMeter
from .models import AsyncWrapper
from .pipeline import AsyncPipeline, PipelineStep
from .queue import Signal, is_stop_signal
from .timer import TimerGroup, IncrementalTimer


def run_pipeline(image_folder, height, width, models):
    pipeline = AsyncPipeline()
    pipeline.add_step("Data", DataStep(image_folder, height, width), parallel=True)

    pipeline.add_step("Backbone", BackboneStep(models[0]), parallel=True)
    pipeline.add_step("Decoder", DecoderStep(models[1]), parallel=True)

    pipeline.add_step("Render", RenderStep(image_folder, height, width), parallel=False)

    pipeline.run()
    # pipeline.close()
    # pipeline.print_statistics()


def softmax(x, axis=None):
    """Normalizes logits to get confidence values along specified axis"""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis)


class DataStep(PipelineStep):
    def __init__(self, image_folder, height, width):
        super().__init__()
        self.image_folder = image_folder
        self.model_size = (height, width)
        self.current_idx = 0
        self.num_images = len(image_folder)

    def retrieve_and_process(self, image_path):
        im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        rgb = im[:, :, :3]  # image of shape (H, W, C) (in BGR order)
        depth = im[:, :, 3]
        imRGBD = np.dstack((rgb, depth))

        assert imRGBD.shape[2] == 4, "image must be RGBRD"

        original_image = np.copy(imRGBD)
        original_image[..., [0, 2]] = original_image[..., [2, 0]]
        original_image = original_image.astype(np.float32)

        height, width = original_image.shape[:2]

        if self.model_size == (240, 320):
            aug = T.Resize((240, 320))
            image = aug.get_transform(original_image).apply_image(original_image)
        elif self.model_size == (360, 480):
            aug = T.Resize((360, 480))
            image = aug.get_transform(original_image).apply_image(original_image)
        elif self.model_size == (480, 640):
            image = original_image
        elif self.model_size == (640, 853):
            aug = T.ResizeShortestEdge([640, 640], 1333)
            image = aug.get_transform(original_image).apply_image(original_image)
        else:
            raise ValueError("The size is not expected.")

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to("cpu")
        return image, height, width

    def setup(self):
        pass

    def process(self, item):
        if self.current_idx == self.num_images:
            return Signal.STOP
        frame = self.retrieve_and_process(self.image_folder[self.current_idx])
        self.current_idx += 1
        return frame

    def end(self):
        pass


class BackboneStep(PipelineStep):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.async_model = AsyncWrapper(self.encoder, self.encoder.num_requests)
        self.decoder = False
        self.first = True

    def __del__(self):
        self.encoder.cancel()

    def process(self, item):
        if self.first:
            self.start_time = time.perf_counter()
            self.first = False
        frame, height, width = item
        input = {"0": frame}
        result = self.async_model.infer(input)
        # if result is not None:
        #     print(np.mean(result['1']))
        return result, {"start_time": self.start_time}


class DecoderStep(PipelineStep):
    def __init__(self, decoder):
        super().__init__()
        self.encoder = False
        self.decoder = decoder
        self.async_model = AsyncWrapper(self.decoder, self.decoder.num_requests)

    def __del__(self):
        self.decoder.cancel()

    def process(self, item):
        feature_maps, timers = item
        if feature_maps is None:
            return
        # print(np.mean(feature_maps['0']))
        # timers['decoder'] = self.own_time.last

        result = self.async_model.infer(feature_maps)
        # if result is not None:
        #     print(result['1'])
        return result, timers


class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, image_folder, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.image_folder = image_folder
        self.idx = 0
        self._frames_processed = 0
        self.total_time_inf = 0
        self.first = True
        self.num_images = len(image_folder)
        self.output_folder = "OpenVINO/Smaller_models/Ubuntu24/Split_in_2/results"

    def process(self, item):
        if item is None:
            return
        result, timers = item
        if result is None:
            return

        if self.first:
            self.start_time = timers["start_time"]
            print("Start time :", self.start_time)
            self.first = False

        # frame = cv2.imread(self.image_folder[self.idx],cv2.IMREAD_UNCHANGED)[:,:,0:3]

        # v = Visualizer(frame)
        # v = v.draw_instance_predictions(result)

        # cv2.imshow('Real-Time Image Display', v.get_image()[:, :, ::-1])
        # cv2.waitKey(1)  # Time after which the window is closing. If remove black box is displayed

        self.idx += 1

        self._frames_processed += 1
        # print("Number of frame displayed: ", self._frames_processed)
        # image_path = os.path.join(self.output_folder, str(self._frames_processed) +'.jpg')
        # cv2.imwrite(image_path, v.get_image()[:, :, ::-1])

        if self._frames_processed == self.num_images:
            end_time = time.perf_counter()
            print("Number of images processed : ", self._frames_processed)
            print("FPS:", self._frames_processed / (end_time - self.start_time))

        # self.total_time_inf += timers['encoder'] + timers['decoder']
        # self.fps = self._frames_processed/(self.total_time_inf/1000)
        # print('Inference time for current frame', timers['encoder'] + timers['decoder'])
        # print('Inference time for the backbone:', timers["encoder"])
        # print('Inference time for the RPN & ROI:', timers["decoder"])

        # print('Global FPS:', self.fps)

    def to_instance(self, result):
        results = Instances((result["0"][0], result["0"][1]))

        bounding_boxes = Boxes(torch.tensor(result["1"]))

        results.set("pred_boxes", bounding_boxes)
        results.set("scores", torch.tensor(result["4"]))
        results.set("pred_masks", torch.tensor(result["3"]))
        results.set("pred_classes", torch.tensor(result["2"]))
        return results

    def end(self):
        cv2.destroyAllWindows()

    def _sync_time(self):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        expected_time = self._t0 + (self._frames_processed + 1) / self.fps
        if self._render_time.avg:
            expected_time -= self._render_time.avg
        if expected_time > now:
            time.sleep(expected_time - now)
