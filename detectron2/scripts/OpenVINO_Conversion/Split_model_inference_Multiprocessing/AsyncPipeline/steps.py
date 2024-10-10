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

import multiprocessing as mp

# from .meters import MovingAverageMeter
from .models import AsyncWrapper, IEModel
from .pipeline import AsyncPipeline, PipelineStep
from .queue import Signal, is_stop_signal
from .timer import TimerGroup, IncrementalTimer


class ModelStepInfo:
    def __init__(
        self,
        model_path,
        core,
        target_device,
        num_requests,
        step_type,
        shared_loc,
        input_space=None,
        output_space=None,
    ):
        self.model_path = model_path
        self.core = core
        self.target_device = target_device
        self.num_requests = num_requests
        self.step_type = step_type
        self.input_space = input_space
        self.output_space = output_space
        self.data = False
        self.num_images = None
        self.image_folder = None
        self.shared_loc = shared_loc

    def start(self, decoder_event, end_event, start_time, end_time):
        self._process = mp.Process(
            target=self._run,
            args=(
                decoder_event,
                end_event,
                start_time,
                end_time,
            ),
        )
        self._process.start()

    def _run(self, decoder_event, end_event, start_time, end_time):
        model = IEModel(
            self.model_path,
            self.core,
            self.target_device,
            self.num_requests,
            self.step_type,
            self.shared_loc,
        )
        if self.step_type == "Backbone":
            model_step = BackboneStep(
                image_folder=self.image_folder,
                input_space=self.input_space,
                output_space=self.output_space,
                encoder=model,
            )
            decoder_event.wait()  # Backbone have to wait until the decoder is ready
            model_step._run(start_time, end_time)
        elif self.step_type == "RPN & ROI":
            model.image_folder = self.image_folder
            model_step = DecoderStep(
                input_space=self.input_space, output_space=self.output_space, decoder=model
            )
            decoder_event.set()  # The decoder is ready
            model_step._run(end_event, start_time, end_time)


class DataInfo:
    def __init__(
        self, image_folder, input_format, height, width, input_queue=None, output_queue=None
    ):
        self.image_folder = image_folder
        self.input_format = input_format
        self.height = height
        self.width = width
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.data = True

    def start(self, event, start_time, end_time):
        self._process = mp.Process(
            target=self._run,
            args=(
                start_time,
                end_time,
            ),
        )
        self._process.start()

    def _run(self, start_time, end_time):
        data_step = DataStep(
            self.input_queue,
            self.output_queue,
            self.image_folder,
            self.input_format,
            self.height,
            self.width,
        )
        data_step._run(start_time, end_time)


class RenderInfo:
    def __init__(self, image_folder, height, width, input_queue=None, output_queue=None):
        self.image_folder = image_folder
        self.height = height
        self.width = width
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.data = False

    def start(self, event, start_time, end_time):
        self._process = mp.Process(target=self._run)
        self._process.start()

    def _run(self):
        render_step = RenderStep(
            self.input_queue, self.output_queue, self.image_folder, self.height, self.width
        )
        render_step._run()

    def run(self):
        render_step = RenderStep(
            self.input_queue, self.output_queue, self.image_folder, self.height, self.width
        )
        render_step._run()


def run_pipeline(image_folder, shared_location, models):
    pipeline = AsyncPipeline()

    manager = mp.Manager()

    pipeline.add_step("Data + Backbone", models[0], shared_location, image_folder, parallel=True)
    pipeline.add_step("Decoder", models[1], shared_location, image_folder, parallel=True)

    # pipeline.add_step("Render",render_info,manager, parallel=False)

    fps = pipeline.run()
    return fps


class DataStep(PipelineStep):
    def __init__(self, input_queue, output_queue, image_folder, input_format, height, width):
        super().__init__(input_queue, output_queue)
        self.image_folder = image_folder
        self.input_format = input_format
        self.model_size = (height, width)
        self.current_idx = 0
        self.num_images = len(image_folder)
        self.encoder = False
        self.decoder = False
        self.render = False
        self.data = True

    def retrieve_and_process(self, image_path):
        original_image = detection_utils.read_image(image_path, format=self.input_format)
        original_image = original_image[:, :, 0:3]

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

        image = image.astype("float32").transpose(2, 0, 1)
        height, width = original_image.shape[:2]
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
    def __init__(self, image_folder, input_space, output_space, encoder):
        super().__init__(input_space, output_space)
        self.encoder = encoder
        self.async_model = AsyncWrapper(self.encoder, self.encoder.num_requests)
        self.first = True
        self.image_folder = image_folder
        self.num_images = len(self.image_folder)
        self.current_idx = 0
        self.input_format = "RGB"

        if "240_320" in self.encoder.model_path:
            self.model_size = (240, 320)
        elif "360_480" in self.encoder.model_path:
            self.model_size = (360, 480)
        elif "480_640" in self.encoder.model_path:
            self.model_size = (480, 640)
        elif "640_853" in self.encoder.model_path:
            self.model_size = (640, 853)
        else:
            raise ValueError("Wrong model path")

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

        return image

    def process(self, frame):
        input = [frame]
        self.async_model.infer(input)

    def preprocess(self):
        if self.current_idx == self.num_images:
            return Signal.STOP
        frame = self.retrieve_and_process(self.image_folder[self.current_idx])
        self.current_idx += 1
        return frame

    def _run(self, start_time, end_time):
        start_time.value = time.perf_counter()
        while True:
            frame_preprocess = self.preprocess()

            if is_stop_signal(frame_preprocess):
                # recuperer les derniers results qui attendenet l'inference
                self.encoder.wait_all()
                self.output_space.set(frame_preprocess)
                break

            self.process(frame_preprocess)


class DecoderStep(PipelineStep):
    def __init__(self, input_space, output_space, decoder):
        super().__init__(input_space, output_space)
        self.encoder = False
        self.decoder = decoder
        self.async_model = AsyncWrapper(self.decoder, self.decoder.num_requests)
        self.render = False
        self.data = False

    def __del__(self):
        self.decoder.cancel()

    def process(self, feature_maps):
        result = self.async_model.infer(feature_maps)

        return result

    def _run(self, end_event, start_time, end_time):

        while True:
            feature_maps = self.input_space.get()

            if is_stop_signal(feature_maps):
                # recuperer les derniers results qui attendenet l'inference
                self.decoder.wait_all()
                break

            self.process(feature_maps)

        end_time.value = time.perf_counter()
        end_event.set()


class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, input_queue, output_queue, image_folder, height, width):
        super().__init__(input_queue, output_queue)
        self.height = height
        self.width = width
        self.image_folder = image_folder
        self.idx = 0
        self._frames_processed = 0
        self.total_time_inf = 0
        self.first = True
        self.num_images = len(image_folder)
        self.decoder = False
        self.encoder = False
        self.render = True
        self.data = False

    def process(self, item):
        if item is None:
            return
        result, timers = item
        if result is None:
            return

        if self.first:
            self.start_time = timers["start_time"]
            self.first = False

        frame = cv2.imread(self.image_folder[self.idx], cv2.IMREAD_UNCHANGED)[:, :, 0:3]

        v = Visualizer(frame)
        v = v.draw_instance_predictions(result)

        cv2.imshow("Real-Time Image Display", v.get_image()[:, :, ::-1])
        cv2.waitKey(1)  # Time after which the window is closing. If remove black box is displayed
        self.idx += 1

        self._frames_processed += 1
        print("Number of frame displayed: ", self._frames_processed)

        if self._frames_processed == self.num_images:
            end_time = time.perf_counter()
            print("FPS:", self._frames_processed / (end_time - self.start_time))

        # self.total_time_inf += timers['encoder'] + timers['decoder']
        # self.fps = self._frames_processed/(self.total_time_inf/1000)
        # print('Inference time for current frame', timers['encoder'] + timers['decoder'])
        # print('Inference time for the backbone:', timers["encoder"])
        # print('Inference time for the RPN & ROI:', timers["decoder"])

        # print('Global FPS:', self.fps)
        return self._frames_processed

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
