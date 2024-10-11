import logging as log
import time
from collections import OrderedDict
from itertools import chain, cycle

import multiprocessing as mp
from quick_queue import QQueue

import time
import numpy as np

from .queue import AsyncQueue, Signal, StubQueue, VoidQueue, is_stop_signal
from .timer import TimerGroup, IncrementalTimer


class PipelineStep:
    def __init__(self, input_space, output_space):
        self.input_space = input_space
        self.output_space = output_space
        self.working = False

        self._start_t = None
        self._process = None

    def process(self, item):
        raise NotImplementedError

    def end(self):
        pass

    def setup(self):
        pass

    def start(self):
        if self.input_queue is None or self.output_queue is None:
            raise Exception("No input or output queue")

        if self._process is not None:
            raise Exception("Thread is already running")
        self._process = mp.Process(target=self._run)
        self._process.start()
        self.working = True

    def join(self):
        # self.input_queue.put(Signal.STOP)
        self._thread.join()
        self._thread = None
        self.working = False

    def _run(self, start_time, end_time):

        self.own_time = IncrementalTimer()

        if self.encoder:
            start_time.value = time.perf_counter()

        while True:
            if self.render:
                if self._frames_processed == 8:
                    print("Problematic it at :", time.perf_counter())
                    self.input_queue.put_remain()
                    print("Number of qsize in input queue of Render:", self.input_queue.qsize())
                self.input_queue.put_remain()

            item = self.input_queue.get()

            if is_stop_signal(item):
                print("\033[33mInput is stop signal\033[0m")
                if self.encoder:
                    print("\033[33mEncoder : Input is stop signal\033[0m")
                    self.encoder.infer_queue.wait_all()
                    print("Size of encoder output queue : ", self.output_queue.qsize())
                    interlist = []
                    while not self.encoder.output_queue.empty():
                        res = self.encoder.output_queue.get()
                        output = res, {"encoder": self.own_time.last}
                        self.output_queue.put(output)
                        # interlist.append(output)
                    # interlist.append(item)
                    # self.output_queue.put_bucket(interlist)
                    # self.output_queue.put_remain()
                    self.output_queue.put(item)
                    print(
                        "Size of encoder output queue before put_remain : ",
                        self.output_queue.qsize(),
                    )
                    self.output_queue.put_remain()

                    print(
                        "Size of encoder output queue after put_remain: ", self.output_queue.qsize()
                    )
                if self.decoder:
                    print("\033[33mDecoder : Input is stop signal\033[0m")
                    self.decoder.infer_queue.wait_all()
                    while not self.decoder.output_queue.empty():
                        res = self.decoder.output_queue.get()
                        output = res, {"decoder": self.own_time.last}
                        self.output_queue.put(output)
                    # self.output_queue.put_remain()
                    print(item)
                    self.output_queue.put(item)

                    self.output_queue.put_remain()
                    self.output_queue.close()
                    # while not self.output_queue.empty():
                    #     print(self.output_queue.get())
                    print("Finishing output queue of decoder at :", time.perf_counter())
                    print(self.output_queue.qsize())

                # self.output_queue.put(item)
                break

            self.own_time.tick()
            output = self.process(item)
            self.own_time.tock()

            if self._check_output(output):  # Only used at the end of DataStep
                print("\033[33mOutput is stop signal\033[0m")
                break

            # To measure the time taken to put in the queue
            # start_time = time.perf_counter()
            # self.output_queue.put(output)
            # elapsed_time = time.perf_counter() - start_time
            # print('\033[94mTime spent to put an element in the queue\033[0m', round(elapsed_time*1000), 'ms for','\033[93m' + str(self) + '\033[0m')

            self.output_queue.put(output)

        if self.data:
            print("End of process of data step : ", time.perf_counter())

        # print(self.input_queue.empty())

        if self.decoder:
            print("End of while loop of decoder at time:", time.perf_counter())
            end_time.value = time.perf_counter()
        if self.encoder:
            print("End of while loop of encoder at time:", time.perf_counter())

    def _check_output(self, item):
        if is_stop_signal(item):
            self.output_queue.put(item)
            self.output_queue.put_remain()
            print(self.output_queue.qsize())
            return True
        return False


class AsyncPipeline:
    def __init__(self):
        self.steps = OrderedDict()
        self.sync_steps = OrderedDict()
        self.async_step = []

        self.incr = 1

        self.all_queue = []
        self.num_images = None

        self._void_queue = VoidQueue()
        self._first = True
        self._last_parallel = False

    def add_step(self, name, step_info, shared_location, image_folder, parallel=True):
        if self._first:
            self.num_images = len(image_folder)
            step_info.image_folder = image_folder
            step_info.input_space = None
            step_info.output_space = shared_location

            if parallel:
                self.steps[name] = step_info
            else:
                self.sync_steps[name] = step_info

            self._first = False

        else:
            step_info.num_images = len(image_folder)
            step_info.image_folder = image_folder
            step_info.input_space = shared_location

            step_info.output_queue = VoidQueue
            if parallel:
                self.steps[name] = step_info
            else:
                self.sync_steps[name] = step_info

    def run(self):
        decoder_event = mp.Event()  # event to say that the decoder is fully loaded
        start_time = mp.Value("d", 0.0)
        end_time = mp.Value("d", 0.0)
        end_event = mp.Event()  # Event to say when all inferences are finished

        for step_info in self.steps.values():
            step_info.start(decoder_event, end_event, start_time, end_time)
        # self._run_sync_steps()

        # for step_info in self.steps.values():
        #     step_info._process.join()

        end_event.wait()

        # print("Start time :", start_time.value)
        # print("End time :", end_time.value)
        elasped_time = end_time.value - start_time.value
        fps = round(self.num_images / elasped_time, 3)

        print("FPS :", fps)
        return fps

    def close(self):
        for step in self.steps.values():
            step.input_queue.put(Signal.STOP_IMMEDIATELY)
        for step in self.steps.values():
            step.join()

    def print_statistics(self):
        log.info("Metrics report:")
        for name, step in chain(
            self.sync_steps.items(),
            self.steps.items(),
        ):
            log.info("\t{} total: {}".format(name, step.total_time))
            log.info("\t{}   own: {}".format(name, step.own_time))

    def _run_sync_steps(self):
        """Run steps in main thread"""
        # for step in self.steps.values():
        #     step.join()

        if not self.sync_steps:
            while not self._void_queue.finished:
                pass
            return

        # for step in cycle(self.sync_steps.values()):
        for step_info in self.sync_steps.values():
            step_info.run()

        #     step.total_time.tick()
        #     item = step.input_queue.get()

        #     if is_stop_signal(item):
        #         step.input_queue.close()
        #         step.output_queue.put(item)
        #         break

        #     step.own_time.tick()
        #     output = step.process(item)
        #     step.own_time.tock()

        #     if is_stop_signal(output):
        #         step.input_queue.close()
        #         step.output_queue.put(output)
        #         break

        #     step.total_time.tock()
        #     step.output_queue.put(output)

        # for step in self.sync_steps.values():
        #     step.working = False
        #     step.end()
