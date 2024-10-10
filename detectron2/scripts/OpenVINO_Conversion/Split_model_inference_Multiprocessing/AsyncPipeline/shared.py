import multiprocessing as mp
from .queue import Signal, is_stop_signal
import numpy as np
import time


class SharedLocation:
    def __init__(self, feature_maps_shape) -> None:

        # feature maps
        self._feature_map_1 = mp.Array("f", np.random.random(feature_maps_shape[0]).flatten())
        self._feature_map_2 = mp.Array("f", np.random.random(feature_maps_shape[1]).flatten())
        self._feature_map_3 = mp.Array("f", np.random.random(feature_maps_shape[2]).flatten())
        self._feature_map_4 = mp.Array("f", np.random.random(feature_maps_shape[3]).flatten())
        self._feature_map_5 = mp.Array("f", np.random.random(feature_maps_shape[4]).flatten())

        self._shared_np_feature_map_1 = np.frombuffer(
            self._feature_map_1.get_obj(), dtype=np.float32
        ).reshape(feature_maps_shape[0])
        self._shared_np_feature_map_2 = np.frombuffer(
            self._feature_map_2.get_obj(), dtype=np.float32
        ).reshape(feature_maps_shape[1])
        self._shared_np_feature_map_3 = np.frombuffer(
            self._feature_map_3.get_obj(), dtype=np.float32
        ).reshape(feature_maps_shape[2])
        self._shared_np_feature_map_4 = np.frombuffer(
            self._feature_map_4.get_obj(), dtype=np.float32
        ).reshape(feature_maps_shape[3])
        self._shared_np_feature_map_5 = np.frombuffer(
            self._feature_map_5.get_obj(), dtype=np.float32
        ).reshape(feature_maps_shape[4])

        # stop signal event
        self._stop_event = mp.Event()

        # locks
        self._write_lock = mp.Lock()
        self._read_lock = mp.Lock

        # events
        self._been_read = mp.Event()  # True if decoder has read feature maps, False otherwise
        self._been_read.set()  # So the backbone can write in the memory at the beginning
        self._new_data = mp.Event()  # initally false

    @property
    def feature_maps(self):
        return [
            self._shared_np_feature_map_1,
            self._shared_np_feature_map_2,
            self._shared_np_feature_map_3,
            self._shared_np_feature_map_4,
            self._shared_np_feature_map_5,
        ]

    def set(self, new_feature_maps):
        self._been_read.wait()
        with self._write_lock:
            if is_stop_signal(new_feature_maps):
                self._stop_event.set()
            else:
                self._shared_np_feature_map_1[:, :, :, :] = new_feature_maps[0]
                self._shared_np_feature_map_2[:, :, :, :] = new_feature_maps[1]
                self._shared_np_feature_map_3[:, :, :, :] = new_feature_maps[2]
                self._shared_np_feature_map_4[:, :, :, :] = new_feature_maps[3]
                self._shared_np_feature_map_5[:, :, :, :] = new_feature_maps[4]

            # This data has not been read by the decoder
            self._been_read.clear()

            # New data is now available
            self._new_data.set()

    def get(self):
        self._new_data.wait()  # Wait until the backbone do one inference
        with self._write_lock:
            if self._stop_event.is_set():
                feature_maps = Signal.STOP
            else:
                feature_maps = self.feature_maps

            # We inform the backbone that he can write now
            self._been_read.set()
            # Not new data since we just pop it
            self._new_data.clear()

        return feature_maps
