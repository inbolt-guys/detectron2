from enum import Enum
from multiprocessing import Queue
import numpy as np
import copy


class BaseQueue:
    def __init__(self):
        self.finished = False

    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.finished = True

    def task_done(self):
        pass

    def clear(self):
        pass

    def close(self):
        self.finished = True


class VoidQueue(BaseQueue):
    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.close()

    def get(self):
        if self.finished:
            return Signal.STOP_IMMEDIATELY

    def qsize(self):
        return 0

    def set(self, item):
        pass


class AsyncQueue(BaseQueue):
    def __init__(self, maxsize=0):
        super().__init__()
        self._queue = Queue(maxsize=maxsize)

    def put(self, item, block=True, timeout=None):
        if self.finished:
            return
        if item is Signal.STOP_IMMEDIATELY:
            self.close()
        else:
            self._queue.put(
                copy.deepcopy(item), block, timeout
            )  # deepcopy otherwise results are duplicated

    def close(self):
        self.finished = True
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.queue.append(Signal.STOP_IMMEDIATELY)
            self._queue.unfinished_tasks = 0
            self._queue.all_tasks_done.notify()
            self._queue.not_full.notify()
            self._queue.not_empty.notify()

    def get(self, block=True, timeout=None):
        if self.finished:
            return Signal.STOP_IMMEDIATELY
        return self._queue.get(block, timeout)

    def clear(self):
        while not self._queue.empty():
            self.get()
            self.task_done()

    def task_done(self):
        if self.finished:
            return
        super().task_done()

    def show(self):
        temp_list = []
        mean_list = []

        # Transfer items to the temporary list
        while not self._queue.empty():
            item = self._queue.get()
            temp_list.append(item)
            if not isinstance(item, Signal):
                if isinstance(item[1], int) or item[1] is None:
                    return
                mean_list.append(np.mean(item[1]["1"]))

        # Display the content of the queue
        print("Queue contents:", mean_list)

        # Put the items back into the queue
        for item in temp_list:
            self._queue.put(item)


class StubQueue(BaseQueue):
    def __init__(self):
        super().__init__()
        self.item = Signal.EMPTY

    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.close()
        assert self.item is Signal.EMPTY
        self.item = item

    def get(self):
        if self.finished:
            return Signal.STOP_IMMEDIATELY
        item = self.item
        self.item = Signal.EMPTY
        assert item is not Signal.EMPTY
        return item


class Signal(Enum):
    OK = 1
    STOP = 2
    STOP_IMMEDIATELY = 3
    ERROR = 4
    EMPTY = 5


def is_stop_signal(item):
    return item is Signal.STOP or item is Signal.STOP_IMMEDIATELY
