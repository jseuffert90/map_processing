import threading
import multiprocessing

class AtomicInteger():
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()

    def getAndInc(self):
        try:
            self._lock.acquire()
            ret_val = self._value
            self._value += 1
            return ret_val
        finally:
            self._lock.release()

class AtomicIntegerProc():
    def __init__(self, value=0):
        self._value = multiprocessing.Value('i', value)
        self._lock = threading.Lock()

    def getAndInc(self):
        with self._value.get_lock():
            ret_val = self._value.value
            self._value.value += 1
            return ret_val
