import threading

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
