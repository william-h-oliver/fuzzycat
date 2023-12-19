import os
import time
from numba import njit, prange
import numpy as np

class FuzzyCat:
    def __init__(self, directory_name, workers = -1, verbose = 1):
        check_directory_name = isinstance(directory_name, str) and directory_name != "" and os.path.exists(directory_name)
        assert check_directory_name, "Parameter 'directory_name' must be a string and must exist!"
        self.directory_name = directory_name

        check_workers = 1 <= workers <= os.cpu_count() or workers == -1
        assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 and <= N_cpu (= {os.cpu_count()})"
        os.environ["OMP_NUM_THREADS"] = f"{workers}" if workers != -1 else f"{os.cpu_count()}"
        self.workers = workers
        self.verbose = verbose

    
    def _printFunction(self, message, returnLine = True):
        if self.verbose:
            if returnLine: print(f"FuzzyCat: {message}\r", end = '')
            else: print(f"FuzzyCat: {message}")
    
    def run(self):
        self._printFunction(f"Started             | {time.strftime('%Y-%m-%d %H:%M:%S')}", returnLine = False)
        begin = time.perf_counter()

        # Get all files in directory
        files = os.listdir(self.directory_name)
        files = [file for file in files if file.endswith(".npy")]

        

        self._totalTime = time.perf_counter() - begin
        self._printFunction(f"Completed           | {time.strftime('%Y-%m-%d %H:%M:%S')}       ", returnLine = False)