import os
import pickle
import time
from numba import njit, prange
import numpy as np

class FuzzyCat:
    def __init__(self, directoryName, workers = -1, verbose = 1):
        check_directoryName = isinstance(directoryName, str) and directoryName != "" and os.path.exists(directoryName)
        assert check_directoryName, "Parameter 'directory_name' must be a string and must exist!"
        self.directoryName = directoryName

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

        # Phase 1
        self.computeDistanceMatrices()

        # Phase 2
        self.scoreSimilarClusters()

        # Phase 3
        self.fuzzyAssignment()

        self._totalTime = time.perf_counter() - begin
        if self.verbose > 1:
            self._printFunction(f"Distance matrices time | {100*self._distanceMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Scoring clusters time  | {100*self._distanceMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Fuzzy assignment time  | {100*self._distanceMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
        self._printFunction(f"Completed              | {time.strftime('%Y-%m-%d %H:%M:%S')}       ", returnLine = False)

    def computeDistanceMatrices(self):
        if self.verbose > 1: self._printFunction('Computing distance matrices...        ')
        start = time.perf_counter()

        # Get all files in directory
        files = sorted([file for file in os.listdir(self.directoryName) if file.endswith(".clusters")])

        # Create new folder to store distance matrices
        distanceMatricesFolder = os.path.join(self.directoryName, "distance_matrices") 
        if not os.path.exists(distanceMatricesFolder):
            os.makedirs(distanceMatricesFolder)

        # Cycle through all files and compute the pairwise distance matrix between the clusters in each of them
        for i, file_i in enumerate(files):
            # Load the clusters from file_i
            with open(os.path.join(self.directoryName, file_i), "rb") as loadFile:
                clusters_i = pickle.load(loadFile)
            for j, file_j in enumerate(files[i + 1:]):
                # Change 'j' according to 'i' 
                j += i + 1

                # If distance matrix not already created...
                distanceMatrixFile = os.path.join(distanceMatricesFolder, f'distances_{i}-{j}.npy')
                if os.path.exists(distanceMatrixFile):
                    # Load the clusters from file_j
                    with open(os.path.join(self.directory_name, file_i), "rb") as loadFile:
                        clusters_j = pickle.load(loadFile)

                    # Calculate the distances between the two clusterings
                    distanceMatrix = self._calculateDistanceMatrix(clusters_i, clusters_j)

                    # Save the distances
                    np.save(distanceMatrixFile, distanceMatrix)
        
        self._distanceMatrixTime = time.perf_counter() - start

    def _calculateDistanceMatrix(self, clusters_i, clusters_j):
        distanceMatrix = np.empty((len(clusters_i), len(clusters_j)), dtype = np.float32)
        for i, cluster_i in enumerate(clusters_i):
            for j, cluster_j in enumerate(clusters_j):
                distanceMatrix[i, j] = self._jaccardDistance_njit(cluster_i, cluster_j)
        return distanceMatrix
    
    @staticmethod
    @njit(parallel = True)
    def _jaccardDistance_njit(c1, c2):
        intersection = 0
        for i in prange(c1.size):
            for j in prange(c2.size):
                if c1[i] == c2[j]:
                    intersection += 1
                    break
        return 1 - intersection/(c1.size + c2.size - intersection)

    def scoreSimilarClusters(self):
        pass

    def fuzzyAssignment(self):
        pass