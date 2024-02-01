import os
import pickle
import time
from numba import njit, prange
import numpy as np

class FuzzyCat:
    def __init__(self, directoryName, n_points, minJaccardIndex = 0.0, workers = -1, verbose = 2):
        check_directoryName = isinstance(directoryName, str) and directoryName != "" and os.path.exists(directoryName)
        assert check_directoryName, "Parameter 'directory_name' must be a string and must exist!"
        self.directoryName = directoryName

        check_n_points = isinstance(n_points, int) and n_points > 0
        assert check_n_points, "Parameter 'n_points' must be a positive integer!"
        self.n_points = n_points

        check_minJaccardIndex = isinstance(minJaccardIndex, (int, float)) and 0 <= minJaccardIndex <= 1
        assert check_minJaccardIndex, "Parameter 'minJaccardIndex' must be a float (or integer) in the interval [0, 1]!"
        self.minJaccardIndex = minJaccardIndex

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
        self._printFunction(f"Started                | {time.strftime('%Y-%m-%d %H:%M:%S')}", returnLine = False)
        begin = time.perf_counter()

        # Phase 1
        self.computeSimilarityMatrices()

        # Phase 2
        self.aggregate()

        # Phase 3
        self.extractFuzzyClusters()

        self._totalTime = time.perf_counter() - begin
        if self.verbose > 1:
            self._printFunction(f"Similarity matrices time | {100*self._similarityMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Scoring clusters time    | {100*self._fuzzyClustersTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Fuzzy assignment time    | {100*self._similarityMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
        self._printFunction(f"Completed              | {time.strftime('%Y-%m-%d %H:%M:%S')}       ", returnLine = False)

    def computeSimilarityMatrices(self):
        if self.verbose > 1: self._printFunction('Computing similarity matrices...        ')
        start = time.perf_counter()

        # Get all files in directory
        files = sorted([file for file in os.listdir(self.directoryName) if file.endswith(".clusters")])

        # Create new folder to store similarity matrices
        similarityMatricesFolder = os.path.join(self.directoryName, "similarity_matrices")
        if not os.path.exists(similarityMatricesFolder):
            os.makedirs(similarityMatricesFolder)

        # Track indices of clusters across different files
        self.fileIndices, self.clusterIndices = [], []

        # Cycle through all files and compute the pairwise similarity matrix between the clusters in each of them
        for i, file_i in enumerate(files):
            # Load the clusters from file_i
            with open(os.path.join(self.directoryName, file_i), "rb") as loadFile:
                clusters_i = pickle.load(loadFile)

            self.fileIndices += [i for _ in range(len(clusters_i))]
            self.clusterIndices += [len(self.clusterIndices) + j for j in range(len(clusters_i))]

            for j, file_j in enumerate(files[i + 1:]):
                # Change 'j' according to 'i' 
                j += i + 1

                # If similarity matrix not already created...
                similarityMatrixFile = os.path.join(similarityMatricesFolder, f'similarities_{i}-{j}.npy')
                if not os.path.exists(similarityMatrixFile):
                    # Load the clusters from file_j
                    with open(os.path.join(self.directoryName, file_j), "rb") as loadFile:
                        clusters_j = pickle.load(loadFile)

                    # Calculate the similarities between the two clusterings
                    similarityMatrix = self._calculateSimilarityMatrix(clusters_i, clusters_j)

                    # Save the similarities
                    np.save(similarityMatrixFile, similarityMatrix)
        
        # Convert lists to arrays
        self.fileIndices, self.clusterIndices = np.array(self.fileIndices), np.array(self.clusterIndices)

        self._similarityMatrixTime = time.perf_counter() - start

    def _calculateSimilarityMatrix(self, clusters_i, clusters_j):
        # Cycle through each cluster in the two clusterings and calculate the similarity between them
        similarityMatrix = np.empty((len(clusters_i), len(clusters_j)), dtype = np.float64)
        for i, cluster_i in enumerate(clusters_i):
            for j, cluster_j in enumerate(clusters_j):
                similarityMatrix[i, j] = self._jaccardIndex_njit(cluster_i, cluster_j)
        return similarityMatrix
    
    @staticmethod
    @njit(fastmath = True)
    def _jaccardIndex_njit(c1, c2):
        c1_sorted = np.sort(c1)
        c2_sorted = np.sort(c2)
        intersection = 0
        i, j = 0, 0
        while i < c1.size and j < c2.size:
            bool_intersec = c1_sorted[i] == c2_sorted[j]
            intersection += bool_intersec
            bool_c1_smaller = c1_sorted[i] < c2_sorted[j]
            i += bool_intersec + bool_c1_smaller
            j += ~bool_c1_smaller
        return intersection/(c1.size + c2.size - intersection)


    def aggregate(self):
        if self.verbose > 1: self._printFunction('Making fuzzy clusters...        ')
        start = time.perf_counter()

        similarityMatricesFolder = os.path.join(self.directoryName, "similarity_matrices") 
        if not os.path.exists(similarityMatricesFolder):
            self._printFunction('[Error] Similarity matrices folder does not exist!')
        else:
            # Get all similarity matrices in directory
            fileNames = sorted([fileName for fileName in os.listdir(similarityMatricesFolder) if fileName.startswith("similarities_") and fileName.endswith(".npy")])

            # Cycle through all files, load similarity matrix, and order the clusters
            pairs, edges = np.empty((0, 2), dtype = np.uint32), np.empty(0, dtype = np.float64)
            for fileName in fileNames:
                # Load similarity matrix
                similarityMatrixFile = os.path.join(similarityMatricesFolder, fileName)
                similarityMatrix = np.load(similarityMatrixFile)

                # Retrieve original file numbers
                i, j = map(int, fileName.split("_")[1].split(".")[0].split("-"))

                # Add new pairs and edges
                pairs, edges = self._newPairs_njit(self.fileIndices, self.clusterIndices, i, j, similarityMatrix, pairs, edges)

            # Aggregate clusters
            self.jaccardIndices, self.ordering, self.groups, self.prominences, self.groups_comp, self.prominences_comp = self._aggregate_njit(pairs, edges, self.clusterIndices.size)
        self._fuzzyClustersTime = time.perf_counter() - start
    
    @staticmethod
    @njit()
    def _newPairs_njit(fileIndices, clusterIndices, i, j, similarityMatrix, pairs, edges):
        clstInd_i = clusterIndices[fileIndices == i]
        clstInd_j = clusterIndices[fileIndices == j]
        pairs_ij = np.where(similarityMatrix > 0)
        newPairs = np.empty((pairs_ij[0].size, 2), dtype = np.uint32)
        newEdges = np.zeros(pairs_ij[0].size, dtype = np.float64)
        for k, pair in enumerate(np.column_stack(pairs_ij)):
            p_0, p_1 = pair
            newPairs[k, 0] = clstInd_i[p_0]
            newPairs[k, 1] = clstInd_j[p_1]
            newEdges[k] = similarityMatrix[p_0, p_1]
        return np.concatenate((pairs, newPairs), axis = 0), np.concatenate((edges, newEdges))

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _aggregate_njit(pairs, edges, n_clusters):
        sortInd = np.argsort(edges)[::-1]
        pairs = pairs[sortInd]
        edges = edges[sortInd]

        # Kruskal's minimum spanning tree + hierarchy tracking
        ids = np.full((n_clusters,), n_clusters, dtype = np.uint32)
        jaccardIndices = np.empty(n_clusters, dtype = np.float64)
        count = 0
        aggregations = [[np.uint32(0) for i in range(0)] for i in range(0)]
        emptyIntList = [np.uint32(0) for i in range(0)]
        # For subgroups
        starts = [np.uint32(0) for i in range(0)]
        sizes = [np.uint32(0) for i in range(0)]
        prominences = [np.float64(0.0) for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # For complementary groups
        starts_comp = [np.uint32(0) for i in range(0)]
        sizes_comp = [np.uint32(0) for i in range(0)]
        prominences_comp = [np.float64(0.0) for i in range(0)]

        for pair, edge in zip(pairs, edges):
            id_0, id_1 = ids[pair]
            if id_0 != n_clusters: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == n_clusters: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    jaccardIndices[p_1] = edge
                    aggregations[id_0].append(p_1)
                    sizes[id_0] += 1
                else: # Different groups -> merge groups
                    if sizes[id_0] < sizes[id_1]: id_0, id_1 = id_1, id_0
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = edge
                    # Track complementary group
                    starts_comp[id_1] = starts[id_0]
                    sizes_comp[id_1] = sizes[id_0]
                    prominences_comp[id_1] = prominences[id_0] - currLogRho
                    # Merge
                    starts[id_1] += sizes[id_0]
                    sizes[id_0] += sizes[id_1]
                    prominences[id_0] = max(prominences[id_0], prominences[id_1])
                    prominences[id_1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == n_clusters: # Neither are aggregated
                ids[pair] = count
                jaccardIndices[pair] = edge
                count += 1
                aggregations.append([pair[0], pair[1]])
                # Create group
                starts.append(0)
                sizes.append(2)
                prominences.append(edge)
                children.append([np.uint32(0) for i in range(0)])
                # Track complementary group
                starts_comp.append(0)
                sizes_comp.append(0)
                prominences_comp.append(0.0)
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                jaccardIndices[p_0] = edge
                aggregations[id_1].append(p_0)
                sizes[id_1] += 1
                prominences[id_1] = max(prominences[id_1], edge)

        # Merge separate aggregations in order of decreasing size
        aggArr = np.unique(ids)
        if aggArr.size == 1: id_0 = aggArr[0]
        else: # If points were not all aggregated together, make it so.
            sortedAggregations = sorted(zip([sizes[id_i] for id_i in aggArr], aggArr))
            _, id_0 = sortedAggregations[-1]
            for size_i, id_i in sortedAggregations[-2::-1]:
                aggregations[id_0].extend(aggregations[id_i])
                aggregations[id_i] = emptyIntList
                # Track complementary group
                starts_comp[id_i] = starts[id_0]
                sizes_comp[id_i] = sizes[id_0]
                prominences_comp[id_i] = prominences[id_0]
                # Merge
                starts[id_i] += sizes[id_0]
                sizes[id_0] += size_i
                children[id_0].append(id_i)
        emptyIntArr = np.empty(0, dtype = np.uint32)
        ids = emptyIntArr
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_0], dtype = np.uint32)
        aggregations[id_0] = emptyIntList

        # Finalise groups and correct for noise
        activeGroups = [id_i for id_i in children[id_0]]
        while activeGroups:
            id_i = activeGroups.pop()
            childIDs = children[id_i]
            if childIDs:
                startAdjust = starts[id_i]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_j, childID in enumerate(childIDs):
                    starts[childID] += startAdjust
                    starts_comp[childID] += startAdjust
                    if id_j > 0: prominences_comp[childID] -= np.sqrt(noise/id_j)
                    noise += prominences[childID]**2
                prominences[id_i] -= np.sqrt(noise/(id_j + 1))
                children[id_i] = emptyIntList

        # Lists to Arrays
        starts = np.array(starts, dtype = np.uint32)
        sizes = np.array(sizes, dtype = np.uint32)
        prominences = np.array(prominences, dtype = np.float64)
        starts_comp = np.array(starts_comp, dtype = np.uint32)
        sizes_comp = np.array(sizes_comp, dtype = np.uint32)
        prominences_comp = np.array(prominences_comp, dtype = np.float64)

        # Clean arrays
        starts = np.delete(starts, id_0)
        groups = np.column_stack((starts, starts + np.delete(sizes, id_0)))
        starts, sizes = emptyIntArr, emptyIntArr
        prominences = np.delete(prominences, id_0)
        starts_comp = np.delete(starts_comp, id_0)
        groups_comp = np.column_stack((starts_comp, starts_comp + np.delete(sizes_comp, id_0)))
        starts_comp, sizes_comp = emptyIntArr, emptyIntArr
        prominences_comp = np.delete(prominences_comp, id_0)

        # Reorder arrays
        reorder = groups[:, 0].argsort()
        return jaccardIndices, ordering, groups[reorder], prominences[reorder], groups_comp[reorder], prominences_comp[reorder]

    def extractFuzzyClusters(self):
        if self.verbose > 1: self._printFunction('Assigning probabilities...        ')
        start = time.perf_counter()

        # Extract fuzzy clusters
        sl = self.prominences > self.minJaccardIndex
        self.fuzzyGroups = self.groups[sl]
        sl = np.logical_and(sl, self.prominences_comp > self.minJaccardIndex)
        fuzzyGroups_comp = self.groups_comp[sl]
            
        # Keep only those complementary groups that are the smallest in their cascade
        sl = np.zeros(fuzzyGroups_comp.shape[0], dtype = np.bool_)
        cascade_starts_unique = np.unique(fuzzyGroups_comp[:, 0])
        for cascade_start in cascade_starts_unique:
            sl[np.where(fuzzyGroups_comp[:, 0] == cascade_start)[0][0]] = 1

        self.fuzzyGroups = np.vstack((self.fuzzyGroups, fuzzyGroups_comp[sl]))
        reorder = np.array(sorted(np.arange(self.fuzzyGroups.shape[0]), key = lambda i: [self.fuzzyGroups[i, 0], self.clusterIndices.size - self.fuzzyGroups[i, 1]]), dtype = np.uint32)
        self.fuzzyGroups = self.fuzzyGroups[reorder]

        # Initialise arrays
        self.pMembership = np.zeros((self.fuzzyGroups.shape[0], self.n_points), dtype = np.float64)
        whichClusters = [np.sort(self.ordering[self.fuzzyGroups[i, 0]:self.fuzzyGroups[i, 1]]) for i in range(self.fuzzyGroups.shape[0])]

        # Get all files in directory
        fileNames = sorted([fileName for fileName in os.listdir(self.directoryName) if fileName.endswith(".clusters")])

        # Cycle through all files and compute the fuzzy assignment of each point
        for i, fileName in enumerate(fileNames):
            # Load the clusters from file_i
            with open(os.path.join(self.directoryName, fileName), "rb") as loadFile:
                clusters_i = pickle.load(loadFile)

            # Adjust membership probabilities for each cluster in fileName
            self.pMembership = self._updateMembershipProbabilities_njit(self.pMembership, whichClusters, clusters_i, self.clusterIndices[self.fileIndices == i])
        
        n_occurrences = self.fuzzyGroups[:, 1] - self.fuzzyGroups[:, 0]
        self.pMembership /= n_occurrences.reshape(-1, 1)
        self.pExistence = n_occurrences/len(fileNames)
        
        self._fuzzyAssignmentTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True)
    def _updateMembershipProbabilities_njit(pMembership, whichClusters, clusters_i, clusterIndices_i):
        for j in prange(len(whichClusters)):
            whichClusters_j = whichClusters[j]
            k, l = 0, 0
            while k < clusterIndices_i.size and l < whichClusters_j.size:
                bool_intersec = clusterIndices_i[k] == whichClusters_j[l]
                pMembership[j, clusters_i[k]] += bool_intersec
                bool_arrk_smaller = clusterIndices_i[k] < whichClusters_j[l]
                k += bool_intersec + bool_arrk_smaller
                l += ~bool_arrk_smaller
        return pMembership