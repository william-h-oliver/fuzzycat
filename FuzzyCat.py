import os
import time
from numba import njit
import numpy as np

class FuzzyCat:
    def __init__(self, directoryName, nPoints, minJaccardIndex = 0.5, minExistenceProbability = 0.5, checkpoint = True, workers = -1, verbose = 2):
        check_directoryName = isinstance(directoryName, str) and directoryName != "" and os.path.exists(directoryName)
        assert check_directoryName, "Parameter 'directoryName' must be a string and must exist!"
        if directoryName[-1] != '/': directoryName += '/'
        self.directoryName = directoryName

        check_nPoints = isinstance(nPoints, int) and nPoints > 0
        assert check_nPoints, "Parameter 'nPoints' must be a positive integer!"
        self.nPoints = nPoints

        check_minJaccardIndex = isinstance(minJaccardIndex, (int, float)) and 0 <= minJaccardIndex <= 1
        assert check_minJaccardIndex, "Parameter 'minJaccardIndex' must be a float (or integer) in the interval [0, 1]!"
        self.minJaccardIndex = minJaccardIndex

        check_minExistenceProbability = isinstance(minExistenceProbability, (int, float)) and 0 <= minExistenceProbability <= 1
        assert check_minExistenceProbability, "Parameter 'minExistenceProbability' must be a float (or integer) in the interval [0, 1]!"
        self.minExistenceProbability = minExistenceProbability

        check_checkpoint = isinstance(checkpoint, bool)
        assert check_checkpoint, "Parameter 'checkpoint' must be a boolean!"
        self.checkpoint = checkpoint

        check_workers = 1 <= workers <= os.cpu_count() or workers == -1
        assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 and <= N_cpu (= {os.cpu_count()})"
        os.environ["OMP_NUM_THREADS"] = f"{workers}" if workers != -1 else f"{os.cpu_count()}"
        self.workers = workers
        self.verbose = verbose

    def _printFunction(self, message, returnLine = True, error = False):
        if self.verbose:
            if returnLine: print(f"FuzzyCat: {message}\r", end = '')
            else: print(f"FuzzyCat: {message}")
        if error: print(f"FuzzyCat: [Error] {message}")
    
    def run(self):
        if os.path.exists(self.directoryName + 'Reclusterings/'):
            self._printFunction(f"Started                          | {time.strftime('%Y-%m-%d %H:%M:%S')}", returnLine = False)
            begin = time.perf_counter()

            # Phase 1
            self.computeSimilarities()

            # Phase 2
            self.aggregate()

            # Phase 3
            self.extractFuzzyClusters()

            self._totalTime = time.perf_counter() - begin
            if self.verbose > 1:
                self._printFunction(f"Similarities time                | {100*self._similarityMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
                self._printFunction(f"Aggregation time                 | {100*self._aggregationTime/self._totalTime:.2f}%    ", returnLine = False)
                self._printFunction(f"Fuzzy cluster extraction time    | {100*self._extractFuzzyClustersTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Completed                        | {time.strftime('%Y-%m-%d %H:%M:%S')}       ", returnLine = False)
        else:
            self._printFunction(f"Directory does not contain any cluster files!", error = True)

    def computeSimilarities(self):
        if self.verbose > 1: self._printFunction('Computing similarities...        ')
        start = time.perf_counter()
        
        # Check if arrays have been computed before
        clstFileNamesFileBool = os.path.exists(self.directoryName + 'clusterFileNames.npy')
        sampNumsFileBool = os.path.exists(self.directoryName + 'sampleNumbers.npy')
        clstIDsFileBool = os.path.exists(self.directoryName + 'clusterIDs.npy')
        pairsFileBool = os.path.exists(self.directoryName + 'pairs.npy')
        edgesFileBool = os.path.exists(self.directoryName + 'edges.npy')
        
        # If so, load them, otherwise, compute them
        if clstFileNamesFileBool and sampNumsFileBool and clstIDsFileBool and pairsFileBool and edgesFileBool:
            self.clusterFileNames = np.load(self.directoryName + 'clusterFileNames.npy')
            self.sampleNumbers = np.load(self.directoryName + 'sampleNumbers.npy')
            self.clusterIDs = np.load(self.directoryName + 'clusterIDs.npy')
            self.pairs = np.load(self.directoryName + 'pairs.npy')
            self.edges = np.load(self.directoryName + 'edges.npy')
        else:
            # Get all cluster files in directory
            self.clusterFileNames = np.array([fileName for fileName in os.listdir(self.directoryName + 'Reclusterings/')])

            # Record some cluster information
            n = len(self.clusterFileNames)
            self.sampleNumbers, self.clusterIDs = np.empty(n, dtype = np.uint32), []
            for i, clstFileName in enumerate(self.clusterFileNames):
                clusterInfo = clstFileName.split('.')[0].split('_')
                self.sampleNumbers[i] = np.uint32(clusterInfo[0])
                self.clusterIDs.append(clusterInfo[1])
            self.clusterIDs = np.array(self.clusterIDs)
            
            # Reorder for faster comparisons
            reorder = np.argsort(self.clusterIDs)
            self.clusterFileNames = self.clusterFileNames[reorder]
            self.sampleNumbers = self.sampleNumbers[reorder]
            self.clusterIDs = self.clusterIDs[reorder]

            # Cycle through all pairs of clusters and compute their similarity
            lazyLoad = [False for i in range(n)]
            self.pairs, self.edges = self._initGraph(n)
            for i in range(n):
                for j in range(i + 1, n):
                    # Load clusters
                    if lazyLoad[i] is False: lazyLoad[i] = np.load(self.directoryName + 'Reclusterings/' + self.clusterFileNames[i])
                    if lazyLoad[j] is False: lazyLoad[j] = np.load(self.directoryName + 'Reclusterings/' + self.clusterFileNames[j])

                    # Calculate the similarity between clusters i and j
                    if issubclass(lazyLoad[i].dtype.type, np.integer) and issubclass(lazyLoad[j].dtype.type, np.integer):
                        self.edges[i*(2*n - i - 1)//2 + j - i - 1] = self._jaccardIndex_njit(lazyLoad[i], lazyLoad[j], self.nPoints)
                    elif issubclass(lazyLoad[i].dtype.type, np.floating) and issubclass(lazyLoad[j].dtype.type, np.floating):
                        self.edges[i*(2*n - i - 1)//2 + j - i - 1] = self._weightedJaccardIndex_njit(lazyLoad[i], lazyLoad[j])
                    else:
                        self._printFunction('Clusters from the following files contain different data types!', error = True)
                        print(f"{self.directoryName}Reclusterings/{self.clusterFileNames[i]}")
                        print(f"{self.directoryName}Reclusterings/{self.clusterFileNames[j]}")
                        return
                lazyLoad[i] = False

            # Save arrays
            if self.checkpoint:
                np.save(self.directoryName + 'clusterFileNames.npy', self.clusterFileNames)
                np.save(self.directoryName + 'sampleNumbers.npy', self.sampleNumbers)
                np.save(self.directoryName + 'clusterIDs.npy', self.clusterIDs)
                np.save(self.directoryName + 'pairs.npy', self.pairs)
                np.save(self.directoryName + 'edges.npy', self.edges)

        self._similarityMatrixTime = time.perf_counter() - start
    
    @staticmethod
    @njit()
    def _initGraph(n):
        graphSize = n*(n - 1)//2
        pairs = np.empty((graphSize, 2), dtype = np.uint32) # Might not need to compute this if i and j can be (efficiently) calculated from knowing the index of [i, j]
        for i in range(n):
            for j in range(i + 1, n):
                pairs[i*(2*n - i - 1)//2 + j - i - 1] = [i, j]
        edges = np.zeros(graphSize, dtype = np.float32)
        return pairs, edges

    @staticmethod
    @njit(fastmath = True)
    def _jaccardIndex_njit(c1, c2, nPoints):
        counts_c1 = np.zeros(nPoints, dtype = np.bool_)
        counts_c1[c1] = 1
        intersection = counts_c1[c2].sum()
        return intersection/(c1.size + c2.size - intersection)
    
    @staticmethod
    @njit(fastmath = True)
    def _weightedJaccardIndex_njit(c1, c2):
        return np.minimum(c1, c2).sum()/np.maximum(c1, c2).sum()

    def aggregate(self):
        if self.verbose > 1: self._printFunction('Making fuzzy clusters...        ')
        start = time.perf_counter()
        self.jaccardIndices, self.ordering, self.groups, self.prominences, self.pExistence_groups = self._aggregate_njit(self.pairs, self.edges, self.sampleNumbers)
        del self.pairs, self.edges
        reorder = np.array(sorted(np.arange(self.groups.shape[0]), key = lambda i: [self.groups[i, 0], self.sampleNumbers.size - self.groups[i, 1]]), dtype = np.uint32)
        self.groups, self.prominences, self.pExistence_groups = self.groups[reorder], self.prominences[reorder], self.pExistence_groups[reorder]
        self._aggregationTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _aggregate_njit(pairs, edges, sampleNumbers):
        sortInd = np.argsort(edges)[::-1]
        pairs = pairs[sortInd]
        edges = edges[sortInd]

        # Kruskal's minimum spanning tree + hierarchy tracking
        n_clusters = sampleNumbers.size
        ids = np.full((n_clusters,), n_clusters, dtype = np.uint32)
        jaccardIndices = np.empty(n_clusters, dtype = np.float32)
        count = 0
        aggregations = [[np.uint32(0) for i in range(0)] for i in range(0)]
        emptyIntList = [np.uint32(0) for i in range(0)]
        # For smaller groups
        starts_leq = [np.uint32(0) for i in range(0)]
        sizes_leq = [np.uint32(0) for i in range(0)]
        prominences_leq = [np.float32(0.0) for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # For larger groups
        starts_geq = [np.uint32(0) for i in range(0)]
        sizes_geq = [np.uint32(0) for i in range(0)]
        prominences_geq = [np.float32(0.0) for i in range(0)]

        for pair, edge in zip(pairs, edges):
            if edge == 0.0: break
            id_0, id_1 = ids[pair]
            if id_0 != n_clusters: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == n_clusters: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    jaccardIndices[p_1] = edge
                    aggregations[id_0].append(p_1)
                    sizes_leq[id_0] += 1
                else: # Different groups -> merge groups
                    if sizes_leq[id_0] < sizes_leq[id_1]: id_0, id_1 = id_1, id_0
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    # Track complementary group
                    starts_geq[id_1] = starts_leq[id_0]
                    sizes_geq[id_1] = sizes_leq[id_0]
                    prominences_geq[id_1] = prominences_leq[id_0] - edge
                    # Merge
                    starts_leq[id_1] += sizes_leq[id_0]
                    sizes_leq[id_0] += sizes_leq[id_1]
                    prominences_leq[id_0] = max(prominences_leq[id_0], prominences_leq[id_1])
                    prominences_leq[id_1] -= edge
                    children[id_0].append(id_1)
            elif id_1 == n_clusters: # Neither are aggregated
                ids[pair] = count
                jaccardIndices[pair] = edge
                count += 1
                aggregations.append([pair[0], pair[1]])
                # Create group
                starts_leq.append(0)
                sizes_leq.append(2)
                prominences_leq.append(edge)
                children.append([np.uint32(0) for i in range(0)])
                # Track complementary group
                starts_geq.append(0)
                sizes_geq.append(0)
                prominences_geq.append(0.0)
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                jaccardIndices[p_0] = edge
                aggregations[id_1].append(p_0)
                sizes_leq[id_1] += 1
                prominences_leq[id_1] = max(prominences_leq[id_1], edge)

        # Merge separate aggregations in order of decreasing size
        aggArr = np.unique(ids)
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, make it so.
            sortedAggregations = sorted(zip([sizes_leq[id_i] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList
                # Track larger group
                starts_geq[id_leq] = starts_leq[id_final]
                sizes_geq[id_leq] = sizes_leq[id_final]
                prominences_geq[id_leq] = prominences_leq[id_final]
                # Merge
                starts_leq[id_leq] += sizes_leq[id_final]
                sizes_leq[id_final] += size_leq
                children[id_final].append(id_leq)
        emptyIntArr = np.empty(0, dtype = np.uint32)
        ids = emptyIntArr
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint32)
        aggregations[id_final] = emptyIntList

        # Finalise groups and correct for noise
        activeGroups = [id_final]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = starts_leq[id_leq]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    starts_leq[childID] += startAdjust
                    starts_geq[childID] += startAdjust
                    if id_geq > 0: prominences_geq[childID] -= np.sqrt(noise/id_geq)
                    noise += prominences_leq[childID]**2
                prominences_leq[id_leq] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Lists to Arrays
        starts_leq = np.array(starts_leq, dtype = np.uint32)
        sizes_leq = np.array(sizes_leq, dtype = np.uint32)
        prominences_leq = np.array(prominences_leq, dtype = np.float32)
        starts_geq = np.array(starts_geq, dtype = np.uint32)
        sizes_geq = np.array(sizes_geq, dtype = np.uint32)
        prominences_geq = np.array(prominences_geq, dtype = np.float32)

        # Clean and reorder arrays
        starts_leq = np.delete(starts_leq, id_final)
        groups_leq = np.column_stack((starts_leq, starts_leq + np.delete(sizes_leq, id_final)))
        starts_leq, sizes_leq = emptyIntArr, emptyIntArr
        prominences_leq = np.delete(prominences_leq, id_final)
        starts_geq = np.delete(starts_geq, id_final)
        groups_geq = np.column_stack((starts_geq, starts_geq + np.delete(sizes_geq, id_final)))
        starts_geq, sizes_geq = emptyIntArr, emptyIntArr
        prominences_geq = np.delete(prominences_geq, id_final)

        # Combine arrays
        groups = np.vstack((groups_leq, groups_geq))
        prominences = np.concatenate((prominences_leq, prominences_geq))

        # Calculate existence probabilities
        pExistence_groups = np.empty(groups.shape[0], dtype = np.float32)
        for i, g in enumerate(groups):
            pExistence_groups[i] = np.unique(sampleNumbers[ordering[g[0]:g[1]]]).size
        pExistence_groups /= np.unique(sampleNumbers).size

        return jaccardIndices, ordering, groups, prominences, pExistence_groups

    def extractFuzzyClusters(self):
        if self.verbose > 1: self._printFunction('Assigning probabilities...        ')
        start = time.perf_counter()

        # Extract fuzzy clusters
        self.fuzzyGroups, self.pExistence = self._extractFuzzyClusters_njit(self.groups, self.prominences, self.pExistence_groups, self.minJaccardIndex, self.minExistenceProbability)
        
        # Cycle through all clusters and adjust membership probabilities of each point
        self.pMembership = np.zeros((self.fuzzyGroups.shape[0], self.nPoints))
        if self.fuzzyGroups.size:
            for i, clstFileName_i in enumerate(self.clusterFileNames):
                cluster_i = np.load(self.directoryName + 'Reclusterings/' + clstFileName_i)
                self._pMembershipUpdate_njit(self.fuzzyGroups, self.ordering, self.pMembership, cluster_i, i)
            self.pMembership /= (self.fuzzyGroups[:, 1] - self.fuzzyGroups[:, 0]).reshape(-1, 1)
        
        self._extractFuzzyClustersTime = time.perf_counter() - start
    
    @staticmethod
    @njit()
    def _extractFuzzyClusters_njit(groups, prominences, pExistence_groups, minJaccardIndex, minExistenceProbability):
        sl = np.logical_and(prominences > minJaccardIndex, pExistence_groups > minExistenceProbability)
        fuzzyGroups = groups[sl]
        pExistence = pExistence_groups[sl]
            
        # Keep only those groups that are the smallest in their cascade
        sl = np.zeros(fuzzyGroups.shape[0], dtype = np.bool_)
        cascade_starts_unique = np.unique(fuzzyGroups[:, 0])
        for cascade_start in cascade_starts_unique:
            sl[np.where(fuzzyGroups[:, 0] == cascade_start)[0][-1]] = 1
        return fuzzyGroups[sl], pExistence[sl]
    
    @staticmethod
    @njit()
    def _pMembershipUpdate_njit(fuzzyGroups, ordering, pMembership, cluster_i, i):
        for j, fuzGrp in enumerate(fuzzyGroups):
            overlapBool = i in ordering[fuzGrp[0]:fuzGrp[1]]
            if overlapBool: pMembership[j, cluster_i] += 1