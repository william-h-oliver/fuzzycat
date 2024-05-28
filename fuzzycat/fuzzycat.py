import os
import time
import numpy as np
from numba import njit

class FuzzyCat:
    def __init__(self, directoryName, nSamples, nPoints, minJaccardIndex = 0.5, minStability = 0.5, checkpoint = False, workers = -1, verbose = 2):
        check_directoryName = isinstance(directoryName, str) and directoryName != "" and os.path.exists(directoryName)
        assert check_directoryName, "Parameter 'directoryName' must be a string and must exist!"
        if directoryName[-1] != '/': directoryName += '/'
        self.directoryName = directoryName

        check_nSamples = issubclass(type(nSamples), (int, np.integer)) and nSamples > 0
        assert check_nSamples, "Parameter 'nSamples' must be a positive integer!"
        self.nSamples = nSamples

        check_nPoints = issubclass(type(nPoints), (int, np.integer)) and nPoints > 0
        assert check_nPoints, "Parameter 'nPoints' must be a positive integer!"
        self.nPoints = nPoints

        check_minJaccardIndex = issubclass(type(minJaccardIndex), (int, float, np.integer, np.floating)) and 0 <= minJaccardIndex <= 1
        assert check_minJaccardIndex, "Parameter 'minJaccardIndex' must be a float (or integer) in the interval [0, 1]!"
        self.minJaccardIndex = minJaccardIndex

        check_minStability = issubclass(type(minStability), (int, float, np.integer, np.floating)) and 0 <= minStability <= 1
        assert check_minStability, "Parameter 'minExistenceProbability' must be a float (or integer) in the interval [0, 1]!"
        self.minStability = minStability

        check_checkpoint = isinstance(checkpoint, bool)
        assert check_checkpoint, "Parameter 'checkpoint' must be a boolean!"
        self.checkpoint = checkpoint

        check_workers = issubclass(type(workers), (int, np.integer)) and (1 <= workers <= os.cpu_count() or workers == -1)
        assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 and <= N_cpu (= {os.cpu_count()})"
        os.environ["OMP_NUM_THREADS"] = f"{workers}" if workers != -1 else f"{os.cpu_count()}"
        self.workers = workers
        self.verbose = verbose

    def _printFunction(self, message, returnLine = True):
        if self.verbose:
            if returnLine: print(f"FuzzyCat: {message}\r", end = '')
            else: print(f"FuzzyCat: {message}")
    
    def run(self):
        assert os.path.exists(self.directoryName + 'Clusters/'), f"Directory {self.directoryName + 'Clusters/'} does not exist!"
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

    def computeSimilarities(self):
        if self.verbose > 1: self._printFunction('Computing similarities...        ')
        start = time.perf_counter()
        
        # Check if arrays have been computed before
        clstFileNamesFileBool = os.path.exists(self.directoryName + 'clusterFileNames.npy')
        pairsFileBool = os.path.exists(self.directoryName + 'pairs.npy')
        edgesFileBool = os.path.exists(self.directoryName + 'edges.npy')
        
        # If so, load them, otherwise, compute them
        if clstFileNamesFileBool and pairsFileBool and edgesFileBool:
            self.clusterFileNames = np.load(self.directoryName + 'clusterFileNames.npy')
            self.pairs = np.load(self.directoryName + 'pairs.npy')
            self.edges = np.load(self.directoryName + 'edges.npy')
        else:
            # Get all cluster files in directory
            self.clusterFileNames = np.array([fileName for fileName in os.listdir(self.directoryName + 'Clusters/') if fileName.endswith('.npy')])

            # Cycle through all pairs of clusters and compute their similarity
            n_clusters = self.clusterFileNames.size
            lazyLoad = [False for i in range(n_clusters)]
            dTypes = np.zeros(n_clusters, dtype = np.int8)
            self.pairs, self.edges = self._initGraph(n_clusters)
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # Load clusters
                    if lazyLoad[i] is False: lazyLoad[i], dTypes[i] = self.readClusterFile(self.directoryName + 'Clusters/' + self.clusterFileNames[i])
                    if lazyLoad[j] is False: lazyLoad[j], dTypes[j] = self.readClusterFile(self.directoryName + 'Clusters/' + self.clusterFileNames[j])

                    # Calculate the similarity between clusters i and j
                    if dTypes[i] == dTypes[j] == 1: self.edges[i*(2*n_clusters - i - 1)//2 + j - i - 1] = self._jaccardIndex_njit(lazyLoad[i], lazyLoad[j], self.nPoints)
                    else: self.edges[i*(2*n_clusters - i - 1)//2 + j - i - 1] = self._weightedJaccardIndex_njit(lazyLoad[i].astype(np.float64), lazyLoad[j].astype(np.float64))

            # Save arrays
            if self.checkpoint:
                np.save(self.directoryName + 'clusterFileNames.npy', self.clusterFileNames)
                np.save(self.directoryName + 'pairs.npy', self.pairs)
                np.save(self.directoryName + 'edges.npy', self.edges)

        self._similarityMatrixTime = time.perf_counter() - start
    
    def readClusterFile(self, fileName):
        cluster = np.load(fileName)
        if issubclass(cluster.dtype.type, np.integer): dType = 1
        elif issubclass(cluster.dtype.type, np.floating): dType = 2
        else: assert False, f"Cluster from file '{fileName}' is of {cluster.dtype.type} data type (must be integer or floating)!"
        return cluster, dType
    
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
        self._sampleNumbers = np.array([np.uint32(splitFileName[0]) for splitFileName in np.char.split(self.clusterFileNames, '_', 1)])
        self.jaccardIndices, self.ordering, self.groups, self.prominences, self.stabilitiesGroups = self._aggregate_njit(self.pairs, self.edges, self._sampleNumbers, self.nSamples)
        del self.pairs, self.edges
        reorder = np.array(sorted(np.arange(self.groups.shape[0]), key = lambda i: [self.groups[i, 0], self.clusterFileNames.size - self.groups[i, 1]]), dtype = np.uint32)
        self.groups, self.prominences, self.stabilitiesGroups = self.groups[reorder], self.prominences[reorder], self.stabilitiesGroups[reorder]
        self._aggregationTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _aggregate_njit(pairs, edges, sampleNumbers, nSamples):
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
            #if edge == 0.0: break
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
        stabilitiesGroups = np.empty(groups.shape[0], dtype = np.int32)
        for i, g in enumerate(groups):
            stabilitiesGroups[i] = np.unique(sampleNumbers[ordering[g[0]:g[1]]]).size
        stabilitiesGroups = stabilitiesGroups.astype(np.float64)/nSamples

        return jaccardIndices, ordering, groups, prominences, stabilitiesGroups

    def extractFuzzyClusters(self):
        if self.verbose > 1: self._printFunction('Assigning probabilities...        ')
        start = time.perf_counter()

        # Extract fuzzy clusters and setup memberships and fuzzy hierarchy arrays
        self.fuzzyClusters, self.stabilities, self.memberships, self._hierarchyCorrection, self.fuzzyHierarchy = self._extractFuzzyClusters_njit(self.groups, self.prominences, self.stabilitiesGroups, self.minJaccardIndex, self.minStability, self.nPoints)

        if self.fuzzyClusters.size:
            # Setup hierarchy information
            whichFuzzyCluster, sampleWeights = self._setupHierarchyInformation_njit(self.ordering, self.fuzzyClusters, self._sampleNumbers, self.nSamples)
            baseNames = np.char.add(np.char.rstrip(self.clusterFileNames, '.npy'), '-')

            # Cycle through all clusters and adjust membership probabilities of each point
            for i, clstFileName_i in enumerate(self.clusterFileNames):
                whichFC_cluster = whichFuzzyCluster[i]
                if whichFC_cluster != -1:
                    # Load cluster
                    clusterArr = np.load(self.directoryName + 'Clusters/' + clstFileName_i)

                    # Find the parent of cluster 'i' from within the same sample
                    whichFC_parents = np.unique(whichFuzzyCluster[np.char.startswith(baseNames[i], baseNames)])

                    # Update memberships
                    thisSampleNumber = self._sampleNumbers[i]
                    sampleWeights_i = sampleWeights[thisSampleNumber]
                    self._updateMemberships_njit(self.memberships, self._hierarchyCorrection, self.fuzzyHierarchy, clusterArr, whichFC_cluster, whichFC_parents, sampleWeights_i)
            # Normalise memberships
            normFactor = self.nSamples*self.stabilities.reshape(-1, 1)
            self.memberships /= normFactor
            self._hierarchyCorrection /= normFactor
            self.memberships_flat = self.memberships - self._hierarchyCorrection
            self.fuzzyHierarchy /= normFactor.T
        else: self.memberships_flat = np.zeros((0, self.nPoints))
        
        self._extractFuzzyClustersTime = time.perf_counter() - start
    
    @staticmethod
    @njit()
    def _extractFuzzyClusters_njit(groups, prominences, stabilitiesGroups, minJaccardIndex, minStability, nPoints):
        sl = np.logical_and(prominences > minJaccardIndex, stabilitiesGroups > minStability) #np.sqrt(prominences*stabilitiesGroups) > minJaccardIndex
        fuzzyClusters = groups[sl]
        stabilities = stabilitiesGroups[sl]

        # Keep only those groups that are the smallest in their cascade
        sl = np.zeros(fuzzyClusters.shape[0], dtype = np.bool_)
        cascade_starts_unique = np.unique(fuzzyClusters[:, 0])
        for cascade_start in cascade_starts_unique:
            sl[np.where(fuzzyClusters[:, 0] == cascade_start)[0][-1]] = 1

        # Eliminate fuzzy child groups
        for i, fg_i in enumerate(fuzzyClusters):
            if sl[i]: sl *= ~np.logical_and(fuzzyClusters[:, 0] > fg_i[0], fuzzyClusters[:, 1] <= fg_i[1])

        # Setup memberships and fuzzy hierarchy arrays
        shape0 = sl.sum()
        memberships = np.zeros((shape0, nPoints))
        _hierarchyCorrection = np.zeros((shape0, nPoints))
        fuzzyHierarchy = np.zeros((shape0, shape0))

        return fuzzyClusters[sl], stabilities[sl], memberships, _hierarchyCorrection, fuzzyHierarchy
    
    @staticmethod
    @njit()
    def _setupHierarchyInformation_njit(ordering, fuzzyClusters, _sampleNumbers, nSamples):
        whichFuzzyCluster = -np.ones(ordering.size, dtype = np.int32)
        sampleWeights = np.zeros((nSamples, fuzzyClusters.shape[0]), dtype = np.int32)
        sampleNumbers_ordered = _sampleNumbers[ordering]
        for i, clst in enumerate(fuzzyClusters):
            whichFuzzyCluster[ordering[clst[0]:clst[1]]] = i
            for j in range(nSamples):
                multiplicity = (sampleNumbers_ordered[clst[0]:clst[1]] == j).sum()
                if multiplicity: sampleWeights[j, i] = 1/multiplicity
        return whichFuzzyCluster, sampleWeights

    @staticmethod
    @njit()
    def _updateMemberships_njit(memberships, _hierarchyCorrection, fuzzyHierarchy, clusterArr, whichFC_cluster, whichFC_parents, sampleWeights_i):
        memberships[whichFC_cluster, clusterArr] += sampleWeights_i[whichFC_cluster]
        for whichFC_parent in whichFC_parents:
            if whichFC_parent != -1 and whichFC_parent != whichFC_cluster:
                weight = sampleWeights_i[whichFC_cluster]*sampleWeights_i[whichFC_parent]
                _hierarchyCorrection[whichFC_parent, clusterArr] += weight
                fuzzyHierarchy[whichFC_parent, whichFC_cluster] += weight