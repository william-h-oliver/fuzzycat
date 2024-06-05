"""
FuzzyCat: A generalised method of producing probabilistic clusters from a series 
of clusterings that have been generated from different representations of the 
same point-based data.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Standard libraries
import os
import time

# Third-party libraries
import numpy as np
from numba import njit

class FuzzyCat:
    """A class to represent the FuzzyCat algorithm.

    FuzzyCat...

    Parameters
    ----------
    nSamples : `int`
        The number of samples and therefore the number of clusterings that were
        generated.
    nPoints : `int`
        The number of points in the data set.
    directoryName : `str` or `None`, default = None
        The file path of the directory where the cluster files are stored. If 
        `None`, the current working directory is used. The directory must 
        contain a subdirectory called 'Clusters' that contains the cluster 
        files. The cluster files must be in the form of numpy arrays and be 
        named according to the 'XXX_A-B-C.npy' format, where 'XXX' is the
        0-padded sample number (integer in [0, nSamples - 1]) and 'A-B-C' is the 
        hierarchical cluster id of the cluster. The cluster files must also have 
        the '.npy' extension. In addition, the cluster arrays must either; 
        contain the indices (integers) of the points that belong to the cluster, 
        or contain the membership probabilities (floats) of the points in the 
        cluster.
    minJaccardIndex : `float`, default = 0.3
        The minimum Jaccard index that a fuzzy cluster must have to be included
        in the final set of fuzzy clusters.
    minStability : `float`, default = 0.5
        The minimum stability that a fuzzy cluster must have to be included in 
        the final set of fuzzy clusters.
    checkpoint : `bool`, default = False
        Whether to save the cluster file names, pairs, and edges arrays to the 
        directory.
    workers : `int`, default = -1
        The number of processors used in parallelised computations. If `workers`
        is set to -1, then FuzzyCat will use all processors available.
        Otherwise, `workers` must be a value between 1 and N_cpu.
    verbose : `int`, default = 2
        The verbosity of the FuzzyCat class. If `verbose` is set to 0, then
        FuzzyCat will not report any of its activity. Increasing `verbose` will
        make FuzzyCat report more of its activity.

    Attributes
    ----------
    clusterFileNames : `numpy.ndarray` of shape (n_clusters,)
        The names of the cluster files that FuzzyCat has used. The files have
        been found in the subdirectory, `directoryName` + 'Clusters/'.
    jaccardIndices : `numpy.ndarray` of shape (n_clusters,)
        The maximum Jaccard index the clusters share with any other cluster.
    ordering : `numpy.ndarray` of shape (n_clusters,)
        The ordering of the clusters in the ordered list of fuzzy structure. The
        ordered-jaccard plot can be created by plotting
        `y = jaccardIndices[ordering]` vs `x = range(ordering.size)`.
    fuzzyClusters : `numpy.ndarray` of shape (n_fuzzyclusters, 2)
        The start and end positions of each fuzzy cluster as it appears in the
        ordered list, such that
        `ordering[fuzzyClusters[i, 0]:fuzzyClusters[i, 1]]` gives an array of
        the indices of the points within cluster `i`.
    stabilities : `numpy.ndarray` of shape (n_fuzzyclusters,)
        The stability of each fuzzy cluster.
    memberships : `numpy.ndarray` of shape (n_fuzzyclusters, nPoints)
        The membership probabilities of each point to each fuzzy cluster.
    memberships_flat : `numpy.ndarray` of shape (n_fuzzyclusters, nPoints)
        The flattened membership probabilities of each point to each fuzzy
        cluster, excluding the hierarchy correction. If the fuzzy clusters are
        inherently hierarchical, then `memberships_flat` may be more easily
        interpretable than `memberships` since `memberships_flat.sum(axis = 0)` 
        will be contained within the interval [0, 1].
    fuzzyHierarchy : `numpy.ndarray` of shape (n_fuzzyclusters, n_fuzzyclusters)
        Information about the fuzzy hierarchy of the fuzzy clusters. The value
        of `fuzzyHierarchy[i, j]` is the probability that fuzzy cluster `j` is a
        child of fuzzy cluster `i`.
    groups : `numpy.ndarray` of shape (n_groups, 2)
        Similar to `fuzzyClusters`, however `groups` includes all possible fuzzy
        clusters before they have been selected for with `minJaccardIndex` and
        `minStability`.
    prominences : `numpy.ndarray` of shape (n_groups,)
        The prominence values of each group in `groups`, such that
        `prominences[i]` corresponds to group `i` in `groups`.
    stabilitiesGroups : `numpy.ndarray` of shape (n_groups,)
        The stability of each group in `groups`, such that
        `stabilitiesGroups[i]` corresponds to group `i` in `groups`.
    """

    def __init__(self, nSamples, nPoints, directoryName = None, minJaccardIndex = 0.3, minStability = 0.5, checkpoint = False, workers = -1, verbose = 2):
        check_directoryName = (isinstance(directoryName, str) and directoryName != "" and os.path.exists(directoryName)) or directoryName is None
        assert check_directoryName, "Parameter 'directoryName' must be a string and must exist!"
        if directoryName is None: directoryName = os.getcwd()
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
        """Runs the FuzzyCat algorithm and produces fuzzy clusters from a 
        directory containing a folder, 'Cluster/', with existing cluster files.

        This method runs `computeSimilarities()`, `aggregate()`, and
        `extractFuzzyClusters()`.
        """

        assert os.path.exists(self.directoryName + 'Clusters/'), f"Directory {self.directoryName + 'Clusters/'} does not exist!"
        self._printFunction(f"Started            | {time.strftime('%Y-%m-%d %H:%M:%S')}", returnLine = False)
        begin = time.perf_counter()

        # Phase 1
        self.computeSimilarities()

        # Phase 2
        self.aggregate()

        # Phase 3
        self.extractFuzzyClusters()

        self._totalTime = time.perf_counter() - begin
        if self.verbose > 1:
            self._printFunction(f"Similarities time  | {100*self._similarityMatrixTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Aggregation time   | {100*self._aggregationTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Extraction time    | {100*self._extractFuzzyClustersTime/self._totalTime:.2f}%    ", returnLine = False)
        self._printFunction(f"Completed          | {time.strftime('%Y-%m-%d %H:%M:%S')}       ", returnLine = False)

    def computeSimilarities(self):
        """Computes the similarities between all pairs of clusters in the
        chosen directory.

        This method requires the directory to contain a subdirectory called
        'Clusters/' that contains the cluster files.

        This method generates the `_pairs` and `_edges` attributes.
        """

        if self.verbose > 1: self._printFunction('Computing similarities...        ')
        start = time.perf_counter()
        
        # Check if arrays have been computed before
        clstFileNamesFileBool = os.path.exists(self.directoryName + 'clusterFileNames.npy')
        pairsFileBool = os.path.exists(self.directoryName + 'pairs.npy')
        edgesFileBool = os.path.exists(self.directoryName + 'edges.npy')
        
        # If so, load them, otherwise, compute them
        if clstFileNamesFileBool and pairsFileBool and edgesFileBool:
            self.clusterFileNames = np.load(self.directoryName + 'clusterFileNames.npy')
            self._pairs = np.load(self.directoryName + 'pairs.npy')
            self._edges = np.load(self.directoryName + 'edges.npy')
        else:
            # Get all cluster files in directory
            self.clusterFileNames = np.array([fileName for fileName in os.listdir(self.directoryName + 'Clusters/') if fileName.endswith('.npy')])

            # Cycle through all pairs of clusters and compute their similarity
            n_clusters = self.clusterFileNames.size
            lazyLoad = [False for i in range(n_clusters)]
            dTypes = np.zeros(n_clusters, dtype = np.int8)
            self._pairs, self._edges = self._initGraph(n_clusters)
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # Load clusters
                    if lazyLoad[i] is False: lazyLoad[i], dTypes[i] = self.readClusterFile(self.directoryName + 'Clusters/' + self.clusterFileNames[i])
                    if lazyLoad[j] is False: lazyLoad[j], dTypes[j] = self.readClusterFile(self.directoryName + 'Clusters/' + self.clusterFileNames[j])

                    # Calculate the similarity between clusters i and j
                    if dTypes[i] == dTypes[j] == 1: self._edges[i*(2*n_clusters - i - 1)//2 + j - i - 1] = self._jaccardIndex_njit(lazyLoad[i], lazyLoad[j], self.nPoints)
                    elif dTypes[i] == dTypes[j]:
                        self._edges[i*(2*n_clusters - i - 1)//2 + j - i - 1] = self._weightedJaccardIndex_njit(lazyLoad[i], lazyLoad[j])
                    else:
                        clusterFloating = np.zeros(self.nPoints)
                        if dTypes[i] == 1:
                            clusterFloating[lazyLoad[i]] = 1
                            self._edges[i*(2*n_clusters - i - 1)//2 + j - i - 1] = self._weightedJaccardIndex_njit(clusterFloating, lazyLoad[j])
                        else:
                            clusterFloating[lazyLoad[j]] = 1
                            self._edges[i*(2*n_clusters - i - 1)//2 + j - i - 1] = self._weightedJaccardIndex_njit(lazyLoad[i], clusterFloating)
                        
            # Save arrays
            if self.checkpoint:
                np.save(self.directoryName + 'clusterFileNames.npy', self.clusterFileNames)
                np.save(self.directoryName + 'pairs.npy', self._pairs)
                np.save(self.directoryName + 'edges.npy', self._edges)

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
        """Aggregates the clusters together to form the ordered list whilst
        keeping track of groups.

        Sorts cluster pairs into descending order of edge weight and aggregates
        the clusters while keeping track of structural information about the
        data.

        This method requires the `_pairs` and '_edges' attributes to have already
        been created, via the `computeSimilarities` method or otherwise.

        This method generates the `jaccardIndices`, `ordering`, `groups`,
        `prominences`, and `stabilitiesGroups` attributes.

        This method deletes the `_pairs` and '_edges' attributes.
        """

        if self.verbose > 1: self._printFunction('Making fuzzy clusters...        ')
        start = time.perf_counter()
        self._sampleNumbers = np.array([np.uint32(splitFileName[0]) for splitFileName in np.char.split(self.clusterFileNames, '_', 1)])
        self.jaccardIndices, self.ordering, self.groups, self.prominences, self.stabilitiesGroups = self._aggregate_njit(self._pairs, self._edges, self._sampleNumbers, self.nSamples)
        del self._pairs, self._edges
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
        stabilitiesGroups = stabilitiesGroups.astype(np.float32)/nSamples

        return jaccardIndices, ordering, groups, prominences, stabilitiesGroups

    def extractFuzzyClusters(self):
        """Classifies fuzzy clusters as the smallest groups that meet the
        `minJaccardIndex` and `minStability` requirements.

        This method requires the `ordering`, `groups`, `prominences`,
        `stabilitiesGroups`, and `_sampleNumbers` attributes to have already 
        been created, via the `aggregate()` method or otherwise. It also 
        requires the `clusterFileNames` attribute to have already been created, 
        via the `computeSimilarities` method or otherwise. In addition, this 
        method requires the directory to contain a subdirectory called
        'Clusters/' that contains the cluster files.

        This method generates the 'fuzzyClusters', `stabilities`, `memberships`,
        `memberships_flat`, and `fuzzyHierarchy` attributes.
        """

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
                    cluster, dType = self.readClusterFile(self.directoryName + 'Clusters/' + clstFileName_i)

                    # Find the parent of cluster 'i' from within the same sample
                    whichFC_parents = np.unique(whichFuzzyCluster[np.char.startswith(baseNames[i], baseNames)])

                    # Update memberships
                    if dType == 1: self._updateMemberships_njit(self.memberships, self._hierarchyCorrection, self.fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights[self._sampleNumbers[i]])
                    else: self._updateWeightedMemberships_njit(self.memberships, self._hierarchyCorrection, self.fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights[self._sampleNumbers[i]])
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
    def _updateMemberships_njit(memberships, _hierarchyCorrection, fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights_i):
        memberships[whichFC_cluster, cluster] += sampleWeights_i[whichFC_cluster]
        for whichFC_parent in whichFC_parents:
            if whichFC_parent != -1 and whichFC_parent != whichFC_cluster:
                weight = sampleWeights_i[whichFC_cluster]*sampleWeights_i[whichFC_parent]
                _hierarchyCorrection[whichFC_parent, cluster] += weight
                fuzzyHierarchy[whichFC_parent, whichFC_cluster] += weight

    @staticmethod
    @njit()
    def _updateWeightedMemberships_njit(memberships, _hierarchyCorrection, fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights_i):
        memberships[whichFC_cluster] += sampleWeights_i[whichFC_cluster]*cluster
        for whichFC_parent in whichFC_parents:
            if whichFC_parent != -1 and whichFC_parent != whichFC_cluster:
                weight = sampleWeights_i[whichFC_cluster]*sampleWeights_i[whichFC_parent]
                _hierarchyCorrection[whichFC_parent] += weight*cluster
                fuzzyHierarchy[whichFC_parent, whichFC_cluster] += weight