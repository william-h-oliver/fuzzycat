"""
FuzzyCat: A generalised method of producing a soft hierarchy of soft clusters 
from a series of existing clusterings that have been generated using 
different representations of the same point-based data set.

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

    FuzzyCat is a generalised method of producing a soft hierarchy of soft 
    clusters from a series of existing clusterings that have been generated 
    using different representations of the same point-based data set. The core 
    concept of the algorithm is that a density-based clustering of these 
    existing clusters can be found, since the Jaccard index (distance) between 
    clusters endows the space of clusters with a similarity (metric). The 
    resultant clusters of clusters thereby propagate any hidden effects of the 
    process that is underlying the different representations of the data set.

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
    minIntraJaccardIndex : `float`, default = 0.5
        The minimum Jaccard index that at least two clusters within a fuzzy 
        cluster must have for it be included in the final set of fuzzy clusters.
    maxInterJaccardIndex : `float`, default = 0.5
        The maximum Jaccard index that any two fuzzy clusters can have for them
        to be able to included in the final set of fuzzy clusters.
    minStability : `float`, default = 0.5
        The minimum stability that a fuzzy cluster must have to be included in 
        the final set of fuzzy clusters.
    windowSize : `int` or `None`, default = None
        The size of the window that is used to compute the Jaccard index between
        clusters. If `None`, the Jaccard index is computed between all pairs of
        clusters in all clusterings. If `windowSize` is an integer, then the 
        Jaccard index is computed between all clusters within `windowSize`-many 
        clusterings of one another. The order of the clusters is determined by 
        the lexographical order of the cluster files in `directoryName`.
    checkpoint : `bool`, default = False
        Whether to save the cluster file names, pairs, and edges arrays to the 
        directory so that less work is needed if FuzzyCat is run again.
    verbose : `int`, default = 0
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
        clusters before they have been selected for with `minIntraJaccardIndex`, 
        `maxInterJaccardIndex`, and `minStability`.
    intraJaccardIndicesGroups : `numpy.ndarray` of shape (n_groups,)
        The maxmimum Jaccard Index value of each group in `groups`, such that
        `intraJaccardIndicesGroups[i]` corresponds to group `i` in `groups`.
    interJaccardIndicesGroups : `numpy.ndarray` of shape (n_groups,)
        The maximum Jaccard Index value that each group shares with another 
        group in `groups`, such that `interJaccardIndicesGroups[i]` corresponds 
        to group `i` in `groups`.
    stabilitiesGroups : `numpy.ndarray` of shape (n_groups,)
        The stability of each group in `groups`, such that
        `stabilitiesGroups[i]` corresponds to group `i` in `groups`.
    """

    def __init__(self, nSamples, nPoints, directoryName = None, minIntraJaccardIndex = 0.5, maxInterJaccardIndex = 0.5, minStability = 0.5, windowSize = None, checkpoint = False, verbose = 0):
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

        check_minIntraJaccardIndex = issubclass(type(minIntraJaccardIndex), (int, float, np.integer, np.floating)) and 0 <= minIntraJaccardIndex <= 1
        assert check_minIntraJaccardIndex, "Parameter 'minIntraJaccardIndex' must be a float (or integer) in the interval [0, 1]!"
        self.minIntraJaccardIndex = minIntraJaccardIndex

        check_maxInterJaccardIndex = issubclass(type(maxInterJaccardIndex), (int, float, np.integer, np.floating)) and 0 <= maxInterJaccardIndex <= 1
        assert check_maxInterJaccardIndex, "Parameter 'maxInterJaccardIndex' must be a float (or integer) in the interval [0, 1]!"
        self.maxInterJaccardIndex = maxInterJaccardIndex

        check_minStability = issubclass(type(minStability), (int, float, np.integer, np.floating)) and 0 <= minStability <= 1
        assert check_minStability, "Parameter 'minExistenceProbability' must be a float (or integer) in the interval [0, 1]!"
        self.minStability = minStability

        check_windowSize = (isinstance(windowSize, int) and windowSize > 0) or windowSize is None
        assert check_windowSize, "Parameter 'windowSize' must be a positive integer or None!"
        self.windowSize = windowSize

        check_checkpoint = isinstance(checkpoint, bool)
        assert check_checkpoint, "Parameter 'checkpoint' must be a boolean!"
        self.checkpoint = checkpoint

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
            self.lazyLoader = [False for i in range(self.clusterFileNames.size)]
            self.dataTypes = np.zeros(self.clusterFileNames.size, dtype = np.int8)
        else:
            # Get all cluster files in directory
            self.clusterFileNames = np.array([fileName for fileName in os.listdir(self.directoryName + 'Clusters/') if fileName.endswith('.npy')])
            n_clusters = self.clusterFileNames.size
            clusteringNumbers = np.array([np.uint32(fileName.split('_')[0]) for fileName in self.clusterFileNames])
            self._pairs, self._edges = self._initGraph(n_clusters, self.windowSize, clusteringNumbers)

            # Cycle through all pairs of clusters and compute their similarity
            self.lazyLoader = [False for i in range(n_clusters)]
            self.dataTypes = np.zeros(n_clusters, dtype = np.int8)
            k = 0
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # Load clusters
                    cluster_i, dataType_i = self.retrieveCluster(i)
                    cluster_j, dataType_j = self.retrieveCluster(j)

                    # Calculate the similarity between clusters i and j
                    if dataType_i == dataType_j == 1: self._edges[k] = self._jaccardIndex_njit(cluster_i, cluster_j, self.nPoints)
                    elif dataType_i == dataType_j:
                        self._edges[k] = self._weightedJaccardIndex_njit(cluster_i, cluster_j)
                    else:
                        clusterFloating = np.zeros(self.nPoints)
                        if dataType_i == 1:
                            clusterFloating[cluster_i] = 1
                            self._edges[k] = self._weightedJaccardIndex_njit(clusterFloating, cluster_j)
                        else:
                            clusterFloating[cluster_j] = 1
                            self._edges[k] = self._weightedJaccardIndex_njit(cluster_i, clusterFloating)
                    k += 1

                    # Check if the window size has been reached
                    if self.windowSize is not None and clusteringNumbers[i] + self.windowSize  - 1 < clusteringNumbers[j]: break
                        
            # Save arrays
            if self.checkpoint:
                np.save(self.directoryName + 'clusterFileNames.npy', self.clusterFileNames)
                np.save(self.directoryName + 'pairs.npy', self._pairs)
                np.save(self.directoryName + 'edges.npy', self._edges)

        self._similarityMatrixTime = time.perf_counter() - start

    @staticmethod
    @njit()
    def _initGraph(n, windowSize, clusteringNumbers):
        pairs = [] # Might not need to compute this if i and j can be (efficiently) calculated from knowing the index of [i, j]
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append([i, j])
                if windowSize is not None and clusteringNumbers[i] + windowSize  - 1 < clusteringNumbers[j]: break
        pairs = np.array(pairs, dtype = np.uint32)
        edges = np.zeros(pairs.shape[0], dtype = np.float32)
        return pairs, edges

    def retrieveCluster(self, index):
        cluster, dataType = self.lazyLoader[index], self.dataTypes[index]
        if cluster is False:
            fileName = self.directoryName + 'Clusters/' + self.clusterFileNames[index]
            cluster = np.load(fileName)
            if issubclass(cluster.dtype.type, (int, np.integer)): dataType = 1
            elif issubclass(cluster.dtype.type, (float, np.floating)): dataType = 2
            else: assert False, f"Cluster from file '{fileName}' is of {cluster.dtype.type} data type (must be integer or floating)!"
            self.lazyLoader[index], self.dataTypes[index] = cluster, dataType
        return cluster, dataType

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
        `intraJaccardIndicesGroups`, `interJaccardIndicesGroups`, and 
        `stabilitiesGroups` attributes.

        This method deletes the `_pairs` and '_edges' attributes.
        """

        if self.verbose > 1: self._printFunction('Making fuzzy clusters...        ')
        start = time.perf_counter()
        self._sampleNumbers = np.array([np.uint32(splitFileName[0]) for splitFileName in np.char.split(self.clusterFileNames, '_', 1)])
        self.jaccardIndices, self.ordering, self.groups, self.intraJaccardIndicesGroups, self.interJaccardIndicesGroups, self.stabilitiesGroups = self._aggregate_njit(self._pairs, self._edges, self._sampleNumbers, self.nSamples)
        del self._pairs, self._edges
        reorder = np.array(sorted(np.arange(self.groups.shape[0]), key = lambda i: [self.groups[i, 0], self.clusterFileNames.size - self.groups[i, 1]]), dtype = np.uint32)
        self.groups, self.intraJaccardIndicesGroups, self.interJaccardIndicesGroups, self.stabilitiesGroups = self.groups[reorder], self.intraJaccardIndicesGroups[reorder], self.interJaccardIndicesGroups[reorder], self.stabilitiesGroups[reorder]
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
        intraJI_leq = [np.float32(0.0) for i in range(0)]
        interJI_leq = [np.float32(0.0) for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # For larger groups
        starts_geq = [np.uint32(0) for i in range(0)]
        sizes_geq = [np.uint32(0) for i in range(0)]
        intraJI_geq = [np.float32(0.0) for i in range(0)]
        interJI_geq = [np.float32(0.0) for i in range(0)]

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
                    intraJI_geq[id_1] = intraJI_leq[id_0]
                    interJI_geq[id_1] = edge

                    # Merge
                    starts_leq[id_1] += sizes_leq[id_0]
                    sizes_leq[id_0] += sizes_leq[id_1]
                    intraJI_leq[id_0] = max(intraJI_leq[id_0], intraJI_leq[id_1])
                    interJI_leq[id_1] = edge

                    children[id_0].append(id_1)
            elif id_1 == n_clusters: # Neither are aggregated
                ids[pair] = count
                jaccardIndices[pair] = edge
                count += 1
                aggregations.append([pair[0], pair[1]])
                # Create group
                starts_leq.append(0)
                sizes_leq.append(2)
                intraJI_leq.append(edge)
                interJI_leq.append(0.0)
                children.append([np.uint32(0) for i in range(0)])
                # Track complementary group
                starts_geq.append(0)
                sizes_geq.append(0)
                intraJI_geq.append(0.0)
                interJI_geq.append(0.0)
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                jaccardIndices[p_0] = edge
                aggregations[id_1].append(p_0)
                sizes_leq[id_1] += 1
                intraJI_leq[id_1] = max(intraJI_leq[id_1], edge)

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
                intraJI_geq[id_leq] = intraJI_leq[id_final]
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
                for childID in childIDs:
                    starts_leq[childID] += startAdjust
                    starts_geq[childID] += startAdjust
                children[id_leq] = emptyIntList

        # Lists to Arrays
        starts_leq = np.array(starts_leq, dtype = np.uint32)
        sizes_leq = np.array(sizes_leq, dtype = np.uint32)
        intraJI_leq = np.array(intraJI_leq, dtype = np.float32)
        interJI_leq = np.array(interJI_leq, dtype = np.float32)
        starts_geq = np.array(starts_geq, dtype = np.uint32)
        sizes_geq = np.array(sizes_geq, dtype = np.uint32)
        intraJI_geq = np.array(intraJI_geq, dtype = np.float32)
        interJI_geq = np.array(interJI_geq, dtype = np.float32)

        # Clean and reorder arrays
        starts_leq = np.delete(starts_leq, id_final)
        groups_leq = np.column_stack((starts_leq, starts_leq + np.delete(sizes_leq, id_final)))
        starts_leq, sizes_leq = emptyIntArr, emptyIntArr
        intraJI_leq = np.delete(intraJI_leq, id_final)
        interJI_leq = np.delete(interJI_leq, id_final)
        starts_geq = np.delete(starts_geq, id_final)
        groups_geq = np.column_stack((starts_geq, starts_geq + np.delete(sizes_geq, id_final)))
        starts_geq, sizes_geq = emptyIntArr, emptyIntArr
        intraJI_geq = np.delete(intraJI_geq, id_final)
        interJI_geq = np.delete(interJI_geq, id_final)

        # Combine arrays
        groups = np.vstack((groups_leq, groups_geq))
        intraJaccardIndicesGroups = np.concatenate((intraJI_leq, intraJI_geq))
        interJaccardIndicesGroups = np.concatenate((interJI_leq, interJI_geq))

        # Calculate existence probabilities
        stabilitiesGroups = np.empty(groups.shape[0], dtype = np.int64)
        for i, g in enumerate(groups):
            stabilitiesGroups[i] = np.unique(sampleNumbers[ordering[g[0]:g[1]]]).size
        stabilitiesGroups = stabilitiesGroups.astype(np.float64)/nSamples

        return jaccardIndices, ordering, groups, intraJaccardIndicesGroups, interJaccardIndicesGroups, stabilitiesGroups

    def extractFuzzyClusters(self):
        """Classifies fuzzy clusters as the smallest groups that meet the
        `minIntraJaccardIndex`, `maxInterJaccardIndex`, and `minStability` 
        requirements.

        This method requires the `ordering`, `groups`, 
        `intraJaccardIndicesGroups`, `interJaccardIndicesGroups`,
        `stabilitiesGroups`, and `_sampleNumbers` attributes to have already 
        been created, via the `aggregate()` method or otherwise. It also 
        requires the `clusterFileNames` attribute to have already been created, 
        via the `computeSimilarities` method or otherwise. In addition, this 
        method requires the directory to contain a subdirectory called
        'Clusters/' that contains the cluster files.

        This method generates the 'fuzzyClusters', `intraJaccardIndices`, 
        `interJaccardIndices`, `stabilities`, `memberships`,
        `memberships_flat`, and `fuzzyHierarchy` attributes.
        """

        if self.verbose > 1: self._printFunction('Assigning probabilities...        ')
        start = time.perf_counter()

        # Extract fuzzy clusters and setup memberships and fuzzy hierarchy arrays
        self.fuzzyClusters, self.intraJaccardIndices, self.interJaccardIndices, self.stabilities, self.memberships, self._hierarchyCorrection, self.fuzzyHierarchy = self._extractFuzzyClusters_njit(self.groups, self.intraJaccardIndicesGroups, self.interJaccardIndicesGroups, self.stabilitiesGroups, self.minIntraJaccardIndex, self.maxInterJaccardIndex, self.minStability, self.nPoints)
        
        if self.fuzzyClusters.size:
            # Setup hierarchy information
            whichFuzzyCluster, sampleWeights = self._setupHierarchyInformation_njit(self.ordering, self.fuzzyClusters, self._sampleNumbers, self.nSamples)
            baseNames = np.char.add(np.char.rstrip(self.clusterFileNames, '.npy'), '-')

            # Cycle through all clusters and adjust membership probabilities of each point
            for i in range(self.clusterFileNames.size):
                whichFC_cluster = whichFuzzyCluster[i]
                if whichFC_cluster != -1:
                    # Load cluster
                    cluster, dataType = self.retrieveCluster(i)

                    # Find the parent of cluster 'i' from within the same sample
                    whichFC_parents = np.unique(whichFuzzyCluster[np.char.startswith(baseNames[i], baseNames)])

                    # Update memberships
                    if dataType == 1: self._updateMemberships_njit(self.memberships, self._hierarchyCorrection, self.fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights[self._sampleNumbers[i]])
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
    def _extractFuzzyClusters_njit(groups, intraJaccardIndicesGroups, interJaccardIndicesGroups, stabilitiesGroups, minIntraJaccardIndex, maxInterJaccardIndex, minStability, nPoints):
        sl = (intraJaccardIndicesGroups >= minIntraJaccardIndex)*(interJaccardIndicesGroups <= maxInterJaccardIndex)*(stabilitiesGroups >= minStability)
        fuzzyClusters = groups[sl]
        intraJaccardIndices = intraJaccardIndicesGroups[sl]
        interJaccardIndices = interJaccardIndicesGroups[sl]
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

        return fuzzyClusters[sl], intraJaccardIndices[sl], interJaccardIndices[sl], stabilities[sl], memberships, _hierarchyCorrection, fuzzyHierarchy
    
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