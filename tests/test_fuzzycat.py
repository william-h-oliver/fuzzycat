import numpy as np
from sklearn import datasets
from fuzzycat import FuzzyCat
from fuzzycat import randomSampleClustering_utils as rsc_utils
from fuzzycat import plotting_utils

def test_fuzzycat():
    # Generate some clusterable fuzzy data
    nPoints, sigma = 5*10**3, 0.075
    P = datasets.make_blobs(nPoints, cluster_std = [1.0, 2.5, 0.5], random_state = 170)[0]
    covP = sigma*np.tile(np.ones(P.shape[1]).reshape(1, P.shape[1], P.shape[1]), (P.shape[0], 1, 1))

    # Resample the data and cluster it
    nSamples = 20
    try:
        alg = 'kmeans'
        rsc_utils.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg)
        alg = 'gaussianmixture'
        rsc_utils.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg)
        alg = 'hdbscan'
        rsc_utils.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg)
        alg = 'astrolink'
        rsc_utils.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg)
    except: assert False, f"Computing clusterings of random samples failed (with {alg})!"

    # Run FuzzyCat on the realisations
    try: fc = FuzzyCat(nSamples, nPoints, checkpoint = True)
    except: assert False, "FuzzyCat class could not be instantiated!"
    
    try: fc.run()
    except: assert False, "FuzzyCat.run() failed!"

    # Run jitted methods in python mode for increased test coverage, these are tested in the runs above and in their salient outputs below
    # sqrtCovPCase6_njit()
    arr = rsc_utils.sqrtCovPCase6_njit.pyfunc(covP)

    # _initGraph()
    arr1, arr2 = fc._initGraph.pyfunc(100)

    # _jaccardIndex_njit()
    num = fc._jaccardIndex_njit.pyfunc(np.arange(10), np.arange(5, 20), 20)

    # _weightedJaccardIndex_njit()
    num = fc._weightedJaccardIndex_njit.pyfunc(np.random.uniform(0, 1, 10), np.random.uniform(0, 1, 10))
    assert 0 <= num <= 1, "Weighted Jaccard index must be in the interval [0, 1]!"

    # _aggregate_njit()
    fc.computeSimilarities()
    fc._sampleNumbers = np.array([np.uint32(splitFileName[0]) for splitFileName in np.char.split(fc.clusterFileNames, '_', 1)])
    arr1, arr2, arr3, arr4, arr5 = fc._aggregate_njit.pyfunc(fc.pairs, fc.edges)

    # _extractFuzzyClusters_njit()
    arr1, arr2, memberships, _hierarchyCorrection, fuzzyHierarchy = fc._extractFuzzyClusters_njit.pyfunc(fc.groups, fc.prominences, fc.stabilitiesGroups, fc.minJaccardIndex, fc.minStability, fc.nPoints)

    # _setupHierarchyInformation_njit()
    whichFuzzyCluster, sampleWeights = fc._setupHierarchyInformation_njit.pyfunc(fc.ordering, fc.fuzzyClusters, fc._sampleNumbers, fc.nSamples)
    baseNames = np.char.add(np.char.rstrip(fc.clusterFileNames, '.npy'), '-')

    # _updateMemberships_njit() and _updateWeightedMemberships_njit()
    for i, clstFileName_i in enumerate(fc.clusterFileNames):
        whichFC_cluster = whichFuzzyCluster[i]
        if whichFC_cluster != -1:
            # Load cluster
            cluster, dType = fc.readClusterFile(fc.directoryName + 'Clusters/' + clstFileName_i)

            # Find the parent of cluster 'i' from within the same sample
            whichFC_parents = np.unique(whichFuzzyCluster[np.char.startswith(baseNames[i], baseNames)])

            # Update memberships
            if dType == 1:
                fc._updateMemberships_njit.pyfunc(memberships, _hierarchyCorrection, fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights[fc._sampleNumbers[i]])
                try: fc._updateWeightedMemberships_njit.pyfunc(memberships, _hierarchyCorrection, fuzzyHierarchy, cluster.astype(np.float32), whichFC_cluster, whichFC_parents, sampleWeights[fc._sampleNumbers[i]])
                except: assert False, "'_updateWeightedMemberships_njit()' failed!"
                break
    del arr, num, arr1, arr2, arr3, arr4, arr5, memberships, _hierarchyCorrection, fuzzyHierarchy, whichFuzzyCluster, sampleWeights, baseNames, cluster, dType

    # Test properties
    # clusterFileNames
    arr = fc.clusterFileNames
    assert isinstance(arr, np.ndarray), "Property 'clusterFileNames' must be a numpy array!"
    assert arr.ndim == 1, "Property 'clusterFileNames' must be 1-dimensional!"
    assert arr.dtype.type == np.str_, "Property 'clusterFileNames' must contain strings!"

    # jaccardIndices
    arr = fc.jaccardIndices
    assert isinstance(arr, np.ndarray), "Property 'jaccardIndices' must be a numpy array!"
    assert arr.dtype.type == np.float32, "Property 'jaccardIndices' must contain 32-bit floats!"
    assert arr.shape == fc.clusterFileNames.shape, "Property 'jaccardIndices' must have the same shape as 'clusterFileNames'!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'jaccardIndices' values must be in the interval [0, 1]!"

    # ordering
    arr = fc.ordering
    assert isinstance(arr, np.ndarray), "Property 'ordering' must be a numpy array!"
    assert arr.dtype.type == np.uint32, "Property 'ordering' must contain unsigned 32-bit integers!"
    assert arr.shape == fc.clusterFileNames.shape, "Property 'ordering' must have the same shape as 'clusterFileNames'!"
    assert arr.size == np.unique(arr).size, "Property 'ordering' must contain unique values!"
    assert arr.min() == 0 and arr.max() == fc.nSamples - 1, "Property 'ordering' must contain integer values in the interval [0, nSamples - 1]!"

    # fuzzyClusters
    arr = fc.fuzzyClusters
    assert isinstance(arr, np.ndarray), "Property 'fuzzyClusters' must be a numpy array!"
    assert arr.ndim == 2, "Property 'fuzzyClusters' must be 2-dimensional!"
    assert arr.dtype.type == np.int32, "Property 'fuzzyClusters' must contain 32-bit integers!"
    assert arr.shape[1] == 2, "Property 'fuzzyClusters' must have 2 columns!"
    assert arr[:, 1] > arr[:, 0], "Property 'fuzzyClusters' must have the first column less than the second column (as these values represent the start and end positions of the fuzzy cluster within the ordered list)!"
    assert np.all(arr >= 0) and np.all(arr < fc.nSamples), "All values in 'fuzzyClusters' must be in the interval [0, nSamples - 1]!"

    # stabilities
    arr = fc.stabilities
    assert isinstance(arr, np.ndarray), "Property 'stabilities' must be a numpy array!"
    assert arr.ndim == 1, "Property 'stabilities' must be 1-dimensional!"
    assert arr.dtype.type == np.float32, "Property 'stabilities' must contain 32-bit floats!"
    assert arr.size == fc.fuzzyClusters.shape[0], "Property 'stabilities' must have the same number of rows as 'fuzzyClusters'!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'stabilities' values must be in the interval [0, 1]!"
    assert np.all(arr >= fc.minStability), "All 'stabilities' values must be greater than or equal to 'minStability'!"

    # memberships
    arr = fc.memberships
    assert isinstance(arr, np.ndarray), "Property 'memberships' must be a numpy array!"
    assert arr.ndim == 2, "Property 'memberships' must be 2-dimensional!"
    assert arr.dtype.type == np.float32, "Property 'memberships' must contain 32-bit floats!"
    assert arr.shape[0] == fc.fuzzyClusters.shape[0], "Property 'memberships' must have the same number of rows as 'fuzzyClusters'!"
    assert arr.shape[1] == fc.nPoints, "Property 'memberships' must have `nPoints`-many columns!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'memberships' values must be in the interval [0, 1]!"

    # memberships_flat
    arr = fc.memberships_flat
    assert isinstance(arr, np.ndarray), "Property 'memberships_flat' must be a numpy array!"
    assert arr.ndim == 2, "Property 'memberships_flat' must be 2-dimensional!"
    assert arr.dtype.type == np.float32, "Property 'memberships_flat' must contain 32-bit floats!"
    assert arr.shape[0] == fc.fuzzyClusters.shape[0], "Property 'memberships_flat' must have the same number of rows as 'fuzzyClusters'!"
    assert arr.shape[1] == fc.nPoints, "Property 'memberships_flat' must have `nPoints`-many columns!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'memberships_flat' values must be in the interval [0, 1]!"
    assert np.all(fc.stabilities.reshape(-1, 1)*arr.sum(axis = 0) <= 1), "For each point, the sum over the fuzzy clusters of 'memberships_flat' values must be less than or equal to 1!"

    # fuzzyHierarchy
    arr = fc.fuzzyHierarchy
    assert isinstance(arr, np.ndarray), "Property 'fuzzyHierarchy' must be a numpy array!"
    assert arr.ndim == 2, "Property 'fuzzyHierarchy' must be 2-dimensional!"
    assert arr.dtype.type == np.float32, "Property 'fuzzyHierarchy' must contain 32-bit floats!"
    assert arr.shape[0] == arr.shape[1] == fc.fuzzyClusters.shape[0], "Property 'fuzzyHierarchy' must be a square array with the same number of rows/columns as 'fuzzyClusters' has rows!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'fuzzyHierarchy' values must be in the interval [0, 1]!"

    # groups
    arr = fc.groups
    assert isinstance(arr, np.ndarray), "Property 'groups' must be a numpy array!"
    assert arr.ndim == 2, "Property 'groups' must be 2-dimensional!"
    assert arr.dtype.type == np.int32, "Property 'groups' must contain 32-bit integers!"
    assert arr.shape[1] == 2, "Property 'groups' must have 2 columns!"
    assert arr[:, 1] > arr[:, 0], "Property 'groups' must have the first column less than the second column (as these values represent the start and end positions of the group within the ordered list)!"
    assert np.all(arr >= 0) and np.all(arr < fc.nSamples), "All values in 'groups' must be in the interval [0, nSamples - 1]!"

    # prominences
    arr = fc.prominences
    assert isinstance(arr, np.ndarray), "Property 'prominences' must be a numpy array!"
    assert arr.ndim == 1, "Property 'prominences' must be 1-dimensional!"
    assert arr.dtype.type == np.float32, "Property 'prominences' must contain 32-bit floats!"
    assert arr.size == fc.groups.shape[0], "Property 'prominences' must have the same number of rows as 'groups'!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'prominences' must be in the interval [0, 1]!"

    # stabilityGroups
    arr = fc.stabilitiesGroups
    assert isinstance(arr, np.ndarray), "Property 'stabilitiesGroups' must be a numpy array!"
    assert arr.ndim == 1, "Property 'stabilitiesGroups' must be 1-dimensional!"
    assert arr.dtype.type == np.float32, "Property 'stabilitiesGroups' must contain 32-bit floats!"
    assert arr.size == fc.groups.shape[0], "Property 'stabilitiesGroups' must have the same number of rows as 'groups'!"
    assert np.all(0 <= arr) and np.all(arr <= 1), "All 'stabilityGroups' values must be in the interval [0, 1]!"

    # Test plotting_utils
    try: plotting_utils.plotOrderedJaccardIndex(fc)
    except: assert False, "Plotting ordered Jaccard index failed!"

    try: plotting_utils.plotStabilities(fc)
    except: assert False, "Plotting stabilities failed!"

    try: plotting_utils.plotMemberships(fc)
    except: assert False, "Plotting memberships failed!"

    try: plotting_utils.plotFuzzyLabelsOnX(fc, P)
    except: assert False, "Plotting fuzzy labels on data points failed!"