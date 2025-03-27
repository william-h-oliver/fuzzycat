"""
Tests for the FuzzyCat class and its utility methods.
"""
# Standard libraries
import os

# Third-party libraries
import numpy as np
from sklearn import datasets
from fuzzycat import FuzzyCat
from fuzzycat import FuzzyData
from fuzzycat import FuzzyPlots

def test_fuzzycat():
    # Set seed for reproducibility
    np.random.seed(0)

    # Remove any files and directories from tests folder that have not been removed by previous tests
    saveFolder = os.getcwd() + '/Clusters/'
    if os.path.exists(saveFolder):
        for fileName in os.listdir(saveFolder):
            os.remove(saveFolder + fileName)
        os.rmdir(saveFolder)
    saveFolder = os.getcwd() + '/'
    for fileName in os.listdir(saveFolder):
         if fileName.endswith('.npy') or fileName.endswith('.png'):
             os.remove(saveFolder + fileName)

    # Generate some clusterable fuzzy data
    nPoints, sigma = 10**4, 0.05**2
    P = datasets.make_moons(nPoints, noise = 0.1)[0]
    covP = sigma*np.tile(np.eye(P.shape[1]).reshape(1, P.shape[1], P.shape[1]), (P.shape[0], 1, 1))

    # Resample the data and cluster it
    nSamples = 50
    try:
        alg = 'kmeans'
        FuzzyData.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg, clusteringAlgorithmArgs = {'n_clusters': 2, 'n_init': 1})
        saveFolder = os.getcwd() + '/Clusters/'
        for fileName in os.listdir(saveFolder):
            os.remove(saveFolder + fileName)

        alg = 'gaussianmixture'
        FuzzyData.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg, clusteringAlgorithmArgs = {'n_components': 2})
        saveFolder = os.getcwd() + '/Clusters/'
        for fileName in os.listdir(saveFolder):
            os.remove(saveFolder + fileName)
        
        alg = 'hdbscan'
        FuzzyData.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg)
        saveFolder = os.getcwd() + '/Clusters/'
        for fileName in os.listdir(saveFolder):
            os.remove(saveFolder + fileName)

        alg = 'astrolink'
        FuzzyData.clusteringsFromRandomSamples(P, covP, nSamples = nSamples, clusteringAlgorithm = alg)
    except: assert False, f"Computing clusterings of random samples failed (with {alg})!"

    # Run FuzzyCat on the realisations
    try: fc = FuzzyCat(nSamples, nPoints)
    except: assert False, "FuzzyCat class could not be instantiated!"
    
    try: fc.run()
    except: assert False, "FuzzyCat.run() failed!"

    # Run jitted methods in python mode for increased test coverage, these are tested in the runs above and in their salient outputs below
    # sqrtCovPCase6_njit()
    arr = FuzzyData.sqrtCovPCase6_njit.py_func(covP)
    del arr

    # _initGraph()
    clusteringNumbers = np.array([np.uint32(fileName.split('_')[0]) for fileName in fc.clusterFileNames])
    arr1, arr2 = fc._initGraph.py_func(100, None, clusteringNumbers)
    del arr1, arr2, clusteringNumbers

    # _jaccardIndex_njit()
    num = fc._jaccardIndex_njit.py_func(np.arange(10), np.arange(5, 20), 20)

    # _weightedJaccardIndex_njit()
    num = fc._weightedJaccardIndex_njit.py_func(np.random.uniform(0, 1, 10), np.random.uniform(0, 1, 10))
    assert 0 <= num <= 1, "Weighted Jaccard index must be in the interval [0, 1]!"
    del num

    # _aggregate_njit()
    fc.computeSimilarities()
    fc._sampleNumbers = np.array([np.uint32(splitFileName[0]) for splitFileName in np.char.split(fc.clusterFileNames, '_', 1)])
    arr1, arr2, arr3, arr4, arr5, arr6 = fc._aggregate_njit.py_func(fc._pairs, fc._edges, fc._sampleNumbers, fc.nSamples)
    del arr1, arr2, arr3, arr4, arr5, arr6

    # _extractFuzzyClusters_njit()
    arr1, arr2, arr3, arr4, memberships, _hierarchyCorrection, fuzzyHierarchy = fc._extractFuzzyClusters_njit.py_func(fc.groups, fc.intraJaccardIndicesGroups, fc.interJaccardIndicesGroups, fc.stabilitiesGroups, fc.minIntraJaccardIndex, fc.maxInterJaccardIndex, fc.minStability, fc.nPoints)
    del arr1, arr2, arr3, arr4

    # _setupHierarchyInformation_njit()
    whichFuzzyCluster, sampleWeights = fc._setupHierarchyInformation_njit.py_func(fc.ordering, fc.fuzzyClusters, fc._sampleNumbers, fc.nSamples)
    baseNames = np.char.add(np.char.rstrip(fc.clusterFileNames, '.npy'), '-')

    # _updateMemberships_njit() and _updateWeightedMemberships_njit()
    for i in range(fc.clusterFileNames.size):
        whichFC_cluster = whichFuzzyCluster[i]
        if whichFC_cluster != -1:
            # Load cluster
            cluster, dataType = fc.retrieveCluster(i)

            # Find the parent of cluster 'i' from within the same sample
            whichFC_parents = np.unique(whichFuzzyCluster[np.char.startswith(baseNames[i], baseNames)])

            # Update memberships
            if dataType == 1:
                fc._updateMemberships_njit.py_func(memberships, _hierarchyCorrection, fuzzyHierarchy, cluster, whichFC_cluster, whichFC_parents, sampleWeights[fc._sampleNumbers[i]])
                try:
                    clusterFloating = np.zeros(fc.nPoints)
                    clusterFloating[cluster] = 1
                    fc._updateWeightedMemberships_njit.py_func(memberships, _hierarchyCorrection, fuzzyHierarchy, clusterFloating, whichFC_cluster, whichFC_parents, sampleWeights[fc._sampleNumbers[i]])
                except: assert False, "'_updateWeightedMemberships_njit()' failed!"
                del whichFC_cluster, cluster, dataType, whichFC_parents, clusterFloating
                break
    del memberships, _hierarchyCorrection, fuzzyHierarchy, whichFuzzyCluster, sampleWeights, baseNames

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
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'jaccardIndices' values must be in the interval [0, 1]!"

    # ordering
    arr = fc.ordering
    assert isinstance(arr, np.ndarray), "Property 'ordering' must be a numpy array!"
    assert arr.dtype.type == np.uint32, "Property 'ordering' must contain unsigned 32-bit integers!"
    assert arr.shape == fc.clusterFileNames.shape, "Property 'ordering' must have the same shape as 'clusterFileNames'!"
    assert arr.size == np.unique(arr).size, "Property 'ordering' must contain unique values!"
    assert arr.min() == 0 and arr.max() == fc.clusterFileNames.size - 1, "Property 'ordering' must contain integer values in the interval [0, clusterFileNames.size - 1]!"

    # fuzzyClusters
    arr = fc.fuzzyClusters
    assert isinstance(arr, np.ndarray), "Property 'fuzzyClusters' must be a numpy array!"
    assert arr.ndim == 2, "Property 'fuzzyClusters' must be 2-dimensional!"
    assert arr.shape[1] == 2, "Property 'fuzzyClusters' must have 2 columns!"
    assert np.all(arr[:, 1] > arr[:, 0]), "Property 'fuzzyClusters' must have the first column less than the second column (as these values represent the start and end positions of the fuzzy cluster within the ordered list)!"
    assert np.all(arr >= 0) and np.all(arr <= fc.clusterFileNames.size), "All values in 'fuzzyClusters' must be in the interval [0, clusterFileNames.size - 1]!"

    # stabilities
    arr = fc.stabilities
    assert isinstance(arr, np.ndarray), "Property 'stabilities' must be a numpy array!"
    assert arr.ndim == 1, "Property 'stabilities' must be 1-dimensional!"
    assert arr.size == fc.fuzzyClusters.shape[0], "Property 'stabilities' must have the same number of rows as 'fuzzyClusters'!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'stabilities' values must be in the interval [0, 1]!"
    assert np.allclose(np.minimum(arr, fc.minStability), fc.minStability), "All 'stabilities' values must be greater than or equal to 'minStability'!"

    # memberships
    arr = fc.memberships
    assert isinstance(arr, np.ndarray), "Property 'memberships' must be a numpy array!"
    assert arr.ndim == 2, "Property 'memberships' must be 2-dimensional!"
    assert arr.shape[0] == fc.fuzzyClusters.shape[0], "Property 'memberships' must have the same number of rows as 'fuzzyClusters'!"
    assert arr.shape[1] == fc.nPoints, "Property 'memberships' must have `nPoints`-many columns!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'memberships' values must be in the interval [0, 1]!"

    # memberships_flat
    arr = fc.memberships_flat
    assert isinstance(arr, np.ndarray), "Property 'memberships_flat' must be a numpy array!"
    assert arr.ndim == 2, "Property 'memberships_flat' must be 2-dimensional!"
    assert arr.shape[0] == fc.fuzzyClusters.shape[0], "Property 'memberships_flat' must have the same number of rows as 'fuzzyClusters'!"
    assert arr.shape[1] == fc.nPoints, "Property 'memberships_flat' must have `nPoints`-many columns!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'memberships_flat' values must be in the interval [0, 1]!"
    assert np.allclose(np.maximum(fc.stabilities.reshape(-1, 1)*arr.sum(axis = 0), 1), 1), "For each point, the sum over the fuzzy clusters of 'memberships_flat' values must be less than or equal to 1!"

    # fuzzyHierarchy
    arr = fc.fuzzyHierarchy
    assert isinstance(arr, np.ndarray), "Property 'fuzzyHierarchy' must be a numpy array!"
    assert arr.ndim == 2, "Property 'fuzzyHierarchy' must be 2-dimensional!"
    assert arr.shape[0] == arr.shape[1] == fc.fuzzyClusters.shape[0], "Property 'fuzzyHierarchy' must be a square array with the same number of rows/columns as 'fuzzyClusters' has rows!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'fuzzyHierarchy' values must be in the interval [0, 1]!"

    # groups
    arr = fc.groups
    assert isinstance(arr, np.ndarray), "Property 'groups' must be a numpy array!"
    assert arr.ndim == 2, "Property 'groups' must be 2-dimensional!"
    assert arr.shape[1] == 2, "Property 'groups' must have 2 columns!"
    assert np.all(arr[:, 1] > arr[:, 0]), "Property 'groups' must have the first column less than the second column (as these values represent the start and end positions of the group within the ordered list)!"
    assert np.all(arr >= 0) and np.all(arr <= fc.clusterFileNames.size), "All values in 'groups' must be in the interval [0, clusterFileNames.size - 1]!"

    # intraJaccardIndicesGroups
    arr = fc.intraJaccardIndicesGroups
    assert isinstance(arr, np.ndarray), "Property 'intraJaccardIndicesGroups' must be a numpy array!"
    assert arr.ndim == 1, "Property 'intraJaccardIndicesGroups' must be 1-dimensional!"
    assert arr.size == fc.groups.shape[0], "Property 'intraJaccardIndicesGroups' must have the same number of rows as 'groups'!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'intraJaccardIndicesGroups' values must be in the interval [0, 1]!"

    # interJaccardIndicesGroups
    arr = fc.interJaccardIndicesGroups
    assert isinstance(arr, np.ndarray), "Property 'interJaccardIndicesGroups' must be a numpy array!"
    assert arr.ndim == 1, "Property 'interJaccardIndicesGroups' must be 1-dimensional!"
    assert arr.size == fc.groups.shape[0], "Property 'interJaccardIndicesGroups' must have the same number of rows as 'groups'!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'interJaccardIndicesGroups' values must be in the interval [0, 1]!"

    # stabilityGroups
    arr = fc.stabilitiesGroups
    assert isinstance(arr, np.ndarray), "Property 'stabilitiesGroups' must be a numpy array!"
    assert arr.ndim == 1, "Property 'stabilitiesGroups' must be 1-dimensional!"
    assert arr.size == fc.groups.shape[0], "Property 'stabilitiesGroups' must have the same number of rows as 'groups'!"
    assert np.allclose(np.minimum(arr, 0), 0) and np.allclose(np.maximum(arr, 1), 1), "All 'stabilityGroups' values must be in the interval [0, 1]!"

    # Test plotting_utils
    try: FuzzyPlots.plotOrderedJaccardIndex(fc)
    except: assert False, "Plotting ordered Jaccard index failed!"

    try: FuzzyPlots.plotStabilities(fc)
    except: assert False, "Plotting stabilities failed!"

    try: FuzzyPlots.plotMemberships(fc)
    except: assert False, "Plotting memberships failed!"

    try: FuzzyPlots.plotFuzzyLabelsOnX(fc, P)
    except: assert False, "Plotting fuzzy labels on data points failed!"

    # Remove generated files and directories from tests folder
    saveFolder = os.getcwd() + '/Clusters/'
    for fileName in os.listdir(saveFolder):
        os.remove(saveFolder + fileName)
    os.rmdir(saveFolder)
    saveFolder = os.getcwd() + '/'
    for fileName in os.listdir(saveFolder):
         if fileName.endswith('.npy') or fileName.endswith('.png'):
             os.remove(saveFolder + fileName)