#Author: William H. Oliver <william.hardie.oliver@gmail.com>
#License: MIT


# Standard libraries
import os

# Third-party libraries
import numpy as np
from numba import njit, prange
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from astrolink import AstroLink


def clusteringsFromRandomSamples(P, covP, nSamples = 100, directoryName = None, clusteringAlgorithm = 'astrolink', clusteringAlgorithmArgs = None, workers = -1):
    """Generates random samples of a fuzzy data set, runs a clustering
    algorithm on each sample, and saves the clusters as .npy files so that
    FuzzyCat can use them.

    Parameters
    ----------
    P : numpy.ndarray
        The mean values of the fuzzy data set from which to generate random
        samples from.
    covP : float or numpy.ndarray
        The covariance matrix of the fuzzy data set from which to generate
        random samples from. The random samples of points can be either 
        homogenous or heterogenous and either spherically-symmetric, 
        axis-aligned, or multivariate.
    nSamples : int, default is 100
        The number of random samples to generate.
    directoryName : str, default is None
        The directory in which to save the clusters. If None, the current
        working directory is used.
    clusteringAlgorithm : str or callable, default is 'astrolink'
        The clustering algorithm to use. If a string, the following are
        supported: 'kmeans', 'gaussianmixture', 'hdbscan', 'astrolink'.
        If a callable, the function must take the following parameters:

        - P_sample : numpy.ndarray
            A random sample of the fuzzy data set.
        - iteration : int
            A uniquely identifying number corresponding to this random sample.
        - nSamples : int
            The total number of random samples that are being generated.
        - directoryName : str
            The directory in which to save the clusters.
        - **clusteringAlgorithmArgs : dict
            Additional keyword arguments to pass to the clustering algorithm.
    clusteringAlgorithmArgs : dict, default is None
        Additional keyword arguments to pass to the clustering algorithm.
    workers : int, default is -1
        The number of CPU cores to use. If -1, all available cores are used.
    """

    check_P = isinstance(P, np.ndarray) and P.ndim == 2
    assert check_P, "Parameter 'P' must be a 2D numpy array!"

    check_covP = True
    if isinstance(covP, (int, float)) and covP >= 0: case = 1 # Homogenous spherically symmetric Gaussian noise
    elif isinstance(covP, np.ndarray):
        if covP.ndim == 1 and covP.size == P.shape[0] and (covP >= 0).all(): case = 2 # Heterogenous spherically symmetric Gaussian noise
        if covP.ndim == 1 and covP.size == P.shape[1] and (covP >= 0).all(): case = 3 # Homogenous axis-aligned Gaussian noise
        elif covP.ndim == 2 and covP.shape == P.shape and (covP >= 0).all(): case = 4 # Heterogenous axis-aligned Gaussian noise
        elif covP.ndim == 2 and covP.shape[0] == covP.shape[1] == P.shape[1]: case = 5 # Homogenous Gaussian noise
        elif covP.ndim == 3 and covP.shape[0] == P.shape[0] and covP.shape[1] == covP.shape[2] == P.shape[1]: case = 6 # Heterogenous Gaussian noise
        else: check_covP = False
    else: check_covP = False
    assert check_covP, "Parameter 'covP' must be a positive scalar, a 1D numpy array, a 2D numpy array, or a 3D numpy array!"

    check_nSamples = isinstance(nSamples, int) and nSamples > 0
    assert check_nSamples, "Parameter 'nSamples' must be a positive integer!"

    check_workers = issubclass(type(workers), (int, np.integer)) and (1 <= workers <= os.cpu_count() or workers == -1)
    assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 and <= N_cpu (= {os.cpu_count()})"
    os.environ["OMP_NUM_THREADS"] = f"{workers}" if workers != -1 else f"{os.cpu_count()}"

    if case in [1, 2, 3, 4]: sqrtCovP = np.sqrt(covP)
    elif case == 5:
        eigenValues, eigenVectors = np.linalg.eigh(covP)
        sqrtCovP = np.sqrt(eigenValues)*eigenVectors @ eigenVectors.T
    else: sqrtCovP = sqrtCovPCase6_njit(covP)

    for i in range(nSamples):
        # Generate random sample
        if case in [1, 3, 4]: P_sample = P + sqrtCovP*np.random.normal(0, 1, P.shape)
        elif case == 2: P_sample = P + sqrtCovP.reshape(-1, 1)*np.random.normal(0, 1, P.shape)
        else: P_sample = P + (sqrtCovP @ np.random.normal(0, 1, (*P.shape, 1))).reshape(P.shape)

        # Run clustering algorithm and save clusters
        runAndSaveClustering(P_sample, i, nSamples, directoryName, clusteringAlgorithm, clusteringAlgorithmArgs)
    
@njit(parallel = True)
def sqrtCovPCase6_njit(covP):
    """A fast implementation of the square root of the covariance matrices of
    that describe heterogenous multivariate Gaussian noise.

    Parameters
    ----------
    covP : numpy.ndarray
        The covariance matrices of the fuzzy data set.
    """
    sqrtCovP = np.empty(covP.shape)
    for i in prange(covP.shape[0]):
        eigenValues, eigenVectors = np.linalg.eigh(covP[i])
        sqrtCovP[i] = np.sqrt(eigenValues)*eigenVectors @ eigenVectors.T
    return sqrtCovP

def runAndSaveClustering(P_sample, iteration, nSamples = 1000, directoryName = None, clusteringAlgorithm = 'astrolink', clusteringAlgorithmArgs = None):
    """Runs a clustering algorithm on a sample of a data set and saves the 
    clusters as .npy files so that FuzzyCat can use them.

    Parameters
    ----------
    P_sample : numpy.ndarray
        A random sample of the fuzzy data set.
    iteration : int
        A uniquely identifying number corresponding to this random sample.
    nSamples : int, default is 1000
        The total number of random samples that are being generated.
    directoryName : str, default is None
        The directory in which to save the clusters. If None, the current
        working directory is used.
    clusteringAlgorithm : str or callable, default is 'astrolink'
        The clustering algorithm to use. If a string, the following are
        supported: 'kmeans', 'gaussianmixture', 'hdbscan', 'astrolink'.
        If a callable, the function must take the following parameters:

        - P_sample : numpy.ndarray
            A random sample of the fuzzy data set.
        - iteration : int
            A uniquely identifying number corresponding to this random sample.
        - nSamples : int
            The total number of random samples that are being generated.
        - directoryName : str  
            The directory in which to save the clusters.
        - **clusteringAlgorithmArgs : dict
            Additional keyword arguments to pass to the clustering algorithm.
    clusteringAlgorithmArgs : dict, default is None
        Additional keyword arguments to pass to the clustering algorithm.
    """
    check_P_sample = isinstance(P_sample, np.ndarray) and P_sample.ndim == 2
    assert check_P_sample, "Parameter 'P_sample' must be a 2D numpy array!"

    check_iteration = isinstance(iteration, int) and iteration >= 0 and iteration < nSamples
    assert check_iteration, f"Parameter 'iteration' must be a positive integer between 0 and nSamples (= {nSamples})!"

    check_nSamples = isinstance(nSamples, int) and nSamples > 0
    assert check_nSamples, "Parameter 'nSamples' must be a positive integer!"
    sampleNumberFormat = np.log10(nSamples).astype(int) + 1

    check_directoryName = (isinstance(directoryName, str) and directoryName != "" and os.path.exists(directoryName)) or directoryName is None
    assert check_directoryName, "Parameter 'directoryName' must be a string and must exist!"
    if directoryName is None: directoryName = os.getcwd()
    if directoryName[-1] != '/': directoryName += '/'
    if not os.path.exists(directoryName): os.makedirs(directoryName)
    if not os.path.exists(directoryName + 'Clusters/'): os.makedirs(directoryName + 'Clusters/')

    check_clusteringAlgorithm = (isinstance(clusteringAlgorithm, str) and clusteringAlgorithm in ['kmeans', 'gaussianmixture', 'hdbscan', 'astrolink']) or callable(clusteringAlgorithm)
    assert check_clusteringAlgorithm, "Parameter 'clusteringAlgorithm' must be a callable function or one of the following strings: 'kmeans', 'gaussianmixture', 'hdbscan', 'astrolink'!"

    check_clusteringAlgorithmArgs = clusteringAlgorithmArgs is None or (isinstance(clusteringAlgorithmArgs, dict) and all(isinstance(key, str) for key in clusteringAlgorithmArgs.keys()))
    assert check_clusteringAlgorithmArgs, "Parameter 'clusteringAlgorithmArgs' must be a dictionary with string keys (or None)!"
    if clusteringAlgorithmArgs is None: clusteringAlgorithmArgs = {}

    if clusteringAlgorithm == 'kmeans':
        # Run KMeans
        c = KMeans(**clusteringAlgorithmArgs)
        c.fit(P_sample)

        # Save clusters
        for clst_id in np.unique(c.labels_):
            np.save(directoryName + 'Clusters/' + f"{iteration:0{sampleNumberFormat}}_{clst_id}.npy", np.where(c.labels_ == clst_id)[0])
    elif clusteringAlgorithm == 'gaussianmixture':
        # Run GaussianMixture
        c = GaussianMixture(**clusteringAlgorithmArgs)
        c.fit(P_sample)
        clusters = c.predict_proba(P_sample)

        # Save clusters
        for clst_id, clst in enumerate(clusters.T):
            np.save(directoryName + 'Clusters/' + f"{iteration:0{sampleNumberFormat}}_{clst_id}.npy", clst)
    elif clusteringAlgorithm == 'hdbscan':
        # Run HDBSCAN
        c = HDBSCAN(**clusteringAlgorithmArgs)
        c.fit(P_sample)

        # Save clusters
        for clst_id in np.unique(c.labels_):
            if clst_id != -1:
                np.save(directoryName + 'Clusters/' + f"{iteration:0{sampleNumberFormat}}_{clst_id}.npy", np.where(c.labels_ == clst_id)[0])
    elif clusteringAlgorithm == 'astrolink':
        # Run AstroLink
        c = AstroLink(P_sample, **clusteringAlgorithmArgs)
        c.run()

        # Save clusters
        for clst, clst_id in zip(c.clusters[1:], c.ids[1:]):
            np.save(directoryName + 'Clusters/' + f"{iteration:0{sampleNumberFormat}}_{clst_id}.npy", c.ordering[clst[0]:clst[1]])
    else:
        # Run custom clustering algorithm and save output
        clusteringAlgorithm(P_sample, iteration, nSamples, directoryName, **clusteringAlgorithmArgs)