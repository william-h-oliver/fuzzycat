# Galaxy Formation and Evolution via Phase-temporal Clustering with AstroLink and FuzzyCat

A pressing and continually evolving sub-field of astrophysics is the study of galaxy formation and evolution. This study seeks to understand how and why a galaxy and its substructure develops over time in the context of the surrounding environment and of the underlying cosmological model. To do this, astrophysicists and cosmologists will look to both observational and simulation data. In observations, we may learn from a single snapshot in time of a very large number of galaxies that arise from the ground-truth cosmological model of our Universe. While in simulations, we may learn from many snapshots of a comparatively small number of galaxies that depend on a pre-specified cosmological model. By comparing these two data types, we can hope to constrain our cosmological models and understanding of galaxy formation and evolution.

In the context of simulated data, a typical approach is to use a halo finder (+ merger tree) code to find a catalogue of haloes and their merger tree &mdash; which are then analysed in terms of their physical properties. However, generally these codes are only tracking self-bound groups that satisfy a minimum overdensity threshold. If this threshold is too high then some haloes may be disregarded and if it is too low then some haloes can be lost in the unbinding procedure. Furthermore, these codes will not capture unbound groups that have been (or are in the process of being) tidally disrupted nor will they capture fleeting structure resulting from density waves and hydrodynamical effects. Not having these kinds of structures included in any subsequent analysis means that cosmological models are never constrained against these the existence of these structures in simulations &mdash; even though they are observed to be present in our Universe.

The goal of this page is to serve as a tutorial and to highlight how the combination of [AstroLink][https://github.com/william-h-oliver/astrolink] and [FuzzyCat][https://github.com/william-h-oliver/fuzzycat] can be used as a powerful tool for studying galaxy formation and evolution. AstroLink is a general-purpose astrophysical clustering algorithm built for extracting meaningful hierarchical structure from point-cloud data defined over any feature space, and combined with FuzzyCat, this pipeline is able to find clusters that are both phase-space- and temporally-robust without making any strong assumptions about the kinds of galactic substructures that are (or are not) physically relevant for study within the field.



## Data: The NIHAO-UHD suite of cosmological hydrodynamical simulations




## Code: Running AstroLink and FuzzyCat on NIHAO-UHD stellar haloes

To do spatio-temporal clustering on NIHAO-UHD galaxies we need a python script. Firstly, we do the necessary imports:

```python
import os
import gc

import numpy as np
import pynbody as pb
import matplotlib.pyplot as plt
import matplotlib.colors as col

from astrolink import AstroLink
from fuzzycat import FuzzyCat, FuzzyPlots
```

Then, the first real piece of code we will use is a method that reads a snapshot file with `pynbody` and returns the necessary information about its main halo:

```python
def loadGalaxyAsArrays(snapshotFilePath, particleName, featureSpaceNames = ['pos', 'vel'], featureSpaceUnits = ['kpc', 'km s**-1']):
    """Returns the main halo data, from the simulation file `snapshotFilePath`, 
    for particle `particleName`, in the feature spaces specified by 
    `featureSpaceNames`, with the units specified by `featureSpaceUnits`.
    """

    # Load the simulation snapshot
    simulation = pb.load(snapshotFilePath)
    
    # Take only the largest halo and make it face-on (stellar disk is in the x-y plane)
    mainHalo = simulation.halos()[1]
    pb.analysis.angmom.faceon(mainHalo)

    # Centre data on the median of the dark matter halo
    darkMatter = np.column_stack([mainHalo.dm[feature].in_units(unit) for feature, unit in zip(featureSpaceNames, featureSpaceUnits)])
    centre = np.median(darkMatter, axis = 0)

    # Get particle data and IDs
    if particleName == 'dark':
        darkMatter -= centre
        darkMatterIDs = mainHalo.dm['iord']
        return darkMatter, darkMatterIDs
    if particleName == 'stars':
        stars = np.column_stack([mainHalo.stars[feature].in_units(unit) for feature, unit in zip(featureSpaceNames, featureSpaceUnits)])
        stars -= centre
        starsIDs = mainHalo.stars['iord']
        return stars, starsIDs
    if particleName == 'gas':
        gas = np.column_stack([mainHalo.gas[feature].in_units(unit) for feature, unit in zip(featureSpaceNames, featureSpaceUnits)])
        gas -= centre
        gasIDs = mainHalo.gas['iord']
        return gas, gasIDs
```

Although we only present results for stellar particles here, this method readily allows for spatio-clustering of the dark matter and/or gas particles too.

With this method we can write a method that uses AstroLink to cluster the particles in each snapshot:

```python
def findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples):
    """Uses AstroLink to find the clusters within each main halo specified by 
    `snapshotFilePaths` and `particleName`. Then saves them in different formats 
    into the directory specified by `workingDirectoryPath`. `nSamples` is used 
    to format the cluster file names.
    """
    
    # The number of leading digits in the saved cluster file names
    sampleNumberFormat = np.log10(nSamples).astype(int) + 1

    # For tracking which star particles have been clustered over all snapshots (for FuzzyCat memory efficiency)
    veryLargeN = 10**8 # Must be larger than the maximum iord value
    particleIDsBool = np.zeros(veryLargeN, dtype = np.bool_)

    # Track cluster file names
    clusterFileNames = []

    # Cycle through each snapshot, run AstroLink, and save the clusters
    for index, snapshotFilePath in enumerate(snapshotFilePaths):
        print(f"Loading {snapshotFilePath.split('/')[-1]}                                                         \t\t", end = '\r')
        # Load the galaxy
        particleArr, particleIDs = loadGalaxyAsArrays(snapshotFilePath, particleName)

        print(f"Running AstroLink on the {particleName} particles of snapshot {snapshotFilePath.split('/')[-1]}   \t\t", end = '\r')
        # Run AstroLink and save the clusters in the snapshot
        c = AstroLink(particleArr)
        c.run()
        for clst, clst_id in zip(c.clusters[1:], c.ids[1:]):
            # Cluster file name
            clusterFileName = f"{index:0{sampleNumberFormat}}_{clst_id}.npy"
            clusterFileNames.append(clusterFileName)

            # Save the cluster with respect to the order of the data in the snapshot file
            cluster_raw = c.ordering[clst[0]:clst[1]]
            np.save(f"{workingDirectoryPath}Clusters_raw/{clusterFileName}", cluster_raw)

            # Save the cluster with respect to the particle IDs in the snapshot file
            cluster_iord = particleIDs[cluster_raw]
            np.save(f"{workingDirectoryPath}Clusters_iord/{clusterFileName}", cluster_iord)

            # Mark the particles that have been clustered
            particleIDsBool[cluster_iord] = 1
    
    # Save the IDs of the star particles that have been clustered
    clusteredIDs = np.where(particleIDsBool)[0]
    np.save(f"{workingDirectoryPath}clusteredIDs.npy", clusteredIDs)

    # Translate the clusters (with respect to the particle IDs) into reduced arrays (with respect to the order of the IDs of clustered particles) for improved memory efficiency with FuzzyCat
    for clusterFileName in clusterFileNames:
        cluster_iord = np.load(workingDirectoryPath + 'Clusters_iord/' + clusterFileName)
        cluster_reduced = np.where(np.isin(clusteredIDs, cluster_iord, assume_unique = True))[0].astype(cluster_iord.dtype)
        np.save(workingDirectoryPath + 'Clusters/' + clusterFileName, cluster_reduced)
```

The core of this method is simple; it loads the particle data, applies AstroLink to that data, and iteratively saves the clusters as .npy files. However, just saving the clusters directly from AstroLink (e.g. `cluster_raw`) means that each cluster file is an array of integers that can be used to slice the `particleArr` array and return the data points corresponding to the particles in each cluster &mdash; in reality, we need to account for the possibility of `particleArr` containing data points from different particles between snapshots. So, the additional code in this method translates the `cluster_raw` arrays into `cluster_iord` arrays (which contain the particle IDs of each particle in each cluster), and then finally into `cluster_reduced` arrays. This last translation isn't technically necessary, but serves to make FuzzyCat handle the clusters in a more memory efficiency way since the particle IDs can be very large in value and we only care about a fraction of the total number of particles in each simulation &mdash; in this case, only those of the particles that ever make it into the main halo.

With a series of reduced cluster files in 'Clusters' folder of the working directory, we can now run FuzzyCat and save its output.

```python
def runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, particleName, nSamples, minStability):
    """Runs FuzzyCat on the clusters contained in `workingDirectoryPath` with 
    parameters `nSamples` and `minStability`. The `nPoints` parameter is 
    determined automatically from a file containing the IDs of clustered 
    particles.
    """

    # Number of points clustered
    clusteredIDs = np.load(f"{workingDirectoryPath}clusteredIDs.npy")
    nPoints = clusteredIDs.size
    del clusteredIDs

    # Run FuzzyCat
    fc = FuzzyCat(nSamples, nPoints, workingDirectoryPath, minStability = minStability, checkpoint = True, verbose = 2)
    fc.run()

    # Plot the basic results
    FuzzyPlots.plotOrderedJaccardIndex(fc)
    FuzzyPlots.plotStabilities(fc)
    FuzzyPlots.plotMemberships(fc)

    # Save outputs
    np.save(f"{workingDirectoryPath}jaccardIndices.npy", fc.jaccardIndices)
    np.save(f"{workingDirectoryPath}ordering.npy", fc.ordering)
    np.save(f"{workingDirectoryPath}fuzzyClusters.npy", fc.fuzzyClusters)
    np.save(f"{workingDirectoryPath}stabilities.npy", fc.stabilities)
    np.save(f"{workingDirectoryPath}memberships.npy", fc.memberships)
    np.save(f"{workingDirectoryPath}memberships_flat.npy", fc.memberships_flat)
    np.save(f"{workingDirectoryPath}fuzzyHierarchy.npy", fc.fuzzyHierarchy)
    np.save(f"{workingDirectoryPath}groups.npy", fc.groups)
    np.save(f"{workingDirectoryPath}intraJaccardIndicesGroups.npy", fc.intraJaccardIndicesGroups)
    np.save(f"{workingDirectoryPath}interJaccardIndicesGroups.npy", fc.interJaccardIndicesGroups)
    np.save(f"{workingDirectoryPath}stabilitiesGroups.npy", fc.stabilitiesGroups)
```

That's all the methods we need to find a spatio-temporal clustering of a simulated galaxy. However, among other things, we also want to be able to visualise our results. So we need a plotting function...

```python
def plotFuzzyClustersOntoSnapshot(particleArr, clusters_raw, fuzzyLabels, workingDirectoryPath, snapshotFileName, axisLimits):
    """Creates a two-panel plot of the fuzzy clusters found by AstroLink and 
    FuzzyCat. The left panel is a 3D scatter plot and the right panel is a 
    top-down view of the region around the disk of the galaxy.
    """

    # Colour the data according to the fuzzy cluster
    colourList = [f"C{i}" for i in range(10) if i != 7]
    colours = np.zeros((particleArr.shape[0], 4))
    sizes = np.zeros(particleArr.shape[0])
    for cluster_raw, fuzzyLabel in zip(clusters_raw, fuzzyLabels):
        colours[cluster_raw] = col.to_rgba(colourList[fuzzyLabel%9], alpha = 1)
        sizes[cluster_raw] = 0.5
    
    # Create figure
    width, height = 16, 8
    figAspectRatio = height/width
    fig = plt.figure(figsize = (width, height))
    fig.patch.set_facecolor('k')

    # Plot the 3D data
    ax = fig.add_axes((0, 0, figAspectRatio, 1), projection = '3d')
    ax.scatter(*particleArr[:, :3].T, facecolors = colours, edgecolors = 'w', s = sizes, lw = 0.05)
    # Adjust data limits
    ax.set_xlim(-axisLimits, axisLimits)
    ax.set_ylim(-axisLimits, axisLimits)
    ax.set_zlim(-axisLimits, axisLimits)
    # Remove axes
    ax.axis('off')
    ax.patch.set_facecolor('k')
    # Add cartesian coordinate axes of length 100 kpc for reference
    ax.quiver([0]*6, [0]*6, [0]*6, [1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1],
              color = 'w', alpha = 1, length = 100, arrow_length_ratio = 0.1)
    ax.text(100, 0, 0, 'X', color = 'w')
    ax.text(0, 100, 0, 'Y', color = 'w')
    ax.text(0, 0, 100, 'Z', color = 'w')
    # Add zoom-in box around disk
    prismColour, prismAlpha = col.to_rgba('w', alpha = 0.2), 0.05
    xyRange, zRange, onesArray = np.array([-25, 25]), np.array([-5, 5]), np.ones(4).reshape(2, 2)
    for i in range(2):
        # z-direction faces
        xx, yy = np.meshgrid(xyRange, xyRange)
        ax.plot_wireframe(xx, yy, zRange[i]*onesArray, color = prismColour)
        ax.plot_surface(xx, yy, zRange[i]*onesArray, color = prismColour, alpha = prismAlpha)
        # x-direction faces
        xy, zz = np.meshgrid(xyRange, zRange)
        ax.plot_wireframe(xyRange[i]*onesArray, xy, zz, color = prismColour)
        ax.plot_surface(xyRange[i]*onesArray, xy, zz, color = prismColour, alpha = prismAlpha)
        # y-direction faces
        ax.plot_wireframe(xy, xyRange[i]*onesArray, zz, color = prismColour)
        ax.plot_surface(xy, xyRange[i]*onesArray, zz, color = prismColour, alpha = prismAlpha)

    # Plot the 2D disk data
    axisCentre, axisHalfWidth = 0.5 + figAspectRatio/2, 0.9*(1 - figAspectRatio)
    axDisk = fig.add_axes((axisCentre - axisHalfWidth/2,
                           0.5*(1 - axisHalfWidth/figAspectRatio),
                           axisHalfWidth,
                           axisHalfWidth/figAspectRatio))
    inBoxBool = (particleArr[:, 0] > xyRange[0])*(particleArr[:, 0] < xyRange[1]) # particles in x limits
    inBoxBool *= (particleArr[:, 1] > xyRange[0])*(particleArr[:, 1] < xyRange[1]) # particles in y limits
    inBoxBool *= (particleArr[:, 2] > zRange[0])*(particleArr[:, 2] < zRange[1]) # particles in z limits
    axDisk.scatter(*particleArr[inBoxBool, :2].T, facecolors = colours[inBoxBool], edgecolors = 'w', s = 2*sizes[inBoxBool], lw = 0.05)
    # Adjust data limits
    axDisk.set_xlim(xyRange[0], xyRange[1])
    axDisk.set_ylim(xyRange[0], xyRange[1])
    # Remove axes
    axDisk.patch.set_facecolor('k')
    for side in ['top', 'left', 'bottom', 'right']:
        axDisk.spines[side].set_color('w')

    # Adjust figure margins
    top, bottom, left, right = 1, 0, 0, 1
    fig.subplots_adjust(top = top, bottom = bottom, left = left, right = right)
    # Add snapshot number
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.text(0, 1, snapshotFileName, ha = 'left', va = 'top', fontsize = 10, color = 'w', transform = plt.gca().transAxes)
    # Save figure
    plt.savefig(f"{workingDirectoryPath}Cluster_plots/plotted_clusters_{snapshotFileName}.png", dpi = 200, bbox_inches = 'tight')
    fig.clf()
    plt.close()
    gc.collect()
```

... and a way to make a movie out of these plots for each snapshot so that we can watch our work.

```python
def makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate):
    """Makes a movie of the fuzzy clusters found by AstroLink and FuzzyCat as 
    they evolve over time.
    """

    clusterFileNames = np.load(workingDirectoryPath + 'clusterFileNames.npy')
    ordering = np.load(workingDirectoryPath + 'ordering.npy')
    fuzzyClusters = np.load(workingDirectoryPath + 'fuzzyClusters.npy')
    whichCluster = -np.ones(clusterFileNames.size, dtype = np.int32)
    for i, clst in enumerate(fuzzyClusters):
        whichCluster[ordering[clst[0]:clst[1]]] = i
    
    for index, snapshotFilePath in enumerate(snapshotFilePaths):
        print(f"Loading {snapshotFilePath.split('/')[-1]}                                                         \t\t", end = '\r')
        # Load the galaxy
        particleArr, _ = loadGalaxyAsArrays(snapshotFilePath, particleName)

        print(f"Loading clusters of {particleName} particles from snapshot {snapshotFilePath.split('/')[-1]}      \t\t", end = '\r')
        # Load AstroLink clusters (found in this snapshot) that belong to the fuzzy clusters from FuzzyCat
        clusters_raw, fuzzyLabels = [], []
        for clusterFileName, whichFuzzyClst in zip(clusterFileNames, whichCluster):
            clstSnapshot = int(clusterFileName.split('_')[0])
            if whichFuzzyClst != -1 and clstSnapshot == index:
                cluster_raw = np.load(workingDirectoryPath + 'Clusters_raw/' + clusterFileName)
                clusters_raw.append(cluster_raw)
                fuzzyLabels.append(whichFuzzyClst)
        
        print(f"Plotting {snapshotFilePath.split('/')[-1]} clusters                                               \t\t", end = '\r')
        snapshotFileName = snapshotFilePath.split('/')[-1]
        plotFuzzyClustersOntoSnapshot(particleArr, clusters_raw, fuzzyLabels, workingDirectoryPath, snapshotFileName, axisLimits)

    # Make movie
    import ffmpeg
    (
        ffmpeg
        .input(f"{workingDirectoryPath}Cluster_plots/plotted_clusters_*.png", pattern_type = 'glob', framerate = frameRate)
        .output(f"{workingDirectoryPath}{workingDirectoryPath.split('/')[-2]}_movie.mp4")
        .run()
    )
```

Lastly, we need to set up our file paths, calculate some properties for the above methods, and run the pipeline:

```python
if __name__ == '__main__':
    """Run spatio-temporal clustering pipeline :)
    """

    # Choose a particle from ['dark', 'stars', 'gas'] to cluster
    particleName = 'stars'

    # Set up the working directory
    galaxyFolderName = '8.26e11_zoom_2_new_run'
    workingDirectoryPath = f"/PATH/TO/YOUR/WORKING/DIRECTORY/nihao_uhd_{galaxyFolderName}_{particleName}/"
    if not os.path.exists(workingDirectoryPath):
        os.makedirs(workingDirectoryPath)
        os.makedirs(f"{workingDirectoryPath}Clusters_raw/")
        os.makedirs(f"{workingDirectoryPath}Clusters_iord/")
        os.makedirs(f"{workingDirectoryPath}Clusters/")
        os.makedirs(f"{workingDirectoryPath}Cluster_plots/")

    # Get the simulation snapshot file paths
    simulationDirectoryPath = f"/PATH/TO/YOUR/SIMULATION/DIRECTORY/nihao_uhd/{galaxyFolderName}/"
    snapshotFilePrefix = '8.26e11.'
    snapshotNumberRange = range(1164, 2001)
    snapshotFilePaths = [f"{simulationDirectoryPath}{snapshotFilePrefix}{i:05}" for i in snapshotNumberRange]

    # Info for the clustering pipeline
    nSamples = len(snapshotFilePaths)
    minLongevityOfFuzzyClusters = 230 # The minimum life-span of fuzzy clusters in Mega-years
    ageOfTheUniverse = 13800 # Age of the Universe in Mega-years
    # Calculate the minStability parameter so that fuzzy clusters live for at least `minLongevityOfFuzzyClusters` Myrs`
    minStability = minLongevityOfFuzzyClusters*(snapshotNumberRange.stop - 1)/(ageOfTheUniverse*snapshotNumberRange.step*nSamples)
    # Choose appropriate axis limits (in kpc) for the movie
    axisLimits = 150
    # Calculate movie frame rate so that 100 Myrs pass every second
    frameRate = 100*(snapshotNumberRange.stop - 1)/(ageOfTheUniverse*snapshotNumberRange.step)

    # Do clustering over snapshots with AstroLink
    findAndSaveClustersFromSnapshots(snapshotFilePaths, workingDirectoryPath, particleName, nSamples)

    # Run FuzzyCat on AstroLink clusters
    runFuzzyCatOnClustersFromSnapshots(workingDirectoryPath, particleName, nSamples, minStability)

    # Make movie of stable clusters over time
    makeMovieOfFuzzyClustersOverTime(snapshotFilePaths, workingDirectoryPath, particleName, axisLimits, frameRate)
```

## Analysis: Let's visualise the results!

If we run the above pipeline on the stellar particles of each of our NIHAO-UHD galaxies, then we get the following movies in the following subsections.

### g2.79e12

.. raw:: html

    <video controls src="./_static/nihao_uhd_2.79e12_zoom_6_rerun_stars_movie.mp4" alt="Phase-temporal clustering of the g2.79e12 NIHAO-UHD galaxy"/></video>


### g8.26e11

Movie coming...

### g1.12e12

.. raw:: html

    <video controls src="./_static/nihao_uhd_g1.12e12_3x9_stars_movie.mp4" alt="Phase-temporal clustering of the g1.12e12 NIHAO-UHD galaxy"/></video>

### g6.96e11

.. raw:: html

    <video controls src="./_static/nihao_uhd_g6.96e11_3x9_stars_movie.mp4" alt="Phase-temporal clustering of the g6.96e11 NIHAO-UHD galaxy"/></video>


### g7.08e11

.. raw:: html

    <video controls src="./_static/nihao_uhd_g7.08e11_5x10_stars_movie.mp4" alt="Phase-temporal clustering of the g7.08e11 NIHAO-UHD galaxy"/></video>


### g7.55e11

.. raw:: html

    <video controls src="./_static/nihao_uhd_g7.55e11_3x9_stars_movie.mp4" alt="Phase-temporal clustering of the g7.55e11 NIHAO-UHD galaxy"/></video>
