# Welcome to FuzzyCat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/fuzzycat/ci.yml?branch=main)](https://github.com/william-h-oliver/fuzzycat/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fuzzycat/badge/)](https://fuzzycat.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/fuzzycat/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/fuzzycat)

FuzzyCat is a general-purpose soft-clustering algorithm that, given a series of clusterings on point-based data, is able to produce data-driven fuzzy clusters whose membership functions encapsulate the effects of any changes in the clusterings due to changes in the feature space of the data. The fuzzy clusters are produced empirically by finding groups of clusters within the many clusterings. The different clusterings may be governed by any underlying process that affects the clusters (e.g. stochastic sampling from uncertain data, temporal evolution of the data, clustering algorithm hyperparameter variation, etc.). In effect, FuzzyCat propagates the effects of the underlying process(es) into a soft-clustering which has had these effects abstracted away into the membership functions of the original point-based data.

The FuzzyCat documentation can be found on [ReadTheDocs](https://fuzzycat.readthedocs.io/).

## Installation

The Python package `fuzzycat` can be installed from PyPI:

```
python -m pip install fuzzycategories
```

## Basic usage

FuzzyCat can be easily applied to any series of clusters that have been found from different representations of fuzzy point-based data. If the fuzzy data is actually uncertain data then these representations could (for example) be a series of random samples generated by sampling independently over each point's probability distribution. In such a scenario, the clusters found from each representation would differ by some amount that depends on the effect that the uncertainties have on the structure within the data set.


### Getting some fuzzy data
To demonstrate this, we first need some data...

```python
import numpy as np
import sklearn.datasets as data

# Generate some structured data with noise
np.random.seed(0)
background = np.random.uniform(-2, 2, (1000, 2))
moons, _ = data.make_moons(n_samples = 2000, noise = 0.1)
moons -= np.array([[0.5, 0.25]])    # centres moons on origin
gauss_1 = np.random.normal(-1.25, 0.2, (500, 2))
gauss_2 = np.random.normal(1.25, 0.2, (500, 2))

P = np.vstack([background, moons, gauss_1, gauss_2])
```

... however, this is not *fuzzy* data.

To make it fuzzy (in this scenario), we also need some description of the probability distribution of each point. The simplest version of this is to take the probability distributions as homogenous and spherically-symmetric 2-dimensional Gaussians. This would mean that the uncertainty of every point is described by Gaussian probability distributions, each having the same covariance matrix, i.e. $\sigma^2 I$ (where $\sigma$ is a constant and $I$ is the identity matrix).

So let's simply take $\sigma = 0.05$ by setting a variable `covP = 0.05`.

### Generating different representations of the fuzzy data
In our scenario, we can generate the different representations by creating random samples of the data. Luckily, for Gaussian uncertainties, FuzzyCat comes prepared with a utility function to do this for us. At a minimum, it requires the mean-values, `P`, and the covariances, `covP`, which will produce 100 representations, run the [AstroLink](https://github.com/william-h-oliver/astrolink) algorithm with its default parameters over each, and save the resultant clusters in the correct format within a new 'Clusters/' folder that will be located within the current directory.

The code to do this is simply...

```python
from fuzzycat import FuzzyData

FuzzyData.clusteringsFromRandomSamples(P, covP)
```

> [!NOTE]
> `covP` can be also be a 1-, 2-, or 3-dimensional `np.ndarray`.

For clarity, here's a gif showing the clusterings produced for each realisation...

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/readme/Resampled_Clusterings.gif" alt="A gif showing the random sample clusterings from AstroLink."/>
</p>


### Applying FuzzyCat
We have now generated various clusterings from our fuzzy data, and we can see that the uncertainities are affect the clusters as the clusters change between resamplings. By applying FuzzyCat, we can collate this information into one soft clustering that encapsulates these effects.

We just need to tell it that we have `nSamples = 100` clusterings of `nPoints = P.shape[0]` (= 4000) points and then run it...

```python
from fuzzycat import FuzzyCat

nSamples, nPoints = 100, P.shape[0]
fc = FuzzyCat(nSamples, nPoints)
fc.run()
```

... and its done! FuzzyCat has found a representative soft clustering that has propagated the effects of the uncertainties into the AstroLink cluster model.

### Visualising the soft clusters
With the soft clustering found, we would like to see what it looks like. This is easy, because FuzzyCat also comes equipped with some useful plotting functions. As such, we can visualise the soft structure with...

```python
from fuzzycat import FuzzyPlots

FuzzyPlots.plotFuzzyLabelsOnX(fc, P, membersOnly = True)
```

... which produces a figure whereby the colour and alpha value of the points of `P` are defined according to their membership within each of the fuzzy clusters. For this scenario, we get...

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/readme/Fuzzy_Labels_clustered_only.png" alt="AstroLink clusters with progagated uncertainties."/>
</p>

... which shows that the effect of these uncertainties on the AstroLink clusters is to give them *fuzzy borders* &mdash; indicated by the colours of the points fading to black and/or mixing around the boundaries of these clusters.

## Development installation

If you want to contribute to the development of `fuzzycat`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/william-h-oliver/fuzzycat.git
cd fuzzycat
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
