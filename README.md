# Welcome to FuzzyCat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/fuzzycat/ci.yml?branch=main)](https://github.com/william-h-oliver/fuzzycat/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fuzzycat/badge/)](https://fuzzycat.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/fuzzycat/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/fuzzycat)

FuzzyCat is a general-purpose soft-clustering algorithm that, given a series of clusterings on point-based data, is able to produce data-driven fuzzy clusters whose membership functions encapsulate the effects of any changes in the feature space of the data between clusterings. The fuzzy clusters are produced empirically by finding groups of clusters within the many clusterings. The different clusterings may be governed by any underlying process that affects the clusters (e.g. stochastic sampling from uncertain data, temporal evolution of the data, clustering algorithm hyperparameter variation, etc.). In effect, FuzzyCat propagates the effects of the underlying process(es) into a soft-clustering which has had these effects abstracted away into the membership functions of the original point-based data.

## Installation

The Python package `fuzzycat` can be installed from PyPI:

```
python -m pip install fuzzy-cat
```

## Basic Usage



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
