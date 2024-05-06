# Welcome to FuzzyCat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/fuzzycat/ci.yml?branch=main)](https://github.com/william-h-oliver/fuzzycat/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fuzzycat/badge/)](https://fuzzycat.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/fuzzycat/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/fuzzycat)

## Installation

The Python package `fuzzycat` can be installed from PyPI:

```
python -m pip install fuzzy-cat
```

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
