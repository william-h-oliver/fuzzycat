# Welcome to FuzzyCat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/fuzzy_cat/ci.yml?branch=main)](https://github.com/william-h-oliver/fuzzy_cat/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fuzzy_cat/badge/)](https://fuzzy_cat.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/fuzzy_cat/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/fuzzy_cat)

## Installation

The Python package `fuzzy_cat` can be installed from PyPI:

```
python -m pip install fuzzy-cat
```

## Development installation

If you want to contribute to the development of `fuzzy_cat`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/william-h-oliver/fuzzy_cat.git
cd fuzzy_cat
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
