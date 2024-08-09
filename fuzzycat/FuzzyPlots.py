#Author: William H. Oliver <william.hardie.oliver@gmail.com>
#License: MIT


# Standard libraries
import gc

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def plotOrderedJaccardIndex(fc, figsize = (8, 8), linewidth = 0.5, save = True, show = False, dpi = 400):
    """Creates a the ordered Jaccard Index plot and overlays the fuzzy clusters 
    that have been determined by FuzzyCat.

    Parameters
    ----------
    fc : FuzzyCat
        An instance of the FuzzyCat class.
    figsize : tuple, default is (8, 8)
        The size of the figure in inches.
    linewidth : float, default is 0.5
        The width of the line.
    save : bool, default is True
        If True, save the figure to the directory stored in `fc.directoryName`.
    show : bool, default is False
        If True, display the figure.
    dpi : int, default is 400
        The resolution of the figure.
    """

    # Check parameters
    check_fc = hasattr(fc, 'jaccardIndices') and hasattr(fc, 'ordering') and hasattr(fc, 'fuzzyClusters') and hasattr(fc, 'directoryName')
    assert check_fc, "Parameter 'fc' must be an instance of the FuzzyCat class and have been used to find fuzzy clusters!"

    # Create a figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Plot the ordered Jaccard Indices
    jacc_ordered = fc.jaccardIndices[fc.ordering]
    ax.plot(range(jacc_ordered.size), jacc_ordered, 'k-', lw = linewidth, zorder = 2)
    for i, clst in enumerate(fc.fuzzyClusters):
        ax.fill_between(range(clst[0], clst[1]), jacc_ordered[clst[0]:clst[1]], 0, color = f"C{i%10}", zorder = 1)

    # Tidy up
    ax.set_xlabel('Ordered Index')
    ax.set_ylabel('Jaccard Index')
    ax.set_xlim(0, jacc_ordered.size - 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])

    # Save, show and close
    if save: plt.savefig(fc.directoryName + 'OrderedJaccardIndex', dpi = dpi)
    if show: plt.show()
    fig.clf()
    plt.close()
    gc.collect()

def plotStabilities(fc, figsize = (8, 8), bins = None, save = True, show = False, dpi = 400):
    """Creates a histogram of the stabilities of the fuzzy clusters that have
    been determined by FuzzyCat.

    Parameters
    ----------
    fc : FuzzyCat
        An instance of the FuzzyCat class.
    figsize : tuple, default is (8, 8)
        The size of the figure in inches.
    bins : array-like, str, or None, default is None
        The bins used in the histogram. If None, 10 bins are set to be 0.1 wide
        spanning the range [0, 1].
    save : bool, default is True
        If True, save the figure to the directory stored in `fc.directoryName`.
    show : bool, default is False
        If True, display the figure.
    dpi : int, default is 400
        The resolution of the figure.
    """

    # Check parameters
    check_fc = hasattr(fc, 'stabilities') and hasattr(fc, 'directoryName')
    assert check_fc, "Parameter 'fc' must be an instance of the FuzzyCat class and have been used to find fuzzy clusters!"

    # Create a figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Plot the histogram of stabilities
    if bins is None: bins = np.linspace(0, 1, 11)
    ax.hist(fc.stabilities, bins = bins, color = 'slategrey', density = True, histtype = 'step', lw = 1, zorder = 2)

    # Tidy up
    ax.set_xlabel('Stability')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yticks([])

    # Save, show and close
    if save: plt.savefig(fc.directoryName + 'Stabilities', dpi = dpi)
    if show: plt.show()
    fig.clf()
    plt.close()
    gc.collect()

def plotMemberships(fc, figsize = (8, 8), bins = None, save = True, show = False, dpi = 400):
    """Creates a histogram of the memberships of the fuzzy clusters that have
    been determined by FuzzyCat.

    Parameters
    ----------
    fc : FuzzyCat
        An instance of the FuzzyCat class.
    figsize : tuple, default is (8, 8)
        The size of the figure in inches.
    bins : array-like or None, default is None
        The bins used in the histogram. If None, 10 bins are set to be 0.1 wide
        spanning the range [0, 1].
    save : bool, default is True
        If True, save the figure to the directory stored in `fc.directoryName`.
    show : bool, default is False
        If True, display the figure.
    dpi : int, default is 400
        The resolution of the figure.
    """

    # Check parameters
    check_fc = hasattr(fc, 'memberships') and hasattr(fc, 'directoryName')
    assert check_fc, "Parameter 'fc' must be an instance of the FuzzyCat class and have been used to find fuzzy clusters!"

    # Create a figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Plot the histogram of memberships
    if bins is None: bins = np.linspace(0, 1, 101)
    for i, probs in enumerate(fc.memberships):
        ax.hist(probs, bins = bins, color = f"C{i%10}", alpha = 0.5, density = True, histtype = 'step', lw = 1, zorder = 2)

    # Tidy up
    ax.set_xlabel('Membership')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))

    # Save, show and close
    if save: plt.savefig(fc.directoryName + 'Memberships', dpi = dpi)
    if show: plt.show()
    fig.clf()
    plt.close()
    gc.collect()

def plotFuzzyLabelsOnX(fc, X, membersOnly = False, figsize = (8, 8), markerSize = 5, save = True, show = False, dpi = None):
    """Creates a scatter plot of the data points in `X` and colours them
    according to the fuzzy clusters that have been determined by FuzzyCat.

    Parameters
    ----------
    fc : FuzzyCat
        An instance of the FuzzyCat class.
    X : `numpy.ndarray`
        The data points that are to be displayed in the figure.
    figsize : tuple, default is (8, 8)
        The size of the figure in inches.
    markerSize : int or float, default is 5
        The size of the markers in the scatter plot.
    save : bool, default is True
        If True, save the figure to the directory stored in `fc.directoryName`.
    show : bool, default is False
        If True, display the figure.
    dpi : int or None, default is None
        The resolution of the figure.
    """

    # Check parameters
    check_fc = hasattr(fc, 'nPoints') and hasattr(fc, 'memberships_flat') and hasattr(fc, 'stabilities') and hasattr(fc, 'directoryName')
    assert check_fc, "Parameter 'fc' must be an instance of the FuzzyCat class and have been used to find fuzzy clusters!"

    check_X = isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == fc.nPoints and (X.shape[1] == 2 or X.shape[1] == 3)
    assert check_X, "Parameter 'X' must be a numpy array that is a 2D or 3D representation of the fuzzy data used for clustering!"

    # Create a figure and axis
    if X.shape[1] == 2: fig, ax = plt.subplots(figsize = figsize)
    elif X.shape[1] == 3: fig, ax = plt.subplots(figsize = figsize, subplot_kw = {'projection': '3d'})
    else: raise ValueError("Parameter 'X' must represent 2D or 3D data points!")

    # Prepare the colours
    colours = np.zeros((X.shape[0], 4))
    colsArr = np.array([col.to_rgba(f"C{i%10}") for i in range(10)])
    for i, (membershipArr, stability) in enumerate(zip(fc.memberships_flat, fc.stabilities)):
        colours += stability*membershipArr[:, np.newaxis]*colsArr[i%10]
    if not membersOnly: colours[:, -1] = 1

    # Plot the data points
    ax.scatter(*X.T, s = markerSize, facecolor = colours, edgecolor = 'none', zorder = 1)

    # Tidy up
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if X.shape[1] == 3: ax.set_zlabel('z')

    # Save, show and close
    if save:
        if dpi is None:
            plt.savefig(fc.directoryName + 'FuzzyLabels')
        else:
            plt.savefig(fc.directoryName + 'FuzzyLabels', dpi = dpi)
    if show: plt.show()
    fig.clf()
    plt.close()
    gc.collect()
