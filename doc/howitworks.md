# How FuzzyCat Works

FuzzyCat is an unsupervised general-purpose soft-clustering algorithm that, given a series of clusterings on object-based data, produces data-driven fuzzy clusters whose membership functions encapsulate the effects of changes in the clusters due to some underlying process. In effect, FuzzyCat propagates these effects into a soft-clustering which has had these effects abstracted away into the membership functions of the original object-based data.

The goal of this page is to provide an intuitive overview of how the FuzzyCat algorithm works. To aid the explanation, we will first look at a simple clustering problem that has no obvious solution &mdash; with which we will later see that FuzzyCat can help.



## A seemingly simple clustering problem

Imagine that we wanted to find the clusters from this 2-dimensional toy data set.

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/data.png" alt="Toy data set with an ambiguous structure."/>
</p>

The data is very simple and so we decide to use a simple algorithm, k-means. [K-means](https://en.wikipedia.org/wiki/K-means_clustering) is an iterative clustering algorithm that partitions a dataset into _k_ distinct clusters by minimizing the sum of squared distances between data points and their respective cluster centroids. However the structure within this data set is somewhat ambiguous in nature, and so we immediately run into the problem of choosing an appropriate value for _k_. Expecting that an appropriate value exists somewhere in the range of [1, 10], we apply k-means with each of these values to the data. The clusterings look appear as follows...

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_clusterings.gif" alt="A gif showing the clusterings of the toy data that result from applying k-means with various values of k."/>
</p>

Still not convinced by the results of using any one _k_ value, the literature suggests that we should look to the plots of the sum of squared errors and the silhouette coefficient as functions of _k_.

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_elbow.png" alt="The k-means elbow plot."/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_silhouette.png" alt="Silhouette coefficient vs k"/>
</p>

Typically, one chooses a value of _k_ by looking for an _elbow_ in the first plot and a peak in the second. However it is not obvious for which _k_ values these features occur. Perhaps no clusters actually exist and we are clasping at straws or perhaps k-means just isn't appropriate to be used on this data set (both are quite valid points in this case, but we have an algorithm to explain...), or perhaps a respectable solution exists and we are just limited by having to make a choice for _k_.

If it is indeed the latter, then we would like abstract the effect of a changing _k_ value into a representation of k-means clusters that remain stable over a wide range of _k_ values &mdash; this is a task for FuzzyCat!



## The FuzzyCat process

In order to run the FuzzyCat algorithm to these clustering results, we first need to translate them into the necessary format, i.e. we need a record of which points belonged to which clusters in which clustering instances. This information can be summarised for the above clustering problem as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_clustering_info.png" alt="Silhouette coefficient vs k"/>
</p>

With this table of clustering information, FuzzyCat then computes the matrix of Jaccard indices calculated for every pair of clusters. For two sets, the Jaccard index is their intersection divided by their union &mdash; which for the first two clusters is 7/16 $\approx$ 0.44. The Jaccard index matrix for our problem looks like...

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_jaccard_matrix.png" alt="Silhouette coefficient vs k"/>
</p>

This matrix is symmetric and the elements of the main diagonal are trivially 1, so we only actually need to compute the upper right (or equivalently, lower left) $n(n - 1)/2$ elements of this matrix. With this, FuzzyCat finds all possible _overdensities_ from the Jaccard space of all clusters and clusterings. This process is nearly identical to the AstroLink aggregation process, described in the original [science paper](https://doi.org/10.1093/mnras/stae1029) and within its [documentation](https://astrolink.readthedocs.io/). The way in which this process works within FuzzyCat is illustrated by the following animation.

**To be continued...**
