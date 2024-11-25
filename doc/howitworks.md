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

We are still not convinced from the visualisation of the results that any one of _k_ values is better than another. Following the literature, it is suggested that we should look to the plots of the sum of squared errors and the silhouette coefficient as functions of _k_.

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_elbow.png" alt="The k-means elbow plot."/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_silhouette.png" alt="Silhouette coefficient vs k."/>
</p>

Typically, one chooses a value of _k_ by looking for an _elbow_ in the first plot and a peak in the second. However it is not obvious for which _k_ values these features occur. Perhaps no clusters actually exist and we are clasping at straws or perhaps k-means just isn't appropriate to be used on this data set (both are quite valid points in this case, but we have an algorithm to explain...), or perhaps a respectable solution exists and we are just limited by having to make a choice for _k_.

If it is indeed the latter, then we would like to abstract the effect of a changing _k_ value into a representation of k-means-like clusters that remain stable over a range of _k_ values &mdash; this is a task for FuzzyCat!



## The FuzzyCat process

In order to run the FuzzyCat algorithm to these clustering results, we first need to translate them into the necessary format, i.e. we need a record of which points belonged to which clusters in which clustering instances. This information can be summarised for the above clustering problem as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_clustering_info.png" alt="Clustering information that FuzzyCat needs."/>
</p>

### A similarity matrix for clusters

With this table of clustering information, FuzzyCat then computes the matrix of Jaccard indices calculated for every pair of clusters. For two clusters, the Jaccard index is their intersection divided by their union &mdash; which for the first two clusters in the table above is 0 (as they do not intersect), but for the first and third clusters it is 16/17 $\approx$ 0.94. As such, the Jaccard index matrix for our problem looks like...

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/k_means_jaccard_matrix.png" alt="Jaccard index matrix computed by FuzzyCat."/>
</p>

This matrix is symmetric and the elements of the main diagonal are trivially equal to 1, so FuzzyCat only actually needs to compute the upper right (or equivalently, lower left) $n(n - 1)/2$ elements of this matrix.

### Finding a representation of the fuzzy clustering structure

Each of these Jaccard indices are now treated as edge weights that connect each of the input clusters, and with these edges, FuzzyCat then finds all possible _overdensities_ from the Jaccard space of all clusters and clusterings. Unfortunately we can't easily visualise this process as it operates within FuzzyCat because clusters inhabit a very high dimensional space. Luckily however, this process is nearly identical to the AstroLink aggregation process (described in section 3.4 of the original [science paper](https://doi.org/10.1093/mnras/stae1029) and within its [documentation](https://astrolink.readthedocs.io/en/latest/howitworks.html#compute-an-ordered-list-and-a-binary-tree-of-groups)) and since it is much simpler to visualise a 2-dimensional toy data set that a very high dimensional space of all possible clusters, we first look at the way this process works within AstroLink.

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/aggregation_process_example.png" alt="Aggregation process illustration from AstroLink."/>
</p>

As mentioned, this process operates in almost exactly the same way within FuzzyCat as it does within AstroLink. Hence the visualisation for AstroLink version can be thought of a kind of reduced-dimensionality version of what happens inside FuzzyCat. Without loss of generality for either algorithm...

- A list of edges (denoted E above) between each object of the input data set are traversed from high edge weight to low edge weight. In AstroLink, the weights of these edges are related to the local-density of the input data, while in FuzzyCat, these edges are the aforementioned Jaccard index values that are calculated for each pair of clusters.

- During this edge-wise traversal, ordered lists of the data objects are built to develop a representation of the structure of the input data. A new ordered list of objects is formed whenever neither of the two objects connected by an edge are found to belong to an existing ordered list.

- As more edges are traversed, these ordered lists grow as more objects are connected to them. Eventually, an edge is processed that would merge two separate ordered lists. In this moment, the two ordered lists represent potentially-informative overdensities, so both AstroLink and FuzzyCat store them for later consideration. After this, the two ordered lists are merged into one, and the process continues until all edges have been processed and one ordered list remains.

Besides the meaning of _data objects_, there are two differences between FuzzyCat and AstroLink with regards to this process...

1. FuzzyCat doesn't need to undertake any equivalent of step 6, since the edges from the Jaccard index matrix denote a densely-connected graph; and
2. Unlike Astrolink, FuzzyCat doesn't keep track of two separate sets of connected components, $G_\geq$ and $G_\leq$, instead all connected components are stored in one list, G.

The latter is because of the extraction procedure that follows and the fact that FuzzyCat doesn't need to differentiate between the larger and smaller of any two connected components within this extraction procedure.

The resulting information gained from this process is a representation of all overdensities of data objects, which for FuzzyCat is the ordered-Jaccard plot ans the list of all possible fuzzy clusters. Returning to our k-means clustering problem, this plot appears as follows.

**ADD figure of ordered-Jaccard plot with all groups**

The ordered-Jaccard plot is notably less intuitive than the ordered-density plot found by AstroLink, but we see that that some structure amongst the chaos has been found.

### Extracting meaningful fuzzy clusters

With a list of all possible fuzzy clusters, FuzzyCat now extracts only those fuzzy clusters that satisfies certain constraints. We want the final fuzzy clusters to represent stable clusters of the underlying data set. **TO BE CONTINUED**

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/OrderedJaccardIndex.png" alt="Ordered-Jaccard plot."/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/FuzzyLabels.png" alt="Ordered-Jaccard plot."/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/william-h-oliver/fuzzycat/main/images/howitworks/ExplodedFuzzyLabels.png" alt="Ordered-Jaccard plot."/>
</p>