import numpy as np

import visualizers
import dimensionreduction

import clustering

PATH_MNIST = "/mnt/hdd/dataset/MNIST/images.test.txt"

#######################
# Data preparation
vectors = []
for line in open(PATH_MNIST):
    vector = line.strip().split()
    vector = [float(x) for x in vector]
    vectors.append(vector)
vectors = np.asarray(vectors)
print("# of instances=%d" % vectors.shape[0])
print("Dimensionality=%d" % vectors.shape[1])

#######################
# Dimension reduction
print("Reducing dimensionality ...")
vectors = dimensionreduction.reduce_dimension(
            vectors=vectors,
            method="pca",
            pre_normalization=False,
            target_dim=2,
            params={})
print("Dimensionality=%d" % vectors.shape[1])

#######################
# Clustering
print("Clustering ...")
model = clustering.clustering(
            vectors=vectors,
            method="gmm",
            params={"n_clusters": 10, "covariance_type": "full"})
cluster_ids = model.get_cluster_assignments()
cluster_names = np.asarray(["C%s" % c_id for c_id in cluster_ids])
cluster_centers = model.get_cluster_centers()
cluster_covariances = model.get_cluster_covariances()
print("# of clusters=%d" % model.n_clusters)

#######################
# Visualization
print("Visualizing ...")
cluster_order = ["C%s" % c_id for c_id in range(model.n_clusters)]
visualizers.scatter(
        vectors=vectors,
        categories=cluster_names, category_name="Cluster", category_order=cluster_order,
        category_centers=cluster_centers, category_covariances=cluster_covariances,
        xlabel="$x$", ylabel="$y$",
        fontsize=30,
        savepath="./mnist.png", figsize=(15,12), dpi=100)


