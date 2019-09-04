from sklearn.datasets.samples_generator import make_blobs

import treetk

import clustering

# Data preparation
vectors, _ = make_blobs(n_samples=50,
                        centers=[[1.0,1.0],[1.0,-1.0],[-1.0,1.0],[-1.0,-1.0]],
                        cluster_std=0.5)

# Test K-Means
model = clustering.clustering(
            vectors=vectors,
            method="kmeans",
            params={"n_clusters": 4})
cluster_ids = model.get_cluster_assignments()
cluster_centers = model.get_cluster_centers()
predicted_cluster_ids = model.predict_clusters(vectors)

# Test Gaussian Mixture Model
model = clustering.clustering(
            vectors=vectors,
            method="gmm",
            params={"n_clusters": 4, "covariance_type": "full"})
cluster_ids = model.get_cluster_assignments()
cluster_centers = model.get_cluster_centers()
cluster_covariances = model.get_cluster_covariances()
predicted_cluster_ids = model.predict_clusters(vectors)

# Test Affinity Propagation
model = clustering.clustering(
            vectors=vectors,
            method="affinitypropagation",
            params={"preference": -50})
cluster_ids = model.get_cluster_assignments()
cluster_centers = model.get_cluster_centers()
predicted_cluster_ids = model.predict_clusters(vectors)

# Test Mean Shift
model = clustering.clustering(
            vectors=vectors,
            method="meanshift",
            params={})
cluster_ids = model.get_cluster_assignments()
cluster_centers = model.get_cluster_centers()
predicted_cluster_ids = model.predict_clusters(vectors)

# Test Agglomerative Clustering
model = clustering.clustering(
            vectors=vectors,
            method="agglomerative",
            params={"n_clusters": 4, "linkage": "ward", "n_neighbors": 10})
cluster_ids = model.get_cluster_assignments()
sexp = model.get_tree_sexp()
tree = treetk.sexp2tree(treetk.preprocess(sexp), with_nonterminal_labels=False, with_terminal_labels=False)
# treetk.pretty_print(tree)

print("OK")
