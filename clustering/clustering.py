from .kmeans import KMeans
from .affinitypropagation import AffinityPropagation
from .meanshift import MeanShift
from .agglomerative import AgglomerativeClustering
from .gmm import GaussianMixtureModel

def clustering(vectors, method, params={}):
    """
    :type vectors: numpy.ndarray(shape=(N, dim), dtype=float)
    :type method: str
    :type params: {str: Any}
    :rtype: ClusteringModel
    """
    # Clustering
    if method == "kmeans":
        model = KMeans(n_clusters=params["n_clusters"])
    elif method == "gmm":
        model = GaussianMixtureModel(n_clusters=params["n_clusters"],
                                     covariance_type=params["covariance_type"])
    elif method == "affinitypropagation":
        model = AffinityPropagation(preference=params["preference"])
    elif method == "meanshift":
        model = MeanShift()
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=params["n_clusters"],
                                        linkage=params["linkage"],
                                        n_neighbors=params["n_neighbors"])
    else:
        raise ValueError("Unknown method=%s" % method)

    model.fit(vectors)

    return model
