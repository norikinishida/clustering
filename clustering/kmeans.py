from sklearn import cluster

from .baseclass import ClusteringModel

class KMeans(ClusteringModel):

    def __init__(self, n_clusters):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.model = cluster.KMeans(n_clusters=n_clusters, random_state=0)

    def fit(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N, dim), dtype=float)
        :rtype: None
        """
        self.model.fit(vectors)

    def get_cluster_assignments(self):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.labels_

    def get_cluster_centers(self):
        """
        :rtype: numpy.ndarray(shape=(n_clusters, dim), dtype=float)
        """
        return self.model.cluster_centers_

    def get_cluster_covariances(self):
        """
        :rtype None
        """
        return None

    def predict_clusters(self, vectors):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.predict(vectors)

