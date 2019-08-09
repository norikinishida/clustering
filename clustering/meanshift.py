from sklearn import cluster

from .baseclass import ClusteringModel

class MeanShift(ClusteringModel):

    def __init__(self):
        super(MeanShift, self).__init__()
        pass

    def fit(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N, dim), dtype=float)
        :rtype: None
        """
        self.bandwidth = cluster.estimate_bandwidth(vectors, quantile=0.2, n_samples=int(0.5 * len(vectors)))
        self.model = cluster.MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
        self.model.fit(vectors)
        self.n_clusters = self.get_cluster_centers().shape[0]

    def get_cluster_assignments(self):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.labels_

    def get_cluster_centers(self):
        """
        :rtype: numpy.ndarray(shape=(n_clusters,dim), dtype=float)
        """
        return self.model.cluster_centers_

    def get_cluster_covariances(self):
        """
        :rtype: None
        """
        return None

    def predict_clusters(self, vectors):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.predict(vectors)

