import numpy as np

from sklearn import cluster

from .baseclass import ClusteringModel

class AffinityPropagation(ClusteringModel):

    def __init__(self, preference):
        super(ClusteringModel, self).__init__()
        self.preference = preference
        self.model = cluster.AffinityPropagation(preference=preference)

    def fit(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N,dim), dtype=float)
        :rtype: None
        """
        self.model.fit(vectors)
        self.vectors = vectors
        self.n_clusters = self.get_cluster_centers().shape[0]

    def get_cluster_assignments(self):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.labels_

    def get_cluster_centers(self):
        """
        :rtype: numpy.ndarray(shape=(n_clusters, dim), dtype=float)
        """
        indices = self.model.cluster_centers_indices_
        return np.asarray([self.vectors[idx] for idx in indices])

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

