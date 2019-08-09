import numpy as np
from sklearn import mixture

from .baseclass import ClusteringModel

class GaussianMixtureModel(ClusteringModel):

    def __init__(self, n_clusters, covariance_type):
        super(ClusteringModel, self).__init__()
        assert covariance_type in ["full", "tied", "diag", "spherical"]
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.model = mixture.GaussianMixture(n_components=n_clusters,
                                            covariance_type=covariance_type)

    def fit(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N, dim), dtype=float)
        :rtype: None
        """
        self.model.fit(vectors)
        self.labels = self.model.predict(vectors)
        self.dim = vectors.shape[1]

    def get_cluster_assignments(self):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.labels

    def get_cluster_centers(self):
        """
        :rtype: numpy.ndarray(shape=(n_clusters, dim), dtype=float)
        """
        return self.model.means_

    def get_cluster_covariances(self):
        """
        :rtype: numpy.ndarray(shape=(n_clusters, dim, dim), dtype=float)
        """
        cov = np.zeros((self.n_clusters, self.dim, self.dim))
        if self.covariance_type == "full":
            cov = self.model.covariances_
        elif self.covariance_type == "tied":
            for i in range(self.n_clusters):
                cov[i,:,:] = self.model.covariances_[:,:]
        elif self.covariance_type == "diag":
            for i in range(self.n_clusters):
                cov[i,:,:] = np.diag(self.model.covariances_[i][:])
        elif self.covariance_type == "spherical":
            for i in range(self.n_clusters):
                cov[i,:,:] = np.eye(self.dim) * self.model.covariances_[i]
        return cov

    def predict_clusters(self, vectors):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.predict(vectors)

