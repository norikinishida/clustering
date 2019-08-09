import itertools

from sklearn import cluster, neighbors

from .baseclass import ClusteringModel

class Node(object):

    def __init__(self, node_id, left, right, is_terminal):
        self.node_id = node_id
        self.left = left
        self.right = right
        self.is_terminal = is_terminal

    def __str__(self):
        if self.is_terminal:
            return str(self.node_id)
        else:
            return "( %s %s )" % (self.left, self.right)

class AgglomerativeClustering(ClusteringModel):

    def __init__(self, n_clusters, linkage, n_neighbors=10):
        super(AgglomerativeClustering, self).__init__()
        assert linkage in ["ward", "average", "complete"]
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.n_neighbors = n_neighbors

    def fit(self, vectors):
        """
        :type vectors: numpy.ndarray(shape=(N, dim), dtype=float)
        :rtype: None
        """
        self.connectivity = neighbors.kneighbors_graph(
                                vectors,
                                n_neighbors=self.n_neighbors,
                                include_self=False)
        self.connectivity = 0.5 * (self.connectivity + self.connectivity.T)
        self.model = cluster.AgglomerativeClustering(
                                n_clusters=self.n_clusters,
                                linkage=self.linkage,
                                connectivity=self.connectivity)
        self.model.fit(vectors)
        self.n_samples = vectors.shape[0]
        # MEMO
        # self.model.n_leaves_
        # self.model.n_components_
        # self.model.children_

    def get_cluster_assignments(self):
        """
        :rtype: numpy.ndarray(shape=(N,), dtype=int)
        """
        return self.model.labels_

    def get_cluster_centers(self):
        """
        :rtype: None
        """
        return None

    def get_cluster_covariances(self):
        """
        :rtype: None
        """
        return None

    def predict_clusters(self, vectors):
        """
        :rtype: None
        """
        return None

    def get_tree_sexp(self):
        """
        :rtype: str
        """

        memo = {}
        for node_id in range(self.n_samples):
            node = Node(node_id=node_id, left=None, right=None, is_terminal=True)
            memo[node_id] = node

        ii = itertools.count(self.n_samples)
        for left_id, right_id in self.model.children_:
            assert left_id in memo
            assert right_id in memo
            left = memo[left_id]
            right = memo[right_id]
            node_id = next(ii)
            node = Node(node_id=node_id, left=left, right=right, is_terminal=False)
            memo[node_id] = node

        return str(node)

