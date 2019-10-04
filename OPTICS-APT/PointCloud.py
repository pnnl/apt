"""
A class describes 3 dimensional point cloud. 

Authors: Jing Wang

This is a version modified for python 3 syntax.
"""
import numpy as np
from sklearn.neighbors import KDTree

__version__ = '0.2'

class PointCloud:
    """
    A class for 3D point cloud.
    """
    def __init__(self):
        """
        Initialize point cloud class. Taking a numpy array m by n size, \
        where m is the number of data points, n is the attribute of each data point.

        Note data size is not checked, it can be anything eligible.
        """
        self.coord = None
        self.dalton = None
        self._num_data = None
        self._data_dim = None
        
    def init_point_cloud(self, pos):
        assert type(pos) == np.ndarray, 'pos must be a numpy ndarray'
        _, dim2 = pos.shape
        self.coord = pos[:, 0:3]
        if dim2 > 3:
            self.dalton = pos[:, 3]
        self._num_data, self._data_dim = self.coord.shape
        self.gen_kdtree()
 
    def get_number_of_data(self):
        """
        Return total number of data points .
        """
        return self._num_data

    def get_data_dimension(self):
        """
        Return the dimension of data points.
        """
        return self._data_dim

    def get_coord(self, idx):
        """
        Return coordinates of point with index idx.
        """
        return self.coord[idx]

    def gen_kdtree(self, metric='euclidean'):
        """
        Generate KDTree for datapoints. Using SciPy.neighbors.KDTree method.
        Default distance metric is set to euclidean.
        """
        self.kdtree = KDTree(self.coord, metric=metric)

    def query_neighbors(self, idx, minpts):
        """
        Obtain distance and index information for minpts-neighbors of the point with index idx.
        It is bassically the same as sklear.neighbors.kdtree.query, but instead \
        of return a 1-by-n numpy array, the results are reshape to simple 1D numpy \
        array of size minpts.

        idx:
            integer, index for the point of interest in point cloud.
        minpts:
            integer, the number of nearest neighbor to return. Note that for our case k=1 return distance and index \
            of itself, since all our query points are in the point cloud.
        Return:
            (dist, ind), each is an 1D numpy array of size minpts.
        """
        dist, ind = self.kdtree.query(self.coord[idx].reshape(-1, 3), k=minpts)
        # note the dist, ind could be multi-dimensional array, with each row correspond to a sample queried. \
        # in this function, only ONE sample is allowed for query, so that return dist[0] to reduce dimension to 1D.
        return dist[0], ind[0]

    def query_radius(self, idx, eps):
        """
        Return distance and index information for neighbors within eps distance to the point with idx .

        idx:
            integer, index for the point of interest in point cloud.
        eps:
            float, the search distance to find nearest neighbors.
        Return:
            (dist, ind), each is an 1D numpy array of size minpts.
        """
        # note the dist, ind could be multi-dimensional array, with each row correspond to a sample queried. \
        # in this function, only ONE sample is allowed for query, so that return dist[0] to reduce dimension to 1D.
        ind, dist = self.kdtree.query_radius(self.coord[idx].reshape(-1, 3), r=eps, return_distance=True)
        return dist[0], ind[0]

    def get_KNN_dist(self, num_bins, k):
        """
        Obtain KNN distribution. This is only a wrapper over query_neighbor method and numpy.histogram function.
        """
        idx = self.get_number_of_data()
        dist, _ = self.query_neighbors(idx, k)
        hist, bin_edge = np.histogram(dist, num_bins) #Note bin_edge is len(hist) + 1
        return hist, bin_edge


#-------------------------------------------------------------------------------
# TEST
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    data_3D = np.array([[1,2,3, 0], [4,5,6, 0], [7,8,9, 0]])
    point_3D = PointCloud()
    point_3D.init_point_cloud(data_3D)
    dist, ind = point_3D.query_radius(0, 20)
    print(dist)
    print(ind)