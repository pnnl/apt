"""
A class for doing OPTICS-APT cluster analysis on APT dataset
"""


from OPTICS import optics
import HierachyExtraction as hierarchy
import TreeNode
from Clustering import Clustering

import numpy as np
from heapdict import heapdict


class OPTICSAPT(Clustering):
    def __init__(self, APTPosData):
        Clustering.__init__(self, APTPosData)

        self.ordered_lst = None # ordered list is a list of ordered index of corresponding points
        self.RD = None # reachability distance per point, need ordered lst to put in right clustering order
        self.CD = None # core distance per point, like RD, also need ordered lst to put in right clustering order
        self.hierarchy_RootNode = None
    
    def do_optics(self, eps, minpts):
        """
        Generate reachability-distance by OPTICS algorithm
        """
        self._paras['OPTICS eps'] = eps
        self._paras['OPTICS minpts'] = minpts

        self.ordered_lst, self.RD, self.CD = optics(self.current_PointCloud, eps, minpts)

        # convert NaN to max eps, this is to comply with distribution-based eps estimation, 
        # i.e. the average RD calculation for determine significance of separation, and the
        # cluster refinement, both of which need GMM to fit RD. A np.nan would result unpredictable
        # behavior.
        ind_isnan = np.where(np.isnan(self.RD))
        self.RD[ind_isnan] = np.nanmax(self.RD)

        ind_isnan = np.where(np.isnan(self.CD))
        self.CD[ind_isnan] = np.nanmax(self.CD)


    def hierarchy_clustering(self, 
                             min_node_size=2, 
                             significance_of_separation=0.75, 
                             k=None,
                             similarity_level=0.9,
                             cluster_member_prob=0.5):
        """
        Hierarchy clustering of data based on OPTICS algorithm returned resule. \
        In addition to the automatic cluster extraction described in Sander's paper, \
        we added a RD histogram-based cluster refinement process to reduce interference \
        from noise RD in leaf nodes.

        min_node_size:
            a min size of node allowed when considering split nodes
        significance_of_separation:
            level of significance when consdiering splitting nodes
        k:
            when calculate local maxima, how many neighbors it needs to be included.
        similarity_level:
            level of similarity when judging current node and its new child are similar\
            enough, so the new child will replace current node.
        """
        print('Hierarchical clustering start')

        if k == None:
            k = self._paras['OPTICS minpts']

        self._paras['hierarchy min node size'] = min_node_size
        self._paras['hierarchy significance of separation'] = significance_of_separation
        self._paras['hierarchy k (for find local maxima)'] = k
        self._paras['hierarchy node similarity criteria'] = similarity_level
        self._paras['hierarchy cluster refinement criteria'] = cluster_member_prob

        ordered_RD = self.RD[self.ordered_lst]
        ordered_CD = self.CD[self.ordered_lst]

        hierarchy_RootNode = hierarchy.automatic_cluster_extraction(ordered_RD,
                                                                    min_node_size = min_node_size,
                                                                    significance_of_separation = significance_of_separation,
                                                                    k = k, 
                                                                    similarity_level = similarity_level)

        # cluster refinement
        hierarchy_leaves = []
        hierarchy_leaves = TreeNode.retrieve_leaf_nodes(hierarchy_RootNode, hierarchy_leaves)
        
        # tracking progress
        count = 0
        num_leaves = len(hierarchy_leaves)
        print('leaves are:', num_leaves)
        for node in hierarchy_leaves:
            count += 1
            percentage = count/num_leaves
            if num_leaves > 10:
                if np.mod(count, np.round(num_leaves/10)) == 0:
                    print('Refine Hierarchy Tree progress:', '{:.0%}'.format(percentage))
            else:
                print('Refine Hierarchy Tree progress:', '{:.0%}'.format(percentage))
            hierarchy.cluster_refine(node, ordered_RD, ordered_CD, cluster_member_prob)
           

        print('Hierarchical clustering finished')
        self.hierarchy_RootNode = hierarchy_RootNode
        return hierarchy_RootNode


    def create_clusters(self, est_bg=None, min_cluster_size=2):
        """
        create clusters from the hierarchy node tree by update cluster_id accordingly.

        est_bg:
            The estimated RD cutoff. Set cluster id to noise if average RD is above the cutoff.
            Default is np.inf
        min_cluster_size:
            The min size of valid cluster size. All clusters with smaller size will be set to noise.
            Default is 2.

        """
        if est_bg == None:
            est_bg = np.inf

        cluster_id = 0 # Note default cluster id is -1, which correspond to noise
        hierarchy_leaves = []
        hierarchy_leaves = TreeNode.retrieve_leaf_nodes(self.hierarchy_RootNode, hierarchy_leaves)
        
        ordered_RD = self.RD[self.ordered_lst]

        for leaf in hierarchy_leaves:
            start, end = leaf.index_range
            if leaf.average_RD(ordered_RD) < est_bg and leaf.size >= min_cluster_size:
                self.update_cluster_id(self.ordered_lst[start:end], cluster_id)
                cluster_id += 1

        return


    def write_RD_to_file(self, output_fname):
        """
        Write ordered_lst, RD, CD, to file
        """
        temp = np.empty((len(self.RD), 3))
        temp[:, 0] = self.ordered_lst
        temp[:, 1] = self.RD
        temp[:, 2] = self.CD
        np.savetxt(output_fname+'.txt', temp, fmt=['%i', '%f', '%f'], header='ordered_idx RD CD')


    def single_density_thresholding(self, sd_eps):
        """
        Using a single density threshold for clustering based on RD plot from \
        optics. This clustering is supporsed to produce DBSCAN-like result. \
        Each potential cluster will be a child of the root node. The root node\
        correspond to the entire dataset.

        sd_eps:
            single density epsilon
        
        return:
            The root node. It may or maynot have any children, depend on the clustering result.
            if the density threshold is not unreasonably high or low, new clusters will be\
            identified and added as the child to the root node.
        """
        self._paras['OPTICS single density clustering eps'] = sd_eps
        lm = heapdict()
        sd_root = TreeNode.TreeNode(0, len(self.RD), lm)
        hierarchy._horizontal_cut(sd_root, sd_eps, self.RD[self.ordered_lst], self.CD[self.ordered_lst])

        return sd_root
