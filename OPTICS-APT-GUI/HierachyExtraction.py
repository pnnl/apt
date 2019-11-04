"""
To extract clustering information from a reachability plot produced by
OPTICS.

Based on "Automatic Extraction of Clusters from Hierarchical
Clustering Representations "

Author: Jing Wang
"""

from heapdict import heapdict
from BinHeap import BinHeap
from queue import Queue
import numpy as np
import TreeNode
import RDHist

def _find_local_maxima(ordered_RD, k=1):
    """
    Find local maxima to the k neighbor of left and k neighbors of right.\
    It equivalent to a moving windows of size 2*k+1. RD is an 1D numpy array.\

    Since we are looking for max, a max heap would be most beneficial.\
    For the sake of simplicity, we just take the negtive of RD, and use \
    a min heap already implemented a while ago.    
    """
    local_maxima = heapdict()
    moving_window = BinHeap()
    window_size = 2*k+1
    poi_idx = -k # index for point of interest start from -k from convinience.
    RD_size = len(ordered_RD)

    assert window_size < RD_size, 'Invalid k or RD, window size can not be smaller than RD size'

    while poi_idx < RD_size:
        w_s_idx = poi_idx - k # window start idx
        w_e_idx = poi_idx + k # window end idx
        if w_e_idx < RD_size:
            moving_window.insert((-ordered_RD[w_e_idx], w_e_idx)) # negtive is to accomadate use of min heap, since RD is always non-negtive.

        if (moving_window.heap_size > window_size) or (w_e_idx >= RD_size): 
            heap_el_idx = moving_window.pos[w_s_idx-1]
            moving_window.extract_element(heap_el_idx)  # delete elements outside of the window.

        if poi_idx >= 0:
            current_RD = -ordered_RD[poi_idx]
            if current_RD <= moving_window.heap[0][0]:
                local_maxima[poi_idx] = current_RD # a negative RD is also used for local maxima since heapdict is a min priority queue
        poi_idx +=1

    return local_maxima

def _determine_split(CurrentNode, ordered_RD, significance_of_separation, min_node_size):
    """
    Helper function, to determine if curent split point is significant.
        RD is an 1D numpy array of reachability distances
        CurrentNode is a node object for current processing
        significance_of_separation is an user-defined value (0.0<sig<1.0) to determine if a split is significant or not.
        min_cluster_size is the smallest allowed cluster.
    """

    ################---------------------------------------------------------------------------------------------------------
    #   This algorithm is still under experiment. It is an extended and modified version of original version,
    #   which was reported by Sander et al. Automatic extraction of clusters from hierarchical clustering representations
    #
    #   Currently, it works as follwoing:
    #       1. all data with RD lower than local maxima points are selected,
    #       2. a histogram of those points are made;
    #       3. a Gaussian Mixture Model of 2 components are used to fit this algorithm
    #       4. the component with smaller mean value is then used to compare with local maxima
    ###############------------------------------------------------------------------------------------------
    def assign_node_quality(start, end):
        """
        Assign left or right node quality.
        """
        nonlocal split_point_RD
        nonlocal ordered_RD
        nonlocal min_node_size

        temp_RD = ordered_RD[start:end]
        temp_RD = temp_RD[temp_RD <= split_point_RD] # filter out points with RD larger than split point RD.
        if len(temp_RD) >= min_node_size:
            _, means, _ = RDHist.GMM_fitting_stats(temp_RD.reshape(-1,1))
            ave_RD = min(means)
            node_size = end - start
        else:
            ave_RD = 0.0
            node_size = 0

        return ave_RD, node_size

    start, end = CurrentNode.index_range
    split_point, split_point_RD = CurrentNode.next_lm()
    split_point_RD = -split_point_RD #the RD in local_maxima_points is negtive to cope with min priority queue

    #assign left node quality
    left_ave_RD, left_node_size = assign_node_quality(start, split_point)

    #assign right node quality
    right_ave_RD, right_node_size = assign_node_quality(split_point, end)

    left_node_eligible = False #initialize left node eligible
    right_node_eligible = False #initialize right node eligible

    is_split_significant = False
    #---------------start testing if result split is significant----------------
    # print('left ave RD is', left_ave_RD)
    # print('right ave RD is', right_ave_RD)
    # print('split point RD is', split_point_RD)
    if left_ave_RD / split_point_RD < significance_of_separation and right_ave_RD / split_point_RD < significance_of_separation:
        #if split_point is not significant, ignore split_point and continue,
        if left_node_size >= min_node_size:
            left_node_eligible = True #left node does not pass test, set to False\
        if right_node_size >= min_node_size:
            right_node_eligible = True #right node does not pass test, set to False

        if left_node_eligible or right_node_eligible:
            CurrentNode.split_point = split_point
            is_split_significant = True
        
    return is_split_significant, left_node_eligible, right_node_eligible


def _cluster_tree(to_be_process, ordered_RD,  significance_of_separation, min_node_size, similarity_level):
    """
    Algorithm for constructing a cluster tree from a reachability plot. \
    THe original algorithm uses recursion, however to improve predictability, \
    it is rewriten to iterative process.

    ordered_RD:
        is an numpy array of reachability distance
    to_be_process:
        is a python queue that stores node to be processed.
    local_maxima_points:
        is a heapdict of local maxima points with keys of reachability.
    significance_of_separation:
        is the value to determine wether a split point is significant.
    min_cluster_size:
        is the minimum cluster size to be considered as significant
    similarity level:
        Defines the similarity between a node and its parent. It only change hierachical level result but will not likely to affect
        leaf nodes, where we interest the most.
    """
    count = 0
    while not to_be_process.empty():
        
        CurrentNode = to_be_process.get()

        #Find the next eligible significant local maxima
        is_split_significant = False
        while (not is_split_significant) and (not CurrentNode.is_lm_empty()):
            is_split_significant, left_node_eligible, right_node_eligible  = _determine_split(CurrentNode, 
                                                                                              ordered_RD, 
                                                                                              significance_of_separation, 
                                                                                              min_node_size)
            # Track progress
            count += 1
            if np.mod(count, 100) == 0:
                print('Processed local maxima:', count)

        if is_split_significant: # split is significant, split Current node.
            #check if node can be moved up one level.
            #----------!!!!!!!!Note: this place is particularly ambiguous in the publication describe this method!!!!!!!---------
            #Even the context describing it does not actually match with pesudo-code.
            #In this case, implementaion is following pesudo-code.
            #It should not affect our result if leaf nodes are all we care.
            is_similar = False
            split_point_RD = ordered_RD[CurrentNode.split_point]
            if not CurrentNode.is_root():
                if split_point_RD / ordered_RD[CurrentNode.parent.split_point] > similarity_level: # if two node are similar, move new node up to attach ParentNode
                    is_similar = True

            left_child, right_child = TreeNode.divide(CurrentNode, CurrentNode.split_point, is_similar)
            
            if left_node_eligible:
                to_be_process.put(left_child)
            else:
                left_child.parent.remove_child(left_child)

            if right_node_eligible:
                to_be_process.put(right_child)
            else:
                right_child.parent.remove_child(right_child)

    return

def automatic_cluster_extraction(ordered_RD, min_node_size=2, significance_of_separation=0.75, k=1, similarity_level=0.9):
    """
    Automatice cluster extraction from reachability plots. The algorithm is based on hierachical clustering.

    ordered_RD:
        Reachability distance array produced by optics
    min_cluster_size:
        the minimum number of points to be considered as cluster. Note it is different from Nmin in IVAS,\
        since this number actually participate in clustering process. Thus change this number could change \
        clustering result, while in IVAS Nmin merely filtered out smaller clusters.If min_cluster_size = None, \
        the min_cluster_size will be set to 0.5% of all points. (Note it could be a large number if total points are larget)
    similarity_level:
        Defines the similarity between a node and its parent. It only change hierachical level result\
        but does not affect
        leaf nodes, where we interest the most.
    significance_of_separation:
        Define a level above which local maxima will be considered as significant and start spliting proecess. The original research
        paper suggest 0.7-0.8 generally should work well.
    k:
        Find local maxima to the k neighbor of left and k neighbors of right.\
    It equivalent to a moving windows of size 2*k+1. RD is an 1D numpy array
    """

    local_maxima_points = _find_local_maxima(ordered_RD, k)

    RootNode = TreeNode.TreeNode(0, len(ordered_RD), local_maxima_points)
    to_be_process = Queue()
    to_be_process.put(RootNode)
    _cluster_tree(to_be_process, ordered_RD, significance_of_separation, min_node_size, similarity_level)

    return RootNode

def _horizontal_cut(node, DB_equ_eps, ordered_RD, ordered_CD):
    """
    Extract cluster information in a DBSCAN-like method.

    ordered_RD:
        the reachability distance correspond to each point in ordered list produced by OPTICS.
    node:
        a node class that represents cluster. ideally, this is leaf node.
    DB_equ_eps:
        DBSCAN equivalent eps (also called dmax in IVAS)
    """
    ##-------------------------------------------------------------------------
    # Note that this is a modified version of the DBSCAN-like method described in Sander's paper.
    # If we alwasy start from the index 0 of current RD, then Sander's method works perfectly.
    # 
    # However, current method will be used to refining nodes, which could be start from any between 0 to len(RD).
    # Thus a modification is requied. 
    # 
    # The issue lie in the index 'start'. In the original algorithm in Sander's paper, since RD[0] is always nan
    # therefore condition '(RD > eps or RD is nan) and (CD < eps)' always works to initiate a new cluster.
    # In our case, the node in the middle would have a value RD, which may or may not larger than eps. 
    # This caused the algorithm unable to initialize a new cluster until it meet a new RD > eps in current node,
    # even if the CD < eps is met from the beginning. This is problematic, since in the original way, CD < eps will
    # always be included in a cluster, while in our case is descarded. Note this issue is only for the first index
    # element. Changing the new cluster initialization at the first index would solve this issue.
    # 
    # The new condition for  a new cluster: 
    #       1. (RD > eps or RD is nan) and (CD < eps)
    #       2. RD <= eps and CD < eps
    ##-------------------------------------------------------------------------
    start, end = node.index_range

    if ordered_CD[start] >= DB_equ_eps:
        is_in_cluster = False
    else:
        is_in_cluster = True
        new_start = start
        new_end = start+1

    # Normal density cut clustering start
    for idx in range(start+1, end):
        if ordered_RD[idx] >= DB_equ_eps or np.isnan(ordered_RD[idx]):
            if is_in_cluster: # if previous point still in cluster, save 'new_start' to 'new_end' as a new cluster 
                new_node = TreeNode.TreeNode(new_start, new_end, heapdict())
                node.add_child(new_node)

            if ordered_CD[idx] < DB_equ_eps: # initiate a new cluster
                is_in_cluster = True
                new_start = idx
                new_end = new_start+1
            else: # RD > eps and CD > eps, does not belong to any cluster.
                is_in_cluster = False

        elif is_in_cluster: # if RD < eps and RD is not nan, and in cluster track
            if idx < end - 1:
                new_end += 1
            elif idx == end - 1:
                new_node = TreeNode.TreeNode(new_start, idx, heapdict())
                node.add_child(new_node)
    return


def cluster_refine(node, ordered_RD, ordered_CD, cluster_member_prob):
    """
    An algorithm to refine clusters (noise detection and nested cluster detection)
    based on GMM fitting.

    Note this function does not guarentee refined clusters satisfy min_cluster_size
    condition. So a post-filtering must be performed when output or visualize data.

    The histogram of RD plot is fitted with Gaussian mixture model with 2 components.
    The one with smaller mean will be considered as 'clusters'. For each observation,
    there is a probability that it belong to one of the two components.
    Observations with a probability higher than 'cluster_member_prob' will be
    considered as part of a cluster.

    ordered_RD:
        the reachability distance correspond to each point in ordered list produced by OPTICS.
    node:
        a node class that represents cluster. ideally, this is leaf node.
    cluster_member_prob:
        Observations with a probability higher than 'cluster_member_prob'
        will be considered as part of a cluster. It should be within 0.0-1.0.
    """
    def dist_refine(node, ordered_RD, ordered_CD, cluster_member_prob, min_size=10):
        """
        Helper function. Cluster refinement based on distribution.
        """
        #Ideally mixed generalized gamma distribution should
        #be used but due to difficulties of implementation, a Gaussian mixture model
        #of component 2 is used.
        if node.size > min_size: # the number need to be re-considered, how much info do we need to do a proper GMM estimation?
            start, end = node.index_range
            temp_RD = ordered_RD[start:end]

            est_eps = RDHist.GMM_fitting_posterior_proba(temp_RD.reshape(-1,1), cluster_member_prob)

            if est_eps != 0:
                _horizontal_cut(node, est_eps, ordered_RD, ordered_CD)
            # print('for', str(start),'to', str(end), ',estimated eps is:', est_eps)

        return

    start, end = node.index_range

    left_split = start
    right_split = end - 1 if end == len(ordered_RD) else end

    _horizontal_cut(node, min(ordered_RD[left_split], ordered_RD[right_split]), ordered_RD, ordered_CD)

    if not node.is_leaf():
        for child in node.children:
            dist_refine(child, ordered_RD, ordered_CD, cluster_member_prob)
    else:
        dist_refine(node, ordered_RD, ordered_CD, cluster_member_prob)

    return

#### For Tests!#######################
if __name__ == '__main__':
    # test _find_local_maxima
    RD = np.random.random(10)
    lm = _find_local_maxima(RD, k=1)
    print('RD is: ', RD)
    for key, val in lm.items():
        print('index is', key, '; local maxima is, ', val)

    # Test _determine_split
    CurrentNode = TreeNode.TreeNode(0, len(RD), lm)
    significance_of_separation = 0.75
    min_cluster_size = 2
    is_split_significant, left_node_eligible, right_node_eligible = \
        _determine_split(CurrentNode, RD, significance_of_separation, min_cluster_size)
    print('is significant?', is_split_significant)
    print('split point is', CurrentNode.split_point, ', and associated RD is', RD[CurrentNode.split_point])
    print('left and righ eligibility are', left_node_eligible, right_node_eligible)

    # Test _cluster_tree and Test automatic_cluster_extraction
    to_be_process = Queue()
    to_be_process.put(CurrentNode)
    similarity_level = 0.9
    _cluster_tree(to_be_process, RD, significance_of_separation, min_cluster_size, similarity_level)
    tree = TreeNode.extract_tree(CurrentNode, RD)
    print('Nodes in the tree are:', tree)

    # Test automatic_cluster_extraction
    Root = automatic_cluster_extraction(RD, min_cluster_size, significance_of_separation, k=1)
    tree = TreeNode.extract_tree(CurrentNode, RD)
    print('Nodes in the tree are:', tree)
    
    # test _horizontal_cut
    NewNode = TreeNode.TreeNode(0, len(RD), lm)
    CD = RD - 0.1
    DB_equ_eps = 0.5
    _horizontal_cut(NewNode, DB_equ_eps, RD, CD)
    print('RD is', RD)
    print('CD is', CD)
    new_tree = TreeNode.extract_tree(NewNode, RD)
    print('tree after (NEW) horizontal cut is', new_tree)

    # NewNode = TreeNode.TreeNode(0, len(RD), lm)
    # CD = RD - 0.1
    # DB_equ_eps = 0.5
    # _horizontal_cut_old(RD, CD, NewNode, DB_equ_eps, np.inf)
    # new_tree = TreeNode.extract_tree(NewNode, RD)
    # print('tree after (OLD) horizontal cut is', new_tree)

    # test cluster_refine
    NewNode = TreeNode.TreeNode(0, len(RD), lm)
    CD = RD - 0.1
    cluster_member_prob = 0.5
    cluster_refine(NewNode, cluster_member_prob, RD, CD)
    print('RD is', RD)
    print('CD is', CD)
    new_tree = TreeNode.extract_tree(NewNode, RD)
    print('tree after cluster refine is', new_tree)
