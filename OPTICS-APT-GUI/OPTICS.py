"""
Created on Mon Feb 15 19:43:48 2016

Python implementation of OPTICS for cluster analysis.

@author: Jing Wang
"""

import numpy as np
import heapdict


__version__ = '0.2'

def optics(X, eps, minpts):
    """
    Cluster analysis using OPTICS algorithm.
    
    X:
        a PointCloud class that represent pos data structure
    eps:
        epsilon is the largest search distance in searching for minpts-th neighbors.
    minpts:
        minpts-th neighbor (contains itself) for defining core distance.
        minpts is in range [1, inf).
        where minpts=1 means the center ion itself
              minpts=2 means the 1st neighbor of center ion
              minpts=3 means the 2nd neighbor of center ion
              ...
    
    return:
        ordered_lst, an numpy array of ordered index of all ions.
        reachability_distance, a numpy array of RD for corresponding data in X. 
                                (order should be preserved, such as RD at idx=0 is for the first point in X)
        core_distance, a numpy array of RD for corresponding data in X. 
                                (order should be preserved, such as RD at idx=0 is for the first point in X)
    """
    #################################-------------------------##################
    # A small helper functions Here!!!!!!!!!!!!!!!!
    #################################-------------------------##################
    def set_core_distance(X, idx, eps, minpts):
        """
        Set the core distance of point with index idx with respect to eps, minpts.

        Note: the minpts has to be at least 2 to be meaningful in this implementation \
        (cause minpts=1 returns the idx point itself). If minpts is 1 or there is no other \
        points within eps, the core distance will be remained as Nan as default.
        """
        nonlocal core_distance

        dist, _ = X.query_neighbors(idx, minpts)
        if dist[-1] < eps and len(dist) > 1:
            core_distance[idx]=dist[-1]
 

    ###################################--------------------####################
    # A more important helper function
    ####################################-------------------####################

    def update(X, current_idx, seeds, eps, minpts):
        """
        Helper function to optics.
        """
        nonlocal core_distance
        nonlocal reachability_distance
        nonlocal processed
        
        neighbor_dist, neighbor_idx = X.query_radius(current_idx, eps)
        core_dist = core_distance[current_idx]
        for item in range(len(neighbor_idx)):
            N_idx = neighbor_idx[item]
            if not processed[N_idx]:
                new_reach_dist = np.maximum(core_dist, neighbor_dist[item]) # np.nan > any number
                if np.isnan(reachability_distance[N_idx]):
                    reachability_distance[N_idx] = new_reach_dist
                    seeds[N_idx] = new_reach_dist
                else:
                    if new_reach_dist < reachability_distance[N_idx]:
                        reachability_distance[N_idx] = new_reach_dist
                        seeds[N_idx] = new_reach_dist
        return

    ###################################--------------------####################
    # Helper function ends here
    ####################################-------------------####################

    num_data = X.get_number_of_data()

    processed = np.zeros(num_data, dtype=bool)

    reachability_distance = np.empty(num_data)
    reachability_distance.fill(np.nan)

    core_distance = np.empty(num_data)
    core_distance.fill(np.nan)

    ordered_lst = np.empty(num_data, dtype=int)
    ordered_lst.fill(np.nan)

    count = 0
    for idx in range(num_data):
        if not processed[idx]:
            processed[idx] = True
            set_core_distance(X, idx, eps, minpts)
            ordered_lst[count] = idx

            # Track progress
            count += 1
            percentage = count/num_data
            if np.mod(percentage, 10) == 0:
                print('OPTICS progress:', '{:.0%}'.format(percentage))

            if not np.isnan(core_distance[idx]):
                seeds = heapdict.heapdict()
                update(X, idx, seeds, eps, minpts)
                while len(seeds) != 0:
                    current_idx = seeds.popitem()[0]
                    # if current_idx == 25: # the priority queue heapdict does not preserve the order of elements somehow, 
                    #                       # for example, in the old implementation 9 is returned while in the new one
                    #                       # 25 is returned, even though nothing related with heapdict changed. This may
                    #                       # cause some randomness in the output but should not affect the result. Though
                    #                       # could be troublesome for unit testing.
                    #     print('25 RD is', reachability_distance[25])
                    #     print('9 RD is', reachability_distance[9])
                    processed[current_idx] = True
                    set_core_distance(X, current_idx, eps, minpts)
                    ordered_lst[count] = current_idx

                    # track progress
                    count += 1
                    percentage = count/num_data*100
                    if np.mod(percentage, 10) == 0:
                        print('OPTICS progress:', '{}%'.format(percentage))
                    
                    if not np.isnan(core_distance[current_idx]):
                        update(X, current_idx, seeds, eps, minpts)

    return ordered_lst, reachability_distance, core_distance


######################--------------------------------#########################
if __name__ == '__main__':
    import APT_IOs as aptios
    from APTPosData import APTPosData
    import time
    
    pos_name = 'test_simple_syn_data_88.pos'
    # rng_name = 'test_simple_syn_data_88.rrng'
    # ion_type = np.array(['all'])
    test_pos = aptios.read_pos(pos_name)
    # _, test_rng = aptios.read_rrng(rng_name)
    # selected_rng = aptios.select_rng(test_rng, ion_type)
    # test_ranged_pos = aptios.filter_by_rng(test_pos, selected_rng)

    pos = APTPosData()
    
    time_init = time.time()

    eps=5
    minpts=10
    ordered_list, RD, CD = optics(pos, eps, minpts)

    # np.savetxt('new_ordered_list.txt', ordered_list, fmt='%d')
    # np.savetxt('new_RD.txt', RD, fmt='%5.3f')
    # np.savetxt('new_CD.txt', CD, fmt='%5.3f')

    time_fin = time.time()
    print('process time is ', str(time_fin-time_init), ' sec ')