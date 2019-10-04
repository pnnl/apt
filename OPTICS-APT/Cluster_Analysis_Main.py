"""
Main program entrance to run cluster analysis.

Author: Jing Wang
"""
from APTPosData import APTPosData
from OPTICS_APT import OPTICSAPT
import Visualization as visual

import time
import pickle


#################------------Define Parameters Start-------------##############


###############################################################################
# define data and range files
#---------------------------------Start----------------------------------------
## an example for test data
# data_filename = 'R31_06458-v03.pos'
# rng_filename = 'MA957_Jing_v4_voltage.RRNG'
# ion_type = ['Ti1O1', 'Y1O1', 'Y1']

data_filename = 'test_simple_syn.pos'
rng_filename = 'test_simple_syn.rrng'


# data_filename = 'R31_06365-v02.pos'
# rng_filename = 'R31_06365-v02.rrng'


#----------------------------------End-----------------------------------------
###############################################################################




###############################################################################
# Clustering analysis:
#       Ion selection,
#       eps, minpts, for OPTICS or DBSCAN
#       method, OPTICS-APT or DBSCAN
#---------------------------------Start----------------------------------------
    # ion_type format e.g. ion_type = ['A1', 'A1B1', ...] or ['all'] (All ions) or ['Ranged'] (Only ranged ions) or ['Noise'] (Only un-ranged ions)
# ion_types = ['Ti1O1', 'Y1', 'Y1O1'] 
ion_types = ['all']

eps = 5.0
minpts = 20
method = 'OPTICS-APT' # what method to extract clusters, choice are 'in_house' or 'DBSCAN'

min_cluster_size = minpts
#----------------------------------End-----------------------------------------
###############################################################################




###############################################################################
# Materials property
#---------------------------------Start----------------------------------------
rho = 88.48
det_eff = 0.37
con = 0.14
# Note!!!!!!!! The background function is yet to be implemented.
# CSR_stats = Clustering.est_background_knn_d(rho, det_eff, con, k) 
#----------------------------------End-----------------------------------------
###############################################################################





###############################################################################
# Expert options, usually no need to change.
#---------------------------------Start----------------------------------------
# density_thresh = CSR_stats[0]-2*CSR_stats[1] # or it could be a user-defined value, should be in several nm.
k = minpts
est_bg = None
min_node_size = minpts
significant_separation = 0.75 # an user defined value, between 0.0 to 1.0,
                                   # that defines how different the average
                                   # RD of potential cluster should be from
                                   # a local maxima or 'split point'. Based on tests
                                   # we believe a value between 0.7-0.9 generally
                                   # produce satisfying result.
cluster_member_prob = 0.5 #basically defined how high the probability
                                     #should be for a data point to be considered
                                     #as part of the cluster, based on GMM components.
                                     #It should be a value between 0.0 and 1.0.
node_similarity = 0.9 # controls whether to merge child node with parent, should not affect
                      # final cluster creation from leaf. It will affect hierarchies.
#----------------------------------End-----------------------------------------
###############################################################################





###############################################################################
# defines output and visualization parameters
#---------------------------------Start----------------------------------------
output_filename = 'test_optics' # base name for output files, no extension.
save_file = False
show_visualization = True# flag to turn on/off visualization.
                          # Note that enables it would require external libraries.

show_background = False # flag to control on/off of background/matrix/noise ions.
                   # Do not suggest to change to on unless sure that matrix would
                   # no block views of clusters

#----------------------------------End-----------------------------------------
###############################################################################


###########---------------Define Paremeters END----############################


###############################################################################
# Processing (no need to change anything normally)
#---------------------------------Start----------------------------------------
init_time = time.time()

PosData = APTPosData()

PosData.load_data(data_filename, rng_filename)

Clusterer = OPTICSAPT(PosData)
Clusterer.ion_selection(ion_types)

if method == 'DBSCAN':
    Clusterer.DBSCAN_clustering(eps, minpts)
elif method == 'OPTICS-APT':
    Clusterer.do_optics(eps, minpts)
    Clusterer.hierarchy_clustering(min_node_size=min_node_size, 
                                    significance_of_separation=significant_separation,
                                    k=k,
                                    similarity_level=node_similarity,
                                    cluster_member_prob=cluster_member_prob)
    Clusterer.create_clusters(est_bg=est_bg, min_cluster_size=min_cluster_size)

    if save_file:
        Clusterer.write_RD_to_file(output_filename+'_ol_RD_CD')

if save_file:
    Clusterer.cluster_stats(output_filename+'_cluster_stats')
    Clusterer.write_cluster_to_file(output_filename+'_clusters')
    Clusterer.write_indexed_cluster_rrng(output_filename+'_clusters')
    Clusterer.write_log(output_filename+'_log')

fin_time = time.time()
print('total computation time is: ', fin_time-init_time)

if show_visualization:
    visual.visualization(Clusterer, background=show_background)
    # visual.visualize_node_tree(Clusterer.hierarchy_RootNode, Clusterer.RD[Clusterer.ordered_lst])

# Clusterer.dump_object(output_filename)
# with open(output_filename+'_RootNode.cl', 'wb') as o_f:
#     pickle.dump(Clusterer.hierarchy_RootNode, o_f)

#----------------------------------End-----------------------------------------
###############################################################################