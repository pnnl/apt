# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:48:27 2016

Main program for clustering APT datasets

@author: Jing Wang
"""

import APT_IOs as aptios
from PointCloud import PointCloud
import ColorTable

import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from itertools import cycle
from collections import Counter

__version__ = '0.2'

def _calc_Rg(data):
    """
    Calculate radius of gyration.
    NOTE: Assuming all ions are equal in mass (homogenous).

    data: an numpy array of dimension N-by-3

    return:
        Guinier radius/radius of gyration (assuming equal mass)
        and center of mass
    """
    r_com = np.average(data, axis=0)

    Rg_square = np.sum(np.power(np.subtract(data, r_com), 2))

    return np.sqrt(np.divide(Rg_square, data.shape[0])), r_com


class Clustering:
    """
    A class for doing clustering on APT point cloud data.
    """
    __version__ = '0.2'

    def __init__(self, APTPosData):
        self.APTPosData = APTPosData
        self.current_PointCloud = None
        self.current_PointCloud_ion_identity = None
        self.cluster_id = np.array([])

        self._paras = dict()


    def DBSCAN_clustering(self, eps, order):
        """
        Original DBSCAN algorithm from sklearn.cluster.DBSCAN()
        """
        self._paras['DBSCAN eps'] = eps
        self._paras['DBSCAN order'] = order

        self.cluster_id = DBSCAN(eps, order).fit_predict(self.current_PointCloud.coord)

    def ion_selection(self, selected_ion_types):
        """
        Select ion types in the rng file for clustering. 

        select_ion_types:
            a list/array of ion types to be selected. \
            To select specific ions, use format ['A1B1', 'C2', ...];
            To select all ions, regardless of ranged or not ranged, use 'all';
            To select all ranged ions, use 'Ranged';
            To select all unranged ions, use 'Noise';

        """
        self._paras['selected ions for clustering'] = selected_ion_types
        selection = self.APTPosData.select_ions(selected_ion_types)

        selected_pos = self.APTPosData.pos[selection]
        self.current_PointCloud_ion_identity = self.APTPosData.identity[selection]

        self.current_PointCloud = PointCloud()
        self.current_PointCloud.init_point_cloud(selected_pos)
        self.cluster_id = -np.ones(len(selected_pos), dtype=int)
    

    def update_cluster_id(self, idx_lst, cluster_id):
        """
        Update cluster_id for point with index in idx_lst.
        """
        self.cluster_id[idx_lst] = cluster_id


    def write_indexed_cluster_rrng(self, output_fname):
        """
        Write a range file for indexed clusters like IVAS did.

        The min cluster id is increased to 1 due to IVAS enforcement on Range must start from 1 and end at size Num.
        """
        cluster_ids = np.unique(self.cluster_id)
        min_id = np.min(cluster_ids)
        if min_id < 0: # shift min cluster id to 0, since IVAS does not display m/z < 0
            cluster_ids -= min_id - 1
        num_clusters = len(cluster_ids)
        delta = 0.1

        indexed_rrng_part1 = ['[Ions]', 'Number={}'.format(num_clusters)]
        indexed_rrng_part2 = ['[Ranges]', 'Number={}'.format(num_clusters)]

        color_table = ColorTable.gen_color_table(style='hex')
        cy_color = cycle(color_table)

        for c_id in cluster_ids: # original cluster id is from -1 to num so that there are num+2 clusters, including matrix.
            indexed_rrng_part1.append('Ion{}=Cluster{}'.format(c_id, c_id))
            hex_rgb = next(cy_color)
            indexed_rrng_part2.append('Range{}={} {} Vol:0.00000 Name:Cluster{} Color:{}'.format(c_id, c_id-delta, c_id+delta, c_id, hex_rgb ))

        indexed_rrng = indexed_rrng_part1 + indexed_rrng_part2
        with open(output_fname+'.indexed.rrng', mode='w') as f:
            for line in indexed_rrng:
                f.write(line+'\n')
                

    def write_cluster_to_file(self, output_fname, file_format = 'all'):
        """
        Write clustering result to file (txt or pos).

        output_name:
            output filename
        file_format:
            output file format (.txt and .pos). default is both.
            File formating as:  [[x,y,z, m/z, cluster_id], ...]
        """
        txt_flag = True if file_format == 'all' or file_format =='txt' else False
        pos_flag = True if file_format == 'all' or file_format =='pos' else False

        output_cluster_id = np.copy(self.cluster_id)
        min_id = np.min(self.cluster_id)
        if min_id < 0: # shift min cluster id to 0, since IVAS does not display m/z < 0
            output_cluster_id -= min_id - 1
        
        cluster_pos = np.concatenate((self.current_PointCloud.coord, output_cluster_id.reshape(-1, 1)), axis=1)
        cluster_pos = cluster_pos.astype('>f4')
        if txt_flag == True:
            #write to txt file
            header = 'X Y Z Cluster_id'
            aptios.write_txt(output_fname+'.txt', cluster_pos, header)

        if pos_flag == True:
            # write to pos file
            aptios.write_pos(output_fname+'.pos', cluster_pos)


    def cluster_stats(self, output_fname=None):
        """
        Calculate statistics of clusters.

        output_fname:
            filename to write the summary stats. If is None, no output is generated.

        Return:
            cluster size statistice summary in the following format:
            [[cluster_id, com_x, com_y, com_z, Rg, RS, Ion1_count, Ion2_count, ...]
              ...]
        """
        unique_ids = np.unique(self.cluster_id)
        unique_ion_types = np.unique(self.current_PointCloud_ion_identity)
        points = self.current_PointCloud.coord

        summary = np.zeros((len(unique_ids), len(unique_ion_types)+7)) #initiate an arry to store cluster statistics
        
        for idx, c_id in enumerate(unique_ids):
            logic = self.cluster_id == c_id
            c_points = points[logic]
            num_ions = len(c_points)

            comps = dict(Counter(self.current_PointCloud_ion_identity[logic]))

            radius_of_gyration, com = _calc_Rg(c_points)
            radius_of_sphere = np.sqrt(5.0/3.0) * radius_of_gyration

            summary[idx, 0:7] = [c_id, com[0], com[1], com[2], radius_of_gyration, radius_of_sphere, num_ions]

            ion_idx = 7
            for ion_type in unique_ion_types:
                try:
                    summary[idx, ion_idx] = comps[ion_type]
                except KeyError:
                    pass
                ion_idx += 1

        if output_fname != None:
            ions = ''
            for ion in unique_ion_types:
                ions += ' ' + ion

            header = 'Cluster_id com_x com_y com_z Rg R(sphere) solute_tot' + ions
            aptios.write_txt(output_fname+'.txt', summary, header=header)

        return summary

    def calc_ARI(self, labels_true):
        """
        Adjusted Rand measure (or adjusted Rand index). This is a function to calculate the metric of the similarity between clustering result with
        benchmark labeld data (ground truth).

        W. M. Rand (1971). "Objective criteria for the evaluation of clustering methods". Journal of the American Statistical Association. American Statistical Association. 66 (336): 846–850. JSTOR 2284239. doi:10.2307/2284239.
        Lawrence Hubert and Phipps Arabie (1985). "Comparing partitions". Journal of Classification. 2 (1): 193–218. doi:10.1007/BF01908075.
        Nguyen Xuan Vinh, Julien Epps and James Bailey (2009). "Information Theoretic Measures for Clustering Comparison: Is a Correction for Chance Necessary?" (PDF). ICML '09: Proceedings of the 26th Annual International Conference on Machine Learning. ACM. pp. 1073–1080.PDF.
        """
        labels_pred = self.cluster_id

        ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
        print("Adjusted Rand index is: ", ARI)

        return ARI


    def write_log(self, output_name):
        """
        Write parameters for optics and cluster extraction to a file
        """

        with open(output_name+'.txt', 'w') as logfile:
            for k, v in self._paras.items():
                logfile.write(str(k) + ' : ' + str(v) + '\n')


    def dump_object(self, obj_name):
        """
        Dump the Clustering Class as binary file for future re-use.
        """
        pickle.dump(self, open(obj_name+'.cl', 'wb'))


# Test
if __name__ == '__main__':
    import time
    init_time = time.time()

    point_cloud = PointCloud()
    test = Clustering(point_cloud)
    test.cluster_id = [-1, 0,0,1,2]
    test.write_indexed_cluster_rrng('test_output_file.txt')    

    fin_time = time.time()
    print('total computation time is: ', fin_time-init_time)
