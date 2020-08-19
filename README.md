## What is this OPTICS-APT?

This package perform a cluster analysis procedure [1] develped based on the OPTICS algorithm [2] for Atom Probe Tomography [3] Data.

[1]: Jing Wang, Daniel K. Schreiber, Nathan Bailey, Peter Hosemann, Mychailo B. Toloczko, The Application of the OPTICS Algorithm to Cluster Analysis in Atom Probe Tomography Data, Microscopy and Microanalysis, 25(2), 338-348, 2019  
[2]: Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, Jorg Sander, OPTICS: Ordering Points To Identify the Clustering Structure, Proceedings of ACM SIGMOD'99 Int. Conf. on Management of Data, Philadelphia PA, 1999  
[3]: Baptiste Gault, Michael P. Moody, Julie M. Cairney, Simon P. Ringer, Atom Probe Microscopy, New York, NY, Springer Science & Business Media, 2012  

## Instruction for installation:

Using environment: Python 3.7 (recommend using Anaconda package) 
Dependent packages:
*	Numpy (seems any recent version will do (tested in v1.16, already in Anaconda package)
*	Sklearn (seems any recent version will do (tested in , already in Anaconda package)
*	Matplotlib (seems any recent version will do, already in Anaconda package)
*	Heapdict (v1.0.0 installation guide at https://anaconda.org/anaconda/heapdict)
*	PyQtGraph (v0.10.0 installation guide at http://www.pyqtgraph.org/, optional if don’t want immediate visualization)
*	PyOpengl (v3 installation guide at https://anaconda.org/anaconda/pyopengl, optional if don’t want immediate visualization)

## Instruction for user input:

*	Put pos file and range file (RRNG format) in the same folder as these scripts
*	The main file ‘Cluster_Analysis_Main.py’ controls how the scripts run:
	*	First section: 
		*	**data_filename**: filenames for input pos or txt file contains coordinate and m/z of each data point.  
		*	**rng_filename**: range file in the format of RRNG.  
		An example dataset name ‘test_simple_syn.pos’ and range file ‘test_simple_syn.rrng’ are provided.

	*	Second section for clustering analysis: 
		*	**Ion selection**: defines what ion types will be used for cluster analysis. Options are explained in the script.  
		*	The **eps** is the maximum distance for searching minpts-th neighbor. It can be arbitrarily large for small system (< 5000 points) but becomes very inefficiency for large system. It should be slightly larger than largest value of minpts-th nearest neighbor distances in the dataset.  
		*	The **minpts** defines minimum number of neighbors for a point to be considered as core. It also smooths the reachability distance (RD) plot. An optimal minpts must be used to ensure a good quality of RD plot and sensitivity to small clusters. Experimentation with minpts is needed for unknown systems. For irradiated steels, a value between 10-50 is usually satisfactory.  
		*	The **method**: a string, determine which method for clustering, either ‘DBSCAN’ or ‘OPTICS-APT’.  
		*	**min_cluster_size**: min size of final clusters, below which will not be labeled noise.  

	*	Third section: Materials property section, currently not used. Saved for future automatic background estimation.
		*	**Rho**: atomic density (#/nm3) of the matrix material.  
		*	**Det_eff**: detector efficiency.  
		*	**Con**: solute concentration.

	*	Forth section: Expert options for hierarchical clustering algorithm. There is no need to change those unless you understand what’s behind. 
		*	**k**: determine local maximum calculation by comparing with k neighbors to the left and right.  
		*	**est_bg**: estimated background RD. Currently automatic selection not available. Will be implemented in future. It still can be used by manual input to filter data.  
		*	**min_node_size**: min size of node allowed. Default to minpts.  
		*	**significant_seperation:** a value to determine when hierarchical node needs to be splited. Default 0.75 works for most cases based on tests in original paper.  
		*	**cluster_member_prob**: a value determines the threshold for cluster membership in a RD hist using GMM. default 0.5.   
		*	**node_similarity**: determine whether the child should replace current node. Does not affect leaves too much, but will reduce hierarchies for simplicity. Default 0.9.

	*	The fifth section: controls output and visualization.  
		*	**output_filename**: filename base for output, no extension.  
		*	**save_file**: save output files, like cluster_statistics, indexed pos, etc or not.  
		*	**show_visualization**: show visualization or not.  
		*	**show_background**: show noise/matrix ions or not.

## Instruction for output:

Several files will be given as output after clustering:
*	**\*_ol_RD_CD.txt**: a file contains the ordered list, reachability distance, and core distance from the OPTICS algorithm.  
*	**\*_cluster_stats.txt**: statistics of clusters, their index, center of mass, radius of gyration, equivalent spherical radius, contained ions.  
*	**\*_clusters.pos or .txt**: pos file for visualization clusters in IVAS. Replaced the m/z with cluster_id. Cluster_id is move up since IVAS does not allow -1 in range file, which represent noise in this algorithm.  
*	**\*_clusters.rrng**: range file for the corresponding pos to IVAS.  
*	**\*_log.txt**: saved parameters used in the analysis.  
*	**\*.cl**: dumped snapshot of clustering class binary object. Can be used for debug purpose. Normally turned off and not output.  

