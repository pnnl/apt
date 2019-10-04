"""
Visualization module. Need to split into smaller ones in future.
"""


import pyqtgraph as pg
import pyqtgraph.opengl as gl

import ColorTable
import TreeNode

import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import sys
from copy import deepcopy


def visualize_node_tree(node, ordered_RD, output_name=None):
    """
    Visualize entire node tree.

    node:
        the starting node of tree/subtree of interesting for plotting
    output_name:
        filename for figure to be saved.
    """
    def graphNode(node, num, ax):
        nonlocal ordered_RD
        nonlocal cycol
        start, end = node.index_range
        ax.hlines(num, start,end, color=next(cycol), linewidth=2.0)
        for item in node.children:
            graphNode(item, num - 0.5, ax)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cycol = cycle('bgrcmk')
    num = 0.0
    graphNode(node, num, ax)
    plt.show(fig)

    if output_name != None:
        plt.savefig(output_name+'.png', orientation='portrait')


def visualize_RD(leaves, ordered_RD, widget, output_file=None):
    """
    Plot RD and clustering info on it.
    """
    plotWidget = widget

    plotWidget.setBackground('w')
    plotWidget.setDownsampling(auto=True)
    plotWidget.plot(ordered_RD, pen=pg.mkPen('k', width=2))

    color_table = ColorTable.gen_color_table(style='int')
    cycol = cycle(color_table)
    # num_leaves = len(leaves)

    count = 0
    for node in leaves:
        item = node.index_range
        ave_RD = node.average_RD(ordered_RD)
        plotWidget.plot(item, [ave_RD, ave_RD], pen=pg.mkPen(color=next(cycol), width=2))
        count += 1


def visualize_RD_hist(RD, widget):
    """
    Plot histogram of RD.
    """
    widget.setBackground('w')
    hist, bins=np.histogram(RD, bins=50, density=True)
    curve = pg.PlotCurveItem(bins, hist, pen=pg.mkPen('b', width=2), stepMode=True)
    widget.addItem(curve)


def visualize_clusters(coord, cluster_id, GLView, background):
    """
    Plot and show cluster point clouds.

    !!!!!!!! Coord is byte sensitive, it will plot scatter so wrong if use default '>f4' data type as in our program.
    Change to default float solves the problem.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    """
    w = GLView
    data = coord.astype(np.float32)
    data[:, 2] = -data[:, 2] # reverse the z axis to show needle right.

    if not background:
        selection = cluster_id != -1
    else:
        selection = np.ones(len(cluster_id), dtype=bool)
        selection.fill(True)

    size = 5

    color_table = ColorTable.gen_color_table(style='float')
    color_table = np.delete(color_table, [0], axis=0)
    cycol = cycle(color_table)
    cluster_color = dict()
    unique_cluster_id = np.unique(cluster_id)

    for unique_id in unique_cluster_id:
        if unique_id == -1:
            cluster_color[unique_id] = [0.0, 0.0, 0.0]
        else:
            cluster_color[unique_id] = next(cycol)

    color = np.empty((data.shape[0], 4))
    color[:, 3] = 1.0
    for idx, c_id in enumerate(cluster_id):
        color[idx, 0:3] = cluster_color[c_id]

    sp1 = gl.GLScatterPlotItem(pos=data[selection], size=size, color=color[selection], pxMode=True)
    sp1.setGLOptions('translucent')
    w.setBackgroundColor('w')
    w.opts['distance'] = 2000
    w.opts['fov'] = 1
    w.addItem(sp1)


def visualization(Clusterer, background=True):
    """
    Visualization of clusters and RD plot.
    """
    app = pg.Qt.QtGui.QApplication.instance()
    if app == None:
        app = pg.Qt.QtGui.QApplication([])

    w = pg.Qt.QtGui.QWidget()
    layout = pg.Qt.QtGui.QGridLayout()
    w.setLayout(layout)
    plot = pg.PlotWidget()
    hist = pg.PlotWidget()
    view = gl.GLViewWidget()

    plot.sizeHint = pg.QtCore.QSize(800, 600)
    view.sizeHint = lambda: pg.QtCore.QSize(800, 600)
    view.setSizePolicy(plot.sizePolicy())
    layout.addWidget(view, 0, 0, 2, 1)
    layout.addWidget(plot, 0, 1)
    layout.addWidget(hist, 1, 1)

    leaves = []
    leaves = TreeNode.retrieve_leaf_nodes(Clusterer.hierarchy_RootNode, leaves)
    visualize_RD_hist(Clusterer.RD, hist)
    visualize_RD(leaves, Clusterer.RD[Clusterer.ordered_lst], plot)
    visualize_clusters(Clusterer.current_PointCloud.coord, Clusterer.cluster_id, view, background)

    w.show()
    sys.exit(app.exec_())

    
if __name__ == '__main__':
    import numpy as np

    fname = 'test_optics_clusters.txt'
    data = np.loadtxt(fname, delimiter=' ', skiprows= 1)
    coord = data[:, 0:3]
    cluster_id = data[:, 3].astype(int)

    coord = np.load('test_coord.npy')
    coord = coord.astype(np.float)
    # print(cluster_id)

    app = pg.Qt.QtGui.QApplication.instance()
    if app == None:
        app = pg.Qt.QtGui.QApplication([])

    w = pg.Qt.QtGui.QWidget()
    layout = pg.Qt.QtGui.QGridLayout()
    w.setLayout(layout)
    view = gl.GLViewWidget()
    plot = pg.PlotWidget()


    view.sizeHint = lambda: pg.QtCore.QSize(800, 600)
    view.setSizePolicy(plot.sizePolicy())
    layout.addWidget(view, 0, 0, 2, 1)

    background = True
    if not background:
        selection = cluster_id != 1
    else:
        selection = np.ones(len(cluster_id), dtype=bool)
        selection.fill(True)

    visualize_clusters(coord[selection], cluster_id[selection], view, background)

    
    # x = data[:, 0]
    # y = data[:, 1]
    # z = data[:, 2]
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='b')
    # plt.show(fig)


    # test = np.random.random((100, 3))
    # test_id = np.random.random((100))
    # test = test[test_id > 0.3]
    # test_id = test_id[test_id > 0.3]
    # visualize_clusters(test, test_id, view)

    w.show()
    sys.exit(app.exec_())