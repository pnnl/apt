"""
Main window for taking input and run functions.
"""



import sys
import os
import time
import numpy as np
from itertools import cycle
from threading import Event, Thread
from multiprocessing import Process, Queue, Pool

# import Visualization as visual # will be deleted after finish updating current one.
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QGridLayout
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QPointF
import pyqtgraph as pg
import pyqtgraph.opengl as gl


import ColorTable
from APTPosData import APTPosData
from OPTICS_APT import OPTICSAPT

class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        
    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == pg.Qt.QtCore.Qt.LeftButton and ev.double():
            self.autoRange()
        elif ev.button() == pg.Qt.QtCore.Qt.RightButton:
            self.raiseContextMenu(ev)
            
    def mouseDragEvent(self, ev):
        if ev.button() == pg.Qt.QtCore.Qt.RightButton:
            ev.ignore()
        elif ev.button() == pg.Qt.QtCore.Qt.LeftButton:
            pg.ViewBox.mouseDragEvent(self, ev)
        elif ev.button() == pg.Qt.QtCore.Qt.MiddleButton:
            # the following code was directly copied from a portion of the pyqtgraph mousedragevent
            # to change the behavior of mouse middle button.
            ev.accept()
            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = lastPos - pos 
            # dif = dif * -1

            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float)
            mask = mouseEnabled.copy()

            tr = dif*mask
            tr = self.mapToView(tr) - self.mapToView(pg.Point(0,0))
            x = tr.x() if mask[0] == 1 else None
            y = tr.y() if mask[1] == 1 else None
            
            self._resetTarget()
            if x is not None or y is not None:
                self.translateBy(x=x, y=y)
            self.sigRangeChangedManually.emit(self.state['mouseEnabled'])


class MyStream(QObject):
    message = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)

    def write(self, message):
        self.message.emit(str(message))


class MainWindown(QMainWindow):
    load_data_ready = pyqtSignal(object)
    def __init__(self):
        self.PosData = None
        self.Clusterer = None
        self.show_visual = False
        self.show_bg = False
        self.pos_vis = None
        self.clustering_vis = None
        self.point_size = None
        
        self.pool = Pool(processes=1) # experimenting


        self.app = QApplication([])
        QMainWindow.__init__(self)
        ui_fname = 'main_window.ui'
        self.ui = uic.loadUi(ui_fname)

        self.ui.button_data_fname.clicked.connect(self.button_open_data_file)
        self.ui.button_range_fname.clicked.connect(self.button_open_range_file)
        self.ui.button_load_data.clicked.connect(self.button_load_data_click)
        self.ui.button_show_ion_map.clicked.connect(self.button_show_ion_map_click)
        self.ui.button_run.clicked.connect(self.button_run_click)
        self.ui.button_terminate.clicked.connect(self.button_terminate_click)
        self.ui.button_show_visual.clicked.connect(self.button_show_visual_click)
        self.ui.button_show_visual.setEnabled(False)
        self.ui.button_save_output.clicked.connect(self.button_save_click)
        self.ui.groupBox_expert.hide()
        self.ui.checkBox_show_bg.stateChanged.connect(self.checked_bg)
        self.ui.checkBox_show_expert.stateChanged.connect(self.checked_expert)

        # self.my_stream = MyStream()
        # self.my_stream.message.connect(self.on_my_stream_message)
        # sys.stdout = self.my_stream # basically overide the stdout method with my_stream, and output it to the text box.

        self.ui.show()
        sys.exit(self.app.exec())


    @pyqtSlot(int, name='checked_bg')
    def checked_bg(self, state):
        if state == Qt.Checked:
            self.show_bg = True
        else:
            self.show_bg = False

    @pyqtSlot(int, name='checked_expert')
    def checked_expert(self, state):
        if state == Qt.Checked:
            self.ui.groupBox_expert.show()
        else:
            self.ui.groupBox_expert.hide()

    @pyqtSlot(str, name='on_myStream_message')
    def on_my_stream_message(self, message):
        self.ui.textBrowser.append(message)


    @pyqtSlot(name='button_open_data_file')
    def button_open_data_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Pos File', os.getcwd(), "Pos files (*.pos)")
        self.ui.le_data_fname.setText(fname[0])

    @pyqtSlot(name='button_open_range_file')
    def button_open_range_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Range File', os.getcwd(), "Range files (*.rrng, *.RRNG)")
        self.ui.le_rrng_fname.setText(fname[0])
        
    @pyqtSlot(name='button_load_data_click')
    def button_load_data_click(self):
        data_fname = self.ui.le_data_fname.text()
        rrng_fname = self.ui.le_rrng_fname.text()
        self.point_size = int(self.ui.le_point_size.text())
        self.ui.textBrowser.setText('Data file loaded: '+data_fname+'\n'+'Range file loaded: '+rrng_fname+'\n')
        self.ui.textBrowser.append('Loadeding...')

        self.DataLoader = APTDataLoader(data_fname, rrng_fname)
        self.thread_data_loader = QThread()
        self.DataLoader.data_ready.connect(self.get_load_data)
        self.DataLoader.finished.connect(self.thread_data_loader.quit)
        self.DataLoader.moveToThread(self.thread_data_loader)
        self.thread_data_loader.started.connect(self.DataLoader.load_APT_data)
        self.thread_data_loader.start()           

    @pyqtSlot(object, name='get_load_data')
    def get_load_data(self, data):
        if data == None:
            self.ui.textBrowser.append('Invalid filenames. Please make sure filenames are correct and files located in current folder.')
        else:
            self.PosData = data
            self.ui.textBrowser.append('Loaded Success.\n')
       

    @pyqtSlot(name='button_show_ion_map_click')
    def button_show_ion_map_click(self):
        self.ui.textBrowser.append('Processing...\n')
        self.pos_vis = APTDataViewer(self.PosData, point_size = self.point_size, max_ion=2000)

    @pyqtSlot(name='button_run_click')
    def button_run_click(self):
        ion_types = self.ui.le_ion_types.text()
        ion_types = ion_types.split(', ')
        method = self.ui.le_method.text()
        eps = float(self.ui.le_eps.text())
        minpts = int(self.ui.le_minpts.text())
        min_cluster_size = int(self.ui.le_min_cluster_size.text()) if self.ui.le_min_cluster_size.text() != 'None' else minpts
        
        rho = float(self.ui.le_density.text())
        det_eff = float(self.ui.le_det_eff.text())
        con = float(self.ui.le_solute_con.text())

        k = int(self.ui.le_k.text()) if self.ui.le_k.text() != 'None' else minpts
        significant_separation = float(self.ui.le_significant_separation.text())
        cluster_member_prob = float(self.ui.le_cluster_member_prob.text())
        min_node_size = int(self.ui.le_min_node_size.text()) if self.ui.le_min_node_size.text() != 'None' else minpts
        est_bg = float(self.ui.le_est_bg.text()) if self.ui.le_est_bg.text() != 'None' else np.inf
        node_similarity = float(self.ui.le_similarity.text())      
    
        if self.PosData == None:
            self.ui.textBrowser.append('Please Load Data First!\n')
        else:
            self.ui.textBrowser.append('Clustering started...')

            try:
                clustering_parameters = {'ion_types':ion_types, 
                            'method':method, 'eps':eps, 'minpts':minpts, 
                            'min_node_size':min_node_size, 
                            'significant_separation':significant_separation, 
                            'k':k, 'node_similarity':node_similarity, 
                            'cluster_member_prob':cluster_member_prob, 
                            'est_bg':est_bg, 'min_cluster_size':min_cluster_size}

                self.DoClustering = DoClustering(self.PosData, clustering_parameters)
                self.thread_do_clustering = QThread()
                self.DoClustering.cluster_ready.connect(self.get_clustering_result)
                self.DoClustering.finished.connect(self.thread_do_clustering.quit)
                self.DoClustering.moveToThread(self.thread_do_clustering)
                self.thread_do_clustering.started.connect(self.DoClustering.cluster_analysis)
                self.thread_do_clustering.start()
            except:
                self.ui.textBrowser.append('Something is wrong in cluster analysis setting, please check again...')


    @pyqtSlot(object, name = 'get_clustering_result')
    def get_clustering_result(self, data):
        self.Clusterer = data
        self.ui.textBrowser.append('Clustering finished.\n')
        self.ui.button_show_visual.setEnabled(True)

    @pyqtSlot(name='button_terminate_click')
    def button_terminate_click(self):
        self.thread_do_clustering.quit()
        self.ui.textBrowser.append('Cluster analysis terminated!\n')

    @pyqtSlot(name = 'button_show_visual_click')
    def button_show_visual_click(self):
        self.clustering_vis = ClusteringView(self.Clusterer, self.show_bg, self.point_size)

    @pyqtSlot(name='button_save_click')
    def button_save_click(self):
        output_filename = self.ui.le_output_fname.text()

        try:
            if self.Clusterer.RD is not None:
                self.Clusterer.write_RD_to_file(output_filename+'_ol_RD_CD')

            self.Clusterer.cluster_stats(output_filename+'_cluster_stats')
            self.Clusterer.write_cluster_to_file(output_filename+'_clusters')
            self.Clusterer.write_indexed_cluster_rrng(output_filename+'_clusters')
            self.Clusterer.write_log(output_filename+'_log')
        except:
            self.ui.textBrowser.append('Invalid output filename. Try again.')

class APTDataLoader(QObject):
    """
    A class to load apt data
    """
    data_ready = pyqtSignal(object)
    finished = pyqtSignal()
    def __init__(self, data_fname, rrng_fname):
        super().__init__()
        self.data_fname = data_fname
        self.rrng_fname = rrng_fname

    @pyqtSlot(name = 'load_APT_data')
    def load_APT_data(self):
        try:
            PosData = APTPosData()
            PosData.load_data(self.data_fname, self.rrng_fname)
            self.data_ready.emit(PosData)
        except:
            self.data_ready.emit(None)
        self.finished.emit()


class APTDataViewer(QObject):
    """
    A class to show ion maps and mass spectrum.
    """
    # finished = pyqtSignal()

    def __init__(self, PosData, point_size=2, max_ion=1000):
        super().__init__()
        self.apt_data_viewer = QWidget()
        self.layout = QGridLayout()
        self.apt_data_viewer.setLayout(self.layout)
        self.apt_data_viewer.sizeHint = lambda: pg.QtCore.QSize(1920, 1200)

        self.point_cloud_view = gl.GLViewWidget()
        self.point_cloud_view.sizeHint = lambda: pg.QtCore.QSize(1920, 700)
        self.point_cloud_view.setBackgroundColor('w')
        self.point_cloud_view.opts['distance'] = 2000
        self.point_cloud_view.opts['fov'] = 1

        vb = CustomViewBox() # A custom view box added to mass spec view, this way the visualization performance is greatly enhanced.
        self.mass_spec = pg.PlotWidget(viewBox=vb)
        self.mass_spec.setBackground('w')
        self.mass_spec.sizeHint = lambda: pg.QtCore.QSize(1920, 500)
     
        self.mass_spec.setDownsampling(auto=True, mode='mean') # because there are some werid issue using downsampling mode when there are 0s in mass spec
        # self.mass_spec.setLogMode(y=True)
        
        self.mass_spec.setLabel('bottom', text='Mass-to-Charge (Dalton)')
        self.mass_spec.setLabel('left', text='Counts')

        self.layout.addWidget(self.point_cloud_view, 0, 0)
        self.layout.addWidget(self.mass_spec, 1, 0)


        self.MassWorker = APTMassWorker(PosData.m2z, PosData.intensity)
        self.thread_mass = QThread()
        self.MassWorker.plot_ready.connect(self.mass_plot_ready)
        self.MassWorker.finished.connect(self.thread_mass.quit)
        self.MassWorker.moveToThread(self.thread_mass)
        self.thread_mass.started.connect(self.MassWorker.plot_mass_spec)
        self.thread_mass.start()

        self.PosWorker = APTPosWorker(PosData.pos[:, 0:3], PosData.identity, PosData.ions, point_size, max_ion)
        self.thread_pos = QThread()
        self.PosWorker.gl_ready.connect(self.point_cloud_ready)
        self.PosWorker.finished.connect(self.thread_pos.quit)
        self.PosWorker.moveToThread(self.thread_pos)
        self.thread_pos.started.connect(self.PosWorker.visualize_pos)
        self.thread_pos.start()

        self.apt_data_viewer.show()


    @pyqtSlot(object, name = 'mass_plot_ready')
    def mass_plot_ready(self, item):
        self.mass_spec.addItem(item)

    @pyqtSlot(object, name = 'point_cloud_ready')
    def point_cloud_ready(self, sp1):
        self.point_cloud_view.addItem(sp1)

class APTMassWorker(QObject):
    """
    Worker class to plot mass spec for APTDataViewer using threading
    """
    finished = pyqtSignal()
    plot_ready = pyqtSignal(object)
    def __init__(self, m2z, intensity):
        super().__init__()
        self.m2z = m2z
        self.intensity = intensity

    @pyqtSlot(name = 'plot_mass_spec')
    def plot_mass_spec(self):
        """
        Plot mass spec. Maybe need some smart downsampling in future. /
        Downsample in current pyqtgraph plot is slow and working weired.
        """
        # vb = CustomViewBox()
        # item = pg.PlotDataItem( self.m2z, self.intensity, pen=pg.mkPen('k', width=2))
        item = pg.PlotCurveItem(self.m2z, self.intensity, pen=pg.mkPen(0.0, width=2), stepMode=True)

        # item = pg.BarGraphItem(viewbox=vb, x=self.m2z, height=self.intensity, width=0.01)

        self.plot_ready.emit(item)
        self.finished.emit()
        return


class APTPosWorker(QObject):
    """
    Worker class to plot point cloud for APTDataViewer using threading.
    """
    finished = pyqtSignal()
    gl_ready = pyqtSignal(object)

    def __init__(self, coord, ion_identity, ion_types, point_size=2, max_ion=1000):
        super().__init__()
        self.coord = coord
        self.ion_identity = ion_identity
        self.ion_types = ion_types
        self.point_size = point_size
        self.max_ion = max_ion

    @pyqtSlot(name = 'visualize_pos')
    def visualize_pos(self):
        """
        Plot and show cluster point clouds.

        !!!!!!!! Coord is byte sensitive, it will plot scatter so wrong if use default '>f4' data type as in our program.
        Change to default float solves the problem.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        data = self.coord.astype(np.float32)
        data[:, 2] = -data[:, 2] # reverse the z axis to show needle right.

        # assigning colors
        color_table = ColorTable.gen_color_table(style='float')
        cycol = cycle(color_table)
        ion_color = dict()
        ion_color['Noise'] = [0.0, 0.0, 0.0]
        for u_id in self.ion_types:
            ion_color[u_id] = next(cycol)

        ion_counts = dict.fromkeys(self.ion_types, 0)
        ion_counts['Noise'] = 0

        ion_types = set(self.ion_types)
        color = np.ones((data.shape[0], 4))
        color[:, 3] = 1.0
        for idx, c_id in enumerate(self.ion_identity):
            if c_id in ion_types:
                color[idx, 0:3] = ion_color[c_id]
                ion_counts[c_id] += 1

        selection = np.empty(len(self.ion_identity), dtype=bool)
        selection.fill(False)
        # down sampleing
        for i_type, i_count in ion_counts.items():
            if i_type in ion_types:
                current_arr = np.nonzero(self.ion_identity == i_type)[0]
                if i_count > self.max_ion:
                    down_sample_arr = np.random.choice(current_arr, size=self.max_ion, replace=False)
                    selection[down_sample_arr] = True
                else:
                    selection[current_arr] = True

        sp1 = gl.GLScatterPlotItem(pos=data[selection], size=self.point_size, color=color[selection], pxMode=True)
        sp1.setGLOptions('translucent')

        self.gl_ready.emit(sp1)
        self.finished.emit()

        return

class DoClustering(QObject):
    """
    A class to do clustering
    """
    cluster_ready = pyqtSignal(object)
    finished = pyqtSignal()
    def __init__(self, PosData, clustering_parameters):
        super().__init__()
        self.PosData = PosData
        self.paras = clustering_parameters

    @pyqtSlot(name = 'cluster_analysis')
    def cluster_analysis(self): # here background estimation could be added in future

        method = self.paras['method']
        
        init_time = time.time()
        Clusterer = OPTICSAPT(self.PosData)
        Clusterer.ion_selection(self.paras['ion_types'])

        if method == 'DBSCAN':
            Clusterer.DBSCAN_clustering(self.paras['eps'], self.paras['minpts'])
        elif method == 'OPTICS-APT':
            Clusterer.do_optics(eps=self.paras['eps'], minpts=self.paras['minpts'])
            Clusterer.hierarchy_clustering(min_node_size=self.paras['min_node_size'], 
                                            significance_of_separation=self.paras['significant_separation'],
                                            k=self.paras['k'],
                                            similarity_level=self.paras['node_similarity'],
                                            cluster_member_prob=self.paras['cluster_member_prob'])
            Clusterer.create_clusters(est_bg=self.paras['est_bg'], min_cluster_size=self.paras['min_cluster_size'])

        fin_time = time.time()
        print('total computation time is: ', fin_time-init_time)

        self.cluster_ready.emit(Clusterer)
        self.finished.emit()

        return


class ClusteringView(QObject):
    """
    A viewer for viewing clustering result of OPTICS-APT, showing ion maps, RD, RD histogram.
    """
    def __init__(self, Clusterer, show_bg, point_size=3):
        super().__init__()

        self.clustering_view = QWidget()
        self.layout = QGridLayout()
        self.clustering_view.setLayout(self.layout)

        self.RD_plot = pg.PlotWidget()
        self.RD_hist = pg.PlotWidget()
        self.point_cloud_view = gl.GLViewWidget()

        self.RD_plot.sizeHint = pg.QtCore.QSize(800, 600)
        self.point_cloud_view.sizeHint = lambda: pg.QtCore.QSize(800, 600)
        self.point_cloud_view.setSizePolicy(self.RD_plot.sizePolicy())

        self.layout.addWidget(self.point_cloud_view, 0, 0, 2, 1)
        self.layout.addWidget(self.RD_plot, 0, 1)
        self.layout.addWidget(self.RD_hist, 1, 1)


        self.point_cloud_view.setBackgroundColor('w')
        self.point_cloud_view.opts['distance'] = 2000
        self.point_cloud_view.opts['fov'] = 1
        
        self.thread_show_clustering = QThread()
        self.ShowClustering = ShowClustering(Clusterer, point_size, show_bg)
        self.ShowClustering.gl_finished.connect(self.thread_show_clustering.quit)
        self.ShowClustering.gl_ready.connect(self.point_cloud_ready)
        self.ShowClustering.moveToThread(self.thread_show_clustering)
        self.thread_show_clustering.started.connect(self.ShowClustering.show_clustering)
        self.thread_show_clustering.start()

        # start plot RD
        ordered_RD = Clusterer.RD[Clusterer.ordered_lst]
        self.RD_plot.setBackground('w')
        self.RD_plot.setDownsampling(auto=True)
        self.RD_plot.plot(ordered_RD, pen=pg.mkPen('k', width=2))

        color_table = ColorTable.gen_color_table(style='int')
        color_table.pop(0)
        cycol = cycle(color_table)
        # num_leaves = len(leaves)
        leaves = Clusterer.obtain_leaf_nodes()
        count = 0
        for node in leaves:
            if node.is_cluster:
                item = node.index_range
                ave_RD = node.average_RD(ordered_RD)
                self.RD_plot.plot(item, [ave_RD, ave_RD], pen=pg.mkPen(color=next(cycol), width=2))
                count += 1


        # start plot RD hist
        self.RD_hist.setBackground('w')
        hist, bins=np.histogram(Clusterer.RD, bins=50, density=True)
        curve = pg.PlotCurveItem(bins, hist, pen=pg.mkPen('b', width=2), stepMode=True)
        self.RD_hist.addItem(curve)


        # self.thread_show_RD = QThread()

        # self.thread_show_RD_hist = QThread()

        self.clustering_view.show()

    @pyqtSlot(float, name = 'processing_time')
    def processing_time(self, t):
        print('processing time is', t, 's')

    @pyqtSlot(object, name = 'point_cloud_ready')
    def point_cloud_ready(self, sp1):
        self.point_cloud_view.addItem(sp1)

    @pyqtSlot(object, name = 'RD_ready')
    def RD_ready(self, item):
        # self.point_cloud_view.addItem(sp1)
        pass

    @pyqtSlot(object, name = 'RD_hist_ready')
    def RD_hist_ready(self, item):
        # self.point_cloud_view.addItem(sp1)
        pass


class ShowClustering(QObject):
    gl_ready = pyqtSignal(object)
    gl_finished = pyqtSignal()

    def __init__(self, Clusterer, point_size, show_bg):
        super().__init__()
        self.Clusterer = Clusterer
        self.point_size = point_size
        self.show_bg = show_bg

    @pyqtSlot(name = 'show_clustering')
    def show_clustering(self):
        
        data = self.Clusterer.current_PointCloud.coord.astype(np.float32)
        data[:, 2] = -data[:, 2] # reverse the z axis to show needle right.
        cluster_id = self.Clusterer.cluster_id

        if not self.show_bg:
            selection = cluster_id != -1
        else:
            selection = np.ones(len(cluster_id), dtype=bool)
            selection.fill(True)

        color_table = ColorTable.gen_color_table(style='float')
        color_table.pop(0)
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

        sp1 = gl.GLScatterPlotItem(pos=data[selection], size=self.point_size, color=color[selection], pxMode=True)
        sp1.setGLOptions('translucent')
        
        self.gl_ready.emit(sp1)
        self.gl_finished.emit()

        return

if __name__ == '__main__':
    MW = MainWindown()