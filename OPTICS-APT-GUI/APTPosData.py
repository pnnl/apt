"""
A class describes APT point cloud data. Should be updated with more functions for future expansion 

Authors: Jing Wang

This is a version modified for python 3 syntax.
"""

from PointCloud import PointCloud
from APTMassSpec import APTMassSpec
import APT_IOs as aptios
from OPTICS_APT import OPTICSAPT

import numpy as np

__version__ = '0.2'

class APTPosData(PointCloud, APTMassSpec):
    """
    A class that represent APT data. It inherits the PointCloud and APTMassSpec.
    
    More methods should be added to accomodate data analysis of APT point cloud in future.
    """
    def __init__(self):
        """
        Initialize APTPosData class need APT pos data (format [[x0,y0,z0,Da0],...])
        """
        PointCloud.__init__(self)
        APTMassSpec.__init__(self)

        self.pos = None
        self.identity = None
        self.rand_mass_label = None
        self._data_size = 0

        self._files = {'Data filename':None, 'RRNG filename':None}


    def load_data(self, data_fname, rrng_fname):
        """
        Load pos file and associated range file.
        """
        if data_fname.rsplit('.')[-1] == 'pos':
            self.pos = aptios.read_pos(data_fname)
        elif data_fname.rsplit('.')[-1] == 'txt':
            self.pos = aptios.read_txt(data_fname)

        dalton = self.pos[:, 3]
        self.load_range_file(rrng_fname)
        self.gen_mass_spec(dalton)
        self.rand_mass_label = np.random.permutation(dalton)
        self.identity = self.range_pos(self.pos)

        # save file names for log
        self._files['Data filename'] = data_fname
        self._files['RRNG filename'] = rrng_fname


    def range_pos(self, pos):
        """
        Filter the pos numpy array by range structured array. 
        The function acts like a filter, to select ions from pos.

        pos:
            a numpy array of format
            [[x0, y0, z0, Da0],
            [x1, y1, z1, Da1],
            ...]
        rng:
            a structured numpy array of format [(low_m, high_m, vol, ion_type,.., color), ...]
            dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')])

        Return:
            identity:
                a numpy array the same size as len(pos).
        """
        # Note that range is pre-sorted during read
        range_low = self.full_range['range_low']
        range_high = self.full_range['range_high']
        ion_types = np.append(self.full_range['ion_type'], 'Noise') # a last element 'Noise' is added to make new_size = old_size + 1, since low_idx and high_idx return idx from 0 to new_size
                                                    # eventually this last element does not matter because it will be guaranteed to be discarded due to outside of intervals
                                                    # we want to keep.
    
        # binary search to find insertion idx, which could be used to decide if m/z in pos is within range, where 1 is True and 0 is False 
        low_idx = np.searchsorted(range_low, pos[:, 3])
        high_idx = np.searchsorted(range_high, pos[:, 3])

        identity = ion_types[high_idx] # select potential ion idensities based on returned interval index. The contains element that does not fall within actual interval 
                                    # and will be discarded using the following logic array.

        logic = np.array(low_idx-high_idx, dtype=bool)

        # Assign ion identity to all points in pos. Outside of range will be assigned 'Noise' type.
        for idx in range(len(identity)):
            if not logic[idx]:
                identity[idx] = 'Noise'

        return identity

    def select_ions(self, select_ion_types):
        """
        select ions from data.

        identity:
            an array of ion_types correspond to each point

        select_ion_types:
            a list/array of ion types to be selected. \
            To select specific ions, use format ['A1B1', 'C2', ...];
            To select all ions, regardless of ranged or not ranged, use 'all';
            To select all ranged ions, use 'Ranged';
            To select all unranged ions, use 'Noise';

        Return:
            an array of booleans, True for in selecte ion types, False for not in selected.
        """
        assert len(select_ion_types) > 0, 'Invalid select ion ion types. Must be select "all", or at least one ion type'

        size = len(self.identity)
        selection = np.zeros(size, dtype=bool)

        if select_ion_types[0] == 'all':
            selection.fill(True)
        elif select_ion_types[0] == 'Ranged':
            selection = self.identity != 'Noise'
        elif select_ion_types[0] == 'Noise':
            selection = self.identity == 'Noise'
        else:
            select_ion_types = set(select_ion_types)
            for idx in range(size):
                if self.identity[idx] in select_ion_types:
                    selection[idx] = True

        return selection


#-------------------------------------------------------------------------------
# TEST
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    pos_name = 'R31_06365-v02.pos'
    rng_name = 'R31_06365-v02.rrng'
    ion_type = np.array(['all'])
    test_pos = aptios.read_pos(pos_name)
    _, test_rng = aptios.read_rrng(rng_name)
    
    
    time_init = time.time()

    time_fin = time.time()
    print('process time is ', str(time_fin-time_init), ' sec ')