"""
A simple class represents the mass spec .

Author: Jing Wang
"""
import APT_IOs as aptios
import numpy as np

MZ_MIN = 0 # mass to charge min
MZ_MAX = 200 # mass to charge max
class APTMassSpec:
    def __init__(self):
        self.m2z = None
        self.intensity = None        
        self.full_range = None
        self.elements = None
        self.ions = None


    def gen_mass_spec(self, mz_lst, bin_num=20000):
        """
        Take an array of mass-to-charge ratio, generate mass spec.
        """
        # m2z is from [MZ_MIN, MZ_MAX], with bin_num bins. Note its length is len(hist) + 1
        # intensity: counts from samples in that bin.
        self.intensity, self.m2z = np.histogram(mz_lst, bins=bin_num, range=(MZ_MIN, MZ_MAX))


    def load_range_file(self, fname):
        """
        range data format is:
            dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')])
        """
        self.elements, self.full_range = aptios.read_rrng(fname)
        self.ions = np.unique(self.full_range['ion_type'])


    def calc_comp(self, selected_ions=None):
        """
        Compute the composition for selected ions. 

        selected_ion default is None. if not given, composition for \
        all ion types are calculated. 

        To be implemented.
        """
        pass
