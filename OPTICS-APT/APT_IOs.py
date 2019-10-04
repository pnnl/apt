"""
A set of functions for input/output of Atom Probe Tomography related data and files.

Authors: Jing Wang

This is a version modified for python 3 syntax.
"""


import sys
import numpy as np
import re

__version__ = '0.2'

def read_txt(filename, header=1):
    """
    Read txt data file. Assuming the first line is file header and the 
    rest m row x n col are data. There should be no missing values. Delimiter is
    assumed to be space.

    It basically calls numpy.loadtx set delimiter default to space and skiprows default to 1
    for my laziness to input those two parameters everytime.
    
    filename:
        filename of the file want to be read. Default data is float type.
    header:
        number of rows that considered as header of the file.
    """
    return np.loadtxt(filename, delimiter=' ', skiprows=header, unpack=False)
   

def write_txt(filename, data, header='X Y Z Da'):
    """
    Write data to txt format. Basically calls np.savetxt, again just a minor helper to keep all output consistent format.
    
    data:
        a numpy array of m-by-n size
    filename:
        a string for filename to store data
    header:
        a string serves as file header   
    """
    np.savetxt(filename, data, delimiter=' ', header=header, comments='')
    return


def read_pos(fname):
    """
    Loads an APT .pos file and return a column-based data numpy array.

    Output formate is:
    [[x0, y0, z0, Da0],
     [x1, y1, z1, Da1],
     ...]
    """
    dt = np.dtype('>f4') #default pos file format is big endian byteorder four byte float. (check this in reference book)
    d = np.fromfile(fname, dtype=dt, count=-1) #use numpy from file to read a binary file. The d format is [x0,y0,z0,D0,x1,y1,z1,D1,...]
    data = np.reshape(d, (-1, 4)) # change the data shape to [m rows, 4 cols] from a total data size of 4m.

    print('Unpack finished.')
    return data


def write_pos(pos_fname, data):
    """
    Writing a numpy array data to APT readable .pos format.

    data:
        a n by 4 numpy array [[x0,y0,z0,Da0],
                              [x1,y1,z1,Da1],
                              ...]
    pos_fname:
        filename for the output pos.
    """
    if data.shape[1] != 4:
        sys.exit('data must be a numpy array of size m-by-4')

    assert data.dtype == '>f4' # very important, default float datatype will give corrupted result.

    flat_data = np.ndarray.flatten(data)
    flat_data.tofile(pos_fname) # Note to self, format in tofile dose not work for changing data type. 
                                   # So it need an assertation ahead to make sure data type is corrects.
    
    print('Pos file writing finished')   
    return


def read_rrng(rrng_fname):
    """
    Loads a .rrng file (IVAS format). Returns an array contains unique 'ions', and an array for 'ranges'.
    Range file format for IVAS is inconsistent among known and unknown range names.

    For known range name (contains only ion names on element table), format is:

        Range1=low_m high_m Vol:some_float ion_species:number (ion_species_2:number ...) Color:hexdecimal_num

    For unknown range names (any name given that not on element table):
        Range1=low_m high_m Vol:some_float Name:some_name Color:hexdecimal_num
    
    rrng_fname:
        filename for the range file.
    
    return:
        (ions, rrng): ions is a numpy array for all ion species in the range file; 
        rrng is a structured numpy array [(low_m, high_m, vol, ion_type,.., color), ...]
        dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')])
    """
    with open(rrng_fname, 'r') as f:
        rf = f.readlines()
    
    # pattern is a compiled regular experssion pattern to match range file format. works for both known unknown names.
    # pattern maching for strings: 
    #       r: raw string literal flag, means '\' won't be treated as escape character.
    #       'Ion([0-9]+)=([A-Za-z0-9]+)': it matches the ion types in first section such as ion1=C, ion2=Ti1O1,...
    #                                     'Ion' will match exactly character; (...) means a group, which will be retrived; [0-9] means a set of characters, in
    #                                     this case, any numeric character between 0 to 9; + means one or more repetition of the preceding regular experssion.
    #       '|' means 'or', that is either the precede one or the trailing one is matched.
    #       '-?': to match 0 or more minus sign, just put here for test purpose (some times I use -1 as M/Z ratio for noise points in test data)
    #       '\d+.\d': to match a float number, such as 123.456
    #       '([a-zA-Z0-9:a-zA-Z0-9 ]+)' : to match ion types, such as 'Ti:1 O:1' or 'Name:Cluster1'. Note the last space within square paranthesis is important.
    #       'Color:([A-Za-z0-9]{6})': to match color hexdecimal number. It matches exactly 6 characters.
    pattern = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+)|Range([0-9]+)=(-?\d+.\d+) (-?\d+.\d+) +Vol:(\d+.\d+) +([a-zA-Z0-9_+:a-zA-Z0-9 ]+) +Color:([A-Za-z0-9]{6})')
        
    elements = []
    rrngs = []
    for line in rf:
        match_re = pattern.search(line)
        if match_re:
            if match_re.groups()[0] is not None:
                elements.append(list(match_re.groups())[1])
            else:
                rrngs.append(match_re.groups()[3:])
                
    dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')]) # Note that in python 3 all strings are unicode now.
    rrngs = np.array(rrngs, dtype=dt)
    elements = np.array(elements)
    
    pattern_unknown_ion = re.compile(r'(?<=Name)([A-Za-z0-9]+)') # To obtain correct ion name for unknown ions in range file. Separate the 'Name' from e.g. 'Name:Cluster1'.

    # To further process ion types, reorganize like 'Ti:1 O:1' to 'Ti1O1'.
    for idx in range(len(rrngs)):
        rrngs[idx][3] = rrngs[idx][3].replace(':', '')
        rrngs[idx][3] = rrngs[idx][3].replace(' ', '')

        n = pattern_unknown_ion.search(rrngs[idx][3])
        if n:
            rrngs[idx][3] = n.groups()[0]

    # check the validity of range, there should be no interval overlap.
    sorted_rng_idx = np.argsort(rrngs['range_low'], kind='mergesort')
    range_low = rrngs['range_low']
    range_low = range_low[sorted_rng_idx]
    range_high = rrngs['range_high']
    range_high = range_high[sorted_rng_idx]
    assert np.all(range_low < range_high), 'Invalid range file: range overlap detected.'

    return elements, rrngs[sorted_rng_idx]




if __name__ == '__main__':
    
    pos_name = 'R31_06365-v02.pos'
    rng_name = 'R31_06365-v02.rrng'
    test_pos = read_pos(pos_name)
    elements, test_rng = read_rrng(rng_name)
    
