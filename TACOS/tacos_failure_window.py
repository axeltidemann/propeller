'''
Runs through the TACOS dataset, and computes error windows for each entry in the log for each node.
The error log finds the occurrence of each failure. The first occurrence is weighed with an exponential decay,
according to the phi parameter. The log is written to separate HDF5 files, one for each node. The window_length
parameter is in minutes.

python tacos_failure_window.py /path/to/data.h5 /path/to/nodes/ window_length phi

Author: Axel Tidemann
'''

import sys
import multiprocessing as mp

import pandas as pd
import numpy as np

hdf5_path = sys.argv[1]
node_path = sys.arg[2]
window_length = sys.argv[3]
phi = sys.argv[4]



def write_failure_log(node, hdf5_path, node_path, window_length, phi, unique_fhsseverity):
    with pd.get_store(hdf5_path) as store:
        failures = store.select('tacos', columns=['fhsseverity'], where="node == node")

    failures.sort(inplace=True)

    with pd.HDFStore('{}/{}_{}min.h5'.format(node_path, node, window_length), 'w') as node_store:
        for index in failures.index:
            slice = failures[index:index + pd.Timedelta(window_length, unit='m')]
            
        

with pd.get_store(hdf5_path) as store:
    unique_nodes = store.select_column('tacos', 'node').unique()
    unique_fhsseverity = store.select_column('tacos', 'fhsseverity').unique()
    
