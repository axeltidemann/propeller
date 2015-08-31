'''
Runs through the TACOS dataset, and computes error windows for each entry in the log for each node.
The error log finds the occurrence of each failure, weighted by exponential decay, related to the window_length. 
window_length is in minutes. The log is written to separate HDF5 files, one for each node. 

python tacos_failure_window.py /path/to/data.h5 window_length

Author: Axel Tidemann
'''

import os
import sys
import math
import multiprocessing as mp
from functools import partial
import uuid

import pandas as pd
import numpy as np

hdf5_path = sys.argv[1]
window_length = sys.argv[2]

def write_failure_log(node, hdf5_path, node_dir, window_length, unique_fhsseverity):
    with pd.get_store(hdf5_path) as store:
        failures = store.select('tacos', columns=['fhsseverity'], where='node == node').sort()

    node_filename = uuid.uuid4()

    with pd.HDFStore('{}/{}.h5'.format(node_dir, node_filename), 'w', complevel=9, complib='blosc') as node_store:
        targets = []
        for index in failures.index:
            target = { key: 0 for key in unique_fhsseverity }
            # Adding one second weeds out concurrent events. Should they be included?
            window = failures[index + pd.Timedelta(1, unit='s'):index + pd.Timedelta(window_length, unit='m')].copy()
            window.fillna('nan', inplace=True)
            window['decay'] = window.apply(lambda row: np.exp(math.log(.01)*(row.index-index).seconds/(60*window_length)))
            for _,row in window[::-1].iterrows(): 
                target[row.fhsseverity] = row.decay
            targets.append(target.values())

        failure_windows = np.array(targets)
        for i,key in enumerate(target.keys()):
            failures['error_{}'.format(key)] = failure_windows[:,i]
            
        node_store.append('data', failures, data_columns=True)
        node_store.append('node', node)
    

node_dir = '{}/{}min_window/'.format(hdf5_path[:hdf5_path.rfind('/')], window_length)
if not os.path.exists(node_dir):
    os.makedirs(node_dir)
        
with pd.get_store(hdf5_path) as store:
    unique_nodes = store.select_column('tacos', 'node').unique()
    unique_fhsseverity = store.select_column('tacos', 'fhsseverity').unique()
    unique_fhsseverity = [ 'nan' if np.isnan(key) else int(key) for key in unique_fhsseverity ]

par_write = partial(write_failure_log, hdf5_path=hdf5_path, node_dir=node_dir, window_length=int(window_length),
                    unique_fhsseverity=unique_fhsseverity)

pool = mp.Pool()
pool.map(par_write, unique_nodes)
