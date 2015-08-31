'''
Read in TACOS data node by node, write a train set and a test set.

python tacos_node_stack.py /path/to/data.h5 ratio

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np

path = sys.argv[1]
ratio = float(sys.argv[2])

with pd.get_store(path) as store, \
     pd.HDFStore('{}.train'.format(path), 'w', complevel=9, complib='blosc') as train_store, \
     pd.HDFStore('{}.test'.format(path), 'w', complevel=9, complib='blosc') as test_store:

    # Remove start, stop when running the experiment on all the data. See also below.
    unique_nodes = store.select_column('tacos', 'node', start=0, stop=10000).unique() 
    unique_nodes = np.random.permutation(unique_nodes)
    
    for node in unique_nodes:
        corpus = store.select('tacos', where='node == node', columns=['alarmtype', 'fhsseverity'], start=0, stop=10000)
        corpus.sort(inplace=True)
        # To see the other columns available: print corpus.columns

        train = corpus.iloc[:int(ratio*len(corpus))]
        test = corpus.iloc[int(ratio*len(corpus)):]

        kwargs = {'data_columns': True, 'min_itemsize': {'alarmtype': 64}}
        train_store.append('train', train, **kwargs)
        test_store.append('test', test, **kwargs)

        # You can access the timestamps like this: train.index
        # You can access the columns like this: train.alarmtype
