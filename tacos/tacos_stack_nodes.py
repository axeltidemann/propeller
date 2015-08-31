'''
Read in TACOS data node by node.

python tacos_node_stack.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np

ratio = .75

with pd.get_store(sys.argv[1]) as store:
    # Remove start, stop when running the experiment on all the data, below as well
    unique_nodes = store.select_column('tacos', 'node', start=0, stop=10000).unique() 
    unique_nodes = np.random.permutation(unique_nodes)
    for node in unique_nodes:
        corpus = store.select('tacos', where='node == node', columns=['alarmtype', 'fhsseverity'], start=0, stop=10000)
        # To see the other columns available: print corpus.columns

        corpus.sort(inplace=True)

        # For each node, it should be no problem to keep everything in memory. Otherwise, chunksize iteration necessary.
        train = corpus.iloc[:int(ratio*len(corpus))]
        test = corpus.iloc[int(ratio*len(corpus)):]

        # You can access the timestamps like this: train.index
        # You can access the columns like this: train.alarmtype
        


        
