'''
Fix the missing counties by looking filling in based on those that already exist.

python tacos_corpus_analysis.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys
import multiprocessing as mp
from functools import partial
import time

import pandas as pd
import numpy as np

        
with pd.get_store(sys.argv[1]) as store:
    # 'identifier' has the fewest NaN values, only ~1e-7
    # 'node' also specifies location, and has only .1% missing values.
    nodes = np.unique(store.select('tacos', columns=['node', 'kpafylke']))

# This means: aggregate kpafylke on node, and indicate where there are more than 1 non-NaN entries.
unique = nodes.groupby(['node']).agg([lambda x: len(np.unique(x[ ~pd.isnull(x) ] )) > 1 ])
fillers = unique[ unique['kpafylke', '<lambda>'] ]
