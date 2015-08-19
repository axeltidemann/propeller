'''
Fix the missing counties by looking filling in based on those that already exist.

python tacos_corpus_analysis.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import pandas as pd

import sys

with pd.get_store(sys.argv[1]) as store:
    # 'identifier' is the one with fewest NaN values, only ~1e-7
    result = store.select('tacos', columns=['identifier', 'kpafylke'])
    
    
