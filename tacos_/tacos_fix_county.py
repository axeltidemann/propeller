'''
Fix the missing counties by looking filling in based on those that already exist.

python tacos_corpus_analysis.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np

with pd.get_store(sys.argv[1]) as store:
    # 'identifier' has the fewest NaN values, only ~1e-7
    # 'node' also specifies location, and has only .1% missing values.
    nodes = store.select('tacos', columns=['node', 'kpafylke'])

nodes.replace(['n/a', 'Ukjent'], [np.nan, np.nan], inplace=True)
# This means: aggregate kpafylke on node, and indicate where there are more than 1 non-NaN entries.
manys = nodes.groupby(['node']).agg([lambda x: len(np.unique(x[ ~pd.isnull(x) ] )) > 1])
# Find nodes where there is exactly one 1 non-NaN entry.
uniques = nodes.groupby(['node']).agg([lambda x: len(np.unique(x[ ~pd.isnull(x) ] )) == 1])

sums = nodes.groupby(['node']).agg([len])
real_sums = sums[ uniques['kpafylke', '<lambda>'] ]

print """
{}% of nodes with kpafylke have more than one kpafylke assigned.
{}% of unique nodes have exactly one kpafylke assigned.
Using nodes, we can fill in {}% of the missing kpafylke values.
""".format(100.*np.mean(manys)/(np.mean(manys)+np.mean(uniques)),
           100.*np.mean(uniques),
           100.*np.sum(real_sums)/len(nodes))
 
