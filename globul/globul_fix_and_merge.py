'''
Chews through the wrongly formatted CSV file, columns should not have been float but object. Given it takes 43
hours to compute the one, this function was written instead. Also, the two would be nice to have merged.

python globul_fix_and_merge.py /path/to/ggsn.h5 /path/to/msc.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

ggsn_path = sys.argv[1]
msc_path = sys.argv[2]
merged_path = ggsn_path[:ggsn_path.rfind('/')]

with pd.get_store(ggsn_path) as ggsn_store, \
     pd.get_store(msc_path) as msc_store, \
     pd.HDFStore('{}/merged.h5'.format(merged_path), 'w', complevel=9, complib='blosc') as store:

    ggsn = ggsn_store.select('ggsn', chunksize=100000)
    for chunk in ggsn:
        chunk.IMSI = chunk.IMSI.astype(str)
        chunk.cell_ID = chunk.cell_ID.astype(str)
        store.append('data', chunk, data_columns=True)

    msc = msc_store.select('msc', chunksize=100000)
    for chunk in msc:
        chunk.callingSubscriberIMSI = chunk.callingSubscriberIMSI.astype(str)
        chunk.rename(columns={'callingSubscriberIMSI': 'IMSI'}, inplace=True)
        chunk.cell_ID = chunk.cell_ID.astype(str)
        store.append('data', chunk, data_columns=True)
