'''
MSC and GGSN data are merged into one canonical HDF5 file with
cell_ID, ISMI and type_code. Important: the type_code stems from
ggsn.recordType and msc.typeCode. The merged file (ggsn+msc.h5) will be put
in the same place as the ggsn.h5 file.

python globul_fix_and_merge.py /path/to/ggsn.h5 /path/to/msc.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

ggsn_path = sys.argv[1]
msc_path = sys.argv[2]
merged_path = '{}ggsn+msc.h5'.format(ggsn_path[:ggsn_path.rfind('/')+1] if ggsn_path.rfind('/') > -1 else '')

with pd.get_store(ggsn_path) as ggsn_store, \
     pd.get_store(msc_path) as msc_store, \
     pd.HDFStore(merged_path, 'w', complevel=9, complib='blosc') as merged_store:

    ggsn = ggsn_store.select('data', chunksize=5e5)
    for chunk in ggsn:
        chunk.rename(columns={'recordType': 'type_code'}, inplace=True)
        merged_store.append('data', chunk, data_columns=True)

    msc = msc_store.select('data', chunksize=5e5)
    for chunk in msc:
        chunk['IMSI'] = chunk.callingSubscriberIMSI + chunk.calledSubscriberIMSI
        chunk.rename(columns={'typeCode': 'type_code'}, inplace=True)
        del chunk['callingSubscriberIMSI']
        del chunk['calledSubscriberIMSI']
        merged_store.append('data', chunk, data_columns=True)

    print '{} and {} merged and stored in {}'.format(ggsn_path, msc_path, merged_path)
