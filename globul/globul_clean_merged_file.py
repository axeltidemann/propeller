'''
Goes through the msc+ggsn merged HDF5 file, and removes typeCode/recordType - we are only interested in
timestamp, IMSI and site ID. For this to happen, we must convert 4-number GGSN cell IDs to proper cell IDs.
This conversion is found in the 'Huawei dump' metadata. Finally, save everything to site ID, which consists
of removing the last number of the cell ID.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from functools import partial

import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'merged',
    help='Path to the merged GGSN/MSC HDF5 file')
parser.add_argument(
    'meta',
    help='Path to the meta HDF5 file')
parser.add_argument(
    '--name',
    help='Name of the output HDF5 file with timestamp, IMSI and site ID',
    default='clean.h5')
parser.add_argument(
    '--chunk_size',
    help='Chunk size to iterate over HDF5 file',
    type=int,
    default=50000) 

args = parser.parse_args()

def cell_to_site(mapping, cell_ID):
    if len(cell_ID) == 5:
        return int(cell_ID)/10
    try:
        return int(mapping[cell_ID])/10
    except:
        return -1

with pd.get_store(args.merged) as merged_store, \
     pd.get_store(args.meta) as meta_store, \
     pd.HDFStore(args.name, 'w', complevel=9, complib='blosc') as clean_store:

    huawei = meta_store['huawei']

    mapping = {}
    for sac, cell_id in zip(huawei.sac, huawei.index):
        mapping[sac] = cell_id

    par_cell_to_site = partial(cell_to_site, mapping)

    data = merged_store.select('data', chunksize=args.chunk_size)

    for chunk in data:
        chunk['site_ID'] = chunk.cell_ID.apply(par_cell_to_site)
        del chunk['type_code']
        del chunk['cell_ID']
        chunk = pd.DataFrame(chunk.query('site_ID > -1'))

        clean_store.append('data', chunk, data_columns=True)
        
    print 'Cleaned Bulgaria data stored in HDF5.'
