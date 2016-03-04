'''
Creates a subset of the Bulgaria file that is only for the site IDs in "SOFIA_CITY".

Author: Axel.Tidemann@telenor.com
'''

import argparse

import pandas as pd
import numpy as np
from keras.utils import generic_utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--name',
    help='Name of the created file HDF5 file.',
    default='sofia.h5')
parser.add_argument(
    '--meta',
    help='Path to the meta HDF5 file')
parser.add_argument(
    '--source',
    help='Where bulgaria.h5 resides')
parser.add_argument(
    '--chunksize',
    help='Chunk size to iterate over HDF5 file',
    type=int,
    default=50000) 
args = parser.parse_args()

with pd.get_store(args.meta) as meta_store, \
     pd.get_store(args.source) as data_store, \
     pd.HDFStore(args.name, 'w', complevel=9, complib='blosc') as sofia_store:

    site_info = meta_store['site_info']
    sofia = site_info.query("Region == 'SOFIA_CITY'")

    progbar = generic_utils.Progbar(data_store.get_storer('data').nrows)
    
    for chunk in data_store.select('data', chunksize=args.chunksize):
        chunk['in_sofia'] = chunk.site_ID.apply(lambda x: int(x) in sofia.index)
        chunk = chunk.query('in_sofia == True')
        del chunk['in_sofia']
        sofia_store.append('data', chunk, data_columns=True)
        progbar.add(args.chunksize)
        
    print 'Sofia users stored in {}'.format(args.name)
