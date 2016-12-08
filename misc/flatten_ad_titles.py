# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
from keras.preprocessing import sequence

def flatten(h5):
    X = []
    with pd.HDFStore(h5, mode='r') as in_store:
        keys = in_store.keys()
        ad_ids = list(set([ key.split('/')[1] for key in keys ]))

        X = []
        for ad_id in list(ad_ids):
            try: 
                data = in_store['{}/text'.format(ad_id)]
                X.append(np.squeeze(data))
            except:
                print '{} did not exist. Weird.'.format(ad_id)
                ad_ids.remove(ad_id)

        X = sequence.pad_sequences(X, maxlen=args.seq_len, truncating='post', padding='post')
        X = np.vstack(X)

    columns = [ 'c{}'.format(i) for i in range(args.seq_len) ]

    h5name = os.path.join(args.store_location, os.path.basename(h5))
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:                
        out_store.append('data', pd.DataFrame(index=ad_ids, data=X, columns=columns))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Flattens HDF5 files with image + title information into just title.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'ad_files',
        help='HDF5 files that contain titles', 
        nargs='+')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    parser.add_argument(
        '--seq_len',
        help='How many characters to store in the flattened file',
        type=int,
        default=100)
    args = parser.parse_args()

    pool = mp.Pool()
    pool.map(flatten, args.ad_files)
