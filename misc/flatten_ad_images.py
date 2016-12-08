# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np

def flatten(h5, store_location):
    X = []
    with pd.HDFStore(h5, mode='r') as in_store:
        keys = in_store.keys()

    for key in keys:
        data = pd.read_hdf(h5, key)
        for x in data.values:
            if sum(x): # No image is coded as all zeros
                X.append(x)
                    
    h5name = os.path.join(store_location, os.path.basename(h5))
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:                
        out_store.append('data', pd.DataFrame(data=np.vstack(X), columns=data.columns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Flattens ad HDF5 files with images.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'ad_files',
        help='HDF5 files that contain the stored features', 
        nargs='+')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    args = parser.parse_args()

    par_flatten = partial(flatten, store_location=args.store_location)
    pool = mp.Pool()
    pool.map(par_flatten, args.ad_files)
