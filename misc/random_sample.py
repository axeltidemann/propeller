# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse

import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Random samples from the specified HDF5 files, evenly 
    distributed across categories.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'source',
        help='''HDF5 file(s)''',
        nargs='+')
    parser.add_argument(
        'target',
        help='Where to put the HDF5 file, will have same name as the original HDF5 file.')
    parser.add_argument(
        'n',
        help='Number of samples',
        type=int)
    parser.add_argument(
        '--filename',
        help='Filename. If not specified, it will be random.h5',
        default=False)
    args = parser.parse_args()

    h5name = args.filename if args.filename else 'random.h5'
    
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as store:
        for h5 in args.source:
            data = pd.read_hdf(h5)
            perm = np.random.permutation(range(len(data)))[:args.n/len(args.source)]
            sample = data.iloc[perm]
            store.append('data', sample)
