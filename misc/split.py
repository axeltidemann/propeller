# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import os
import multiprocessing as mp
from functools import partial

import pandas as pd

def h5_split(h5, ratio, train_target, test_target):

    with pd.HDFStore(h5, mode='r') as in_store:
        keys = sorted(in_store.keys())

    train_store = pd.HDFStore(os.path.join(args.train_target, os.path.basename(h5)),
                              mode='w', complevel=9, complib='blosc')

    test_store = pd.HDFStore(os.path.join(args.test_target, os.path.basename(h5)),
                             mode='w', complevel=9, complib='blosc')
        
    for key in keys:
        data = pd.read_hdf(h5, key)
        train_store.append(key, data.iloc[0:int(len(data)*ratio)])
        test_store.append(key, data.iloc[int(len(data)*ratio):])

    train_store.close()
    test_store.close()
        
    print '{} split into {} and {}'.format(h5, train_target, test_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Splits HDF5 state folders into a training and testing split.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'source',
        help='''HDF5 file(s) for categories.''',
        nargs='+')
    parser.add_argument(
        'train_target',
        help='Where to put the train HDF5 files.')
    parser.add_argument(
        'test_target',
        help='Where to put the test HDF5 files.')
    parser.add_argument(
        '--ratio',
        help='Train/test split ratio',
        default=.8,
        type=float)
    args = parser.parse_args()

    par_split = partial(h5_split, ratio=args.ratio, train_target=args.train_target, test_target=args.test_target)
    pool = mp.Pool()
    pool.map(par_split, args.source)
