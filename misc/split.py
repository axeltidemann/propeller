# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import os
import multiprocessing as mp

import pandas as pd

def h5_split(h5):

    with pd.HDFStore(h5, mode='r') as in_store:
        keys = sorted(in_store.keys())

        train_store = pd.HDFStore(os.path.join(args.train_target, os.path.basename(h5)),
                                  mode='w', complevel=9, complib='blosc')

        test_store = pd.HDFStore(os.path.join(args.test_target, os.path.basename(h5)),
                                 mode='w', complevel=9, complib='blosc')

        if len(keys) == 1:
            data = in_store[keys[0]]

            train_store.append('data', data[:int(len(data)*args.ratio)])
            test_store.append('data', data[int(len(data)*args.ratio):])
        else:
            for key in keys[:int(len(keys)*args.ratio)]: 
                train_store.append(key, in_store[key])

            for key in keys[int(len(keys)*args.ratio):]:
                test_store.append(key, in_store[key])

    train_store.close()
    test_store.close()
        
    print '{} split into {} and {}'.format(h5, args.train_target, args.test_target)


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
    
    pool = mp.Pool()
    pool.map(h5_split, args.source)
