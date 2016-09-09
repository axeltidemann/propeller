# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import os

import pandas as pd

def h5_split(h5, ratio, train_target, test_target, table):
    data = pd.read_hdf(h5, table)

    with pd.HDFStore(os.path.join(args.train_target, os.path.basename(h5)), 'w') as store:
        store['data'] = data.iloc[0:int(len(data)*ratio)]

    with pd.HDFStore(os.path.join(args.test_target, os.path.basename(h5)), 'w') as store:
        store['data'] = data.iloc[int(len(data)*ratio):]

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
    parser.add_argument(
        '--table',
        help='Name of HDF5 table',
        default='data')
    args = parser.parse_args()

    for h5 in args.source:
        h5_split(h5, args.ratio, args.train_target, args.test_target, args.table)
