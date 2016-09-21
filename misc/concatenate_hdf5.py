# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import os

import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Concatenates HDF5 files. Individual filenames will be keys in the resulting HDF5 file.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'source',
        help='''HDF5 file(s).''',
        nargs='+')
    parser.add_argument(
        '--filename',
        default='merged.h5',
        help='The resulting HDF5 file')
    parser.add_argument(
        '--shuffle',
        help='Whether to shuffle the data.',
        action='store_true')
    args = parser.parse_args()

    with pd.HDFStore(args.filename, 'w') as store:
        for h5 in args.source:
            print 'Reading {}'.format(h5)
            df = pd.read_hdf(h5)

            if args.shuffle:
                df = df.reindex(np.random.permutation(df.index))

            store.append(os.path.basename(h5), df)
