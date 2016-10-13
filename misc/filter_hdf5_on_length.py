# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser(description='''Softlinks
HDF5 files to target folder, that have more than the --lower_bound rows.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='HDF5 file(s)',
    nargs='+')
parser.add_argument(
    'target',
    help='Where to link the files')
parser.add_argument(
    '--lower_bound',
    help='Number of rows must be higher than this value',
    default=999,
    type=int)
args = parser.parse_args()

for h5 in args.source:
    with pd.HDFStore(h5, mode='r') as store:
        nrows = store.get_storer('data').nrows
        basename = os.path.basename(h5)
        if nrows > args.lower_bound:
            print '{}: {} rows'.format(basename, nrows)
            os.symlink(h5, os.path.join(args.target, basename))
