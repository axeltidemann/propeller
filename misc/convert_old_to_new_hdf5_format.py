# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='''
Stores HDF5 files processed in fixed format to table format.
Also writes the features as separate columns.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'h5',
    help='HDF5 file(s) with images.',
    nargs='+')
parser.add_argument(
    'target',
    help='Where to put the HDF5 file(s)')
args = parser.parse_args()

for h5 in args.h5:
    try:
        old = pd.read_hdf(h5)
    except:
        print '{} could not be read - reprocess'.format(h5)
        continue
        
    X = np.vstack(old.state)
    columns = [ 'f{}'.format(i) for i in range(X.shape[1]) ]

    df = pd.DataFrame(data=X, index=old.index, columns=columns)
    df.index.name='filename'

    h5name = os.path.join(args.target, os.path.basename(os.path.normpath(h5)))

    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as store:
        store.append('data', df)
