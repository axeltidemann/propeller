# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'csv',
    help='CSV file(s) with maps from ad ID to images. Each csv filename will be stripped of its extension and used as key.',
    nargs='+')
parser.add_argument(
    '--mapping_filename',
    help='Filename of the HDF5 file', 
    default='mapping.h5')
args = parser.parse_args()

with pd.HDFStore(args.mapping_filename, mode='w', complevel=9, complib='blosc') as store:
    for csv in args.csv:
        key, _ = os.path.splitext(os.path.basename(csv))
        data = pd.read_csv(csv, header=None, names=['ad_id'] + [ 'image{}'.format(i) for i in range(9) ], index_col=[0],
                           converters={ i: lambda x: x.strip()[1:-1] if len(x) else np.nan for i in range(1,10) })
        store.append(key, data)
