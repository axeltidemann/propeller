# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='''
Reads an HDF5 file with mappings of ad IDs to images, calculates
plots bar charts showing how many images are in each.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'mapping',
    help='HDF5 file with ad ids and images')
parser.add_argument(
    '--out_file',
    help='Filename of the HDF5 file', 
    default='bar.h5')
args = parser.parse_args()

with pd.HDFStore(args.mapping, mode='r') as store:
    for category in store.keys():
        data = store[category]
        entries = data.count(axis=1)
        print '{}: {} entries. number of images: mean {}, median {}, std {}'.format(category, len(data), np.mean(entries), np.median(entries), np.std(entries))
