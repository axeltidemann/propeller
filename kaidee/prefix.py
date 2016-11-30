import os
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='''
Prefixes image paths to ad id->image files mapping.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'h5',
    help='HDF5 file where keys correspond to categories')
parser.add_argument(
    '--mapping_filename',
    help='Filename of the HDF5 file', 
    default='mapping.h5')
parser.add_argument(
    '--prefix',
    help='Where the images are', 
    default='/home/ubuntu/workspace/downloads')
args = parser.parse_args()

in_store = pd.HDFStore(args.h5, mode='r')

with pd.HDFStore(args.mapping_filename, mode='w', complevel=9, complib='blosc') as out_store:
    for key in in_store.keys():
        data = in_store[key]
        ABORT.
