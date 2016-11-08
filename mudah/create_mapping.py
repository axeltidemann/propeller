# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import glob
import os
import json

parser = argparse.ArgumentParser(description='''
Maps category IDs to output of neural network.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='HDF5 files',
    nargs='+')
parser.add_argument(
    '--filename',
    help='Name of mapping file',
    default='mapping.txt')
args = parser.parse_args()

categories = sorted([ os.path.basename(h5).replace('.h5','') for h5 in args.source ])

mapping = { i: category for i, category in enumerate(categories) }

with open(args.filename, 'w') as _file:
    json.dump(mapping, _file, indent=4)
