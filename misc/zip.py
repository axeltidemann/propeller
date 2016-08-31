# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import zipfile
import zlib
compression = zipfile.ZIP_DEFLATED

import pandas as pd

parser = argparse.ArgumentParser(description='''
Zips the image files present in a HDF5 file.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='HDF5 file(s) with images.',
    nargs='+')
parser.add_argument(
    '--filename',
    help='Name of zip file',
    default='images.zip')
parser.add_argument(
    '--table',
    help='HDF5 table to read',
    default='data')
args = parser.parse_args()

with zipfile.ZipFile(args.filename, mode='w', allowZip64=True) as zf:

    for h5 in args.source:
        print 'Zipping {}'.format(h5)
        data = pd.read_hdf(h5, args.table)
        for _file in data.index:
            zf.write(_file, compress_type=compression)
