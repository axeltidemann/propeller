# Copyright 2017 Telenor ASA, Author: Axel Tidemann

import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'h5',
    nargs='+',
    help='HDF5 files that need change in path')
parser.add_argument(
    '--prefix',
    default='/mnt/kaidee/10K_ads_with_first_image/images/')
parser.add_argument(
    '--suffix',
    default='.jpeg')
args = parser.parse_args()

for filename in args.h5:
    with pd.HDFStore(filename) as store:

        category,_ = os.path.splitext(os.path.basename(filename))
        images = store['visual']

        images.index = [ '{}{}/{}{}'.format(args.prefix, category, i, args.suffix) for i in images.index ]
        
        del store['visual']
        
        store.append('visual', images)
        
