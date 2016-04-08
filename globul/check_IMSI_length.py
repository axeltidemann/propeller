'''
Quick check for maximum IMSI length in the Bulgaria dataset.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import defaultdict

import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--path',
    help='Path to the HDF5 file')
parser.add_argument(
    '--frame_table',
    help='Which frame table to use',
    default='data')
parser.add_argument(
    '--chunk_size',
    help='Chunk size to iterate over HDF5 file',
    type=int,
    default=50000) 

args = parser.parse_args()

with pd.get_store(args.path) as store:
    data = store.select(args.frame_table, chunksize=args.chunk_size)
    counter = defaultdict(int)
    
    for chunk in data:
        for IMSI in chunk.IMSI.unique():
            counter[len(IMSI)] += 1
    print counter
