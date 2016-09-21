# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import os

import pandas as pd
import numpy as np

from clusters import find

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Cleanses a category
    based on clusters in the data.''',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'data',
        help='Path to HDF5 file(s) with states',
        nargs='+')
    parser.add_argument(
        'target',
        help='Folder to put the cleansed HDF5 file(s), will have same name as original HDF5 file, with suffix number.')
    parser.add_argument(
        '--lower_bound',
        help='Lower level of cosine similarity',
        default=.8,
        type=float)
    parser.add_argument(
        '--upper_bound',
        help='Upper level of cosine similarity',
        default=.95,
        type=float)
    parser.add_argument(
        '--min_cluster_size',
        help='Minimum relative cluster size, these will be put in their own category',
        default=.01,
        type=float)
    parser.add_argument(
        '--include_rejected',
        action='store_true',
        help='Whether to include the rejected images')
    args = parser.parse_args()

    for h5 in args.data:
        data = pd.read_hdf(h5)
        
        clusters = find(data, args.lower_bound, args.upper_bound, args.min_cluster_size)
        
        for i, (node, edges) in enumerate(clusters.iteritems()):
            if not args.include_rejected and node == 'rejected':
                continue
            
            if node is not 'rejected':
                edges.append(node) # To include the seed
                
            h5name = os.path.join(args.target, '{}_{}'.format(os.path.basename(h5), i))
            with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as store:
                store.append('data', data.iloc[edges])
                
            print '{}: storing cluster {} as {} with {} ({}%) samples'.format(h5, i, h5name,
                                                                              len(edges), 100.*len(edges)/len(data))
