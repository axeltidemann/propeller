# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np


def ids_to_features(category, in_store, feature_files):
    with pd.HDFStore(in_store, mode='r') as store:
        
        fname = [ h5 for h5 in feature_files if '{}.h5'.format(category) in h5 ]
        assert len(fname) == 1, 'This should be 1. Filename mismatch?'
        mapping = store[category]
        entries = mapping.count(axis=1)
        features = pd.read_hdf(fname[0])
        out = '{}: {} entries in mapping file, number of images processed: {}.'.format(category, len(mapping), len(features))
        out += ' mean {}, median {}, std {} images per ad.'.format(np.mean(entries), np.median(entries), np.std(entries))

        fail = 0
        good_queries = []
        zeros = np.zeros((1, features.shape[1]))

        # 2048 columns works (actually 4096 as well), but that is not scalable. Instead, images for ads are put in separate dataframes.

        h5name = '{}{}.h5'.format(args.store_location, category)
        with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
            for column in mapping.columns:
                X = []                
                for fname in mapping[column]:
                    image = zeros
                    if pd.notnull(fname):
                        my_query = '{}{}/{}'.format(args.prefix, category, fname)
                        needle = features.query('index == @my_query') # This should fail very rarely, due to images not being processed
                        if len(needle):
                            image = needle
                            good_queries.append(my_query)
                        else:
                            fail += 1
                    X.append(image)

                X = np.vstack(X)

                df = pd.DataFrame(data=X, index=mapping.index, columns=features.columns)
                df.index.name='ad_id'
                out_store.append(column, df)

        out += '\n\tThere were {} failed reads, {} successful ones. {} unique filename queries.'.format(fail, len(good_queries), len(set(good_queries)))
        print out
        
        # Ideally, len(set(good_queries)) == len(features). Either something wrong, or there are duplicate entries in the mapping file. Investigate.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Concatenates feature vectors into HDF5 files, to be used by the rest
    of the training framework.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'mapping',
        help='HDF5 file with ad ids and image names')
    parser.add_argument(
        'feature_files',
        help='HDF5 files that contain the stored features', 
        nargs='+')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    parser.add_argument(
        '--prefix',
        help='Prefix of the paths in the feature HDF5 index', 
        default='/home/ubuntu/workspace/downloads')
    args = parser.parse_args()

    with pd.HDFStore(args.mapping, mode='r') as store:
        keys = store.keys()

    par_fill = partial(ids_to_features, in_store=args.mapping, feature_files=args.feature_files)
    pool = mp.Pool()
    pool.map(par_fill, keys)
