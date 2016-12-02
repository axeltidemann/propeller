# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
from functools import partial
import json

import pandas as pd
import numpy as np
import ipdb

def combine(category):
    with pd.HDFStore(args.mapping, mode='r') as store:

        one_hot = json.load(open(args.one_hot_encoding))
        
        fname = [ h5 for h5 in args.feature_files if '{}.h5'.format(category) in h5 ]
        assert len(fname) == 1, 'This should be 1. Filename mismatch?'
        mapping = store[category]
        entries = mapping.count(axis=1)
        features = pd.read_hdf(fname[0])
        out = '{}: {} entries in mapping file, number of images processed: {}.'.format(category, len(mapping), len(features))
        out += ' mean {}, median {}, std {} images per ad.'.format(np.mean(entries), np.median(entries), np.std(entries))

        # Read in entire CSV file - maybe it will work with pandas, even? You know the category and the CSV location.
        titles = pd.read_csv('{}/{}.txt'.format(args.title_location, category), header=None, names=['ad_id', 'title'], index_col=[0])

        fail = 0
        h5name = '{}{}.h5'.format(args.store_location, category)
        with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
            for row in mapping.itertuples():
                X = []
                ad_id = row[0]
                for fname in row[1:]:
                    if pd.notnull(fname):
                        image_query = '{}{}/{}'.format(args.prefix, category, fname)
                        image = features.query('index == @image_query')
                        if len(image):
                            X.append(image)

                title = titles.query('index == @ad_id')

                if not (len(X) and len(title)):
                    print '{} lacked title or image, skipping'.format(ad_id)
                    fail += 1
                    continue

                encoded = [ one_hot[c] for c in title.values[0][0][1:-1].decode(args.decoding) ]
                title_encoded = pd.DataFrame(data=encoded, columns=['title_encoded'])

                X = np.vstack(X)
                images = pd.DataFrame(data=X, columns=features.columns)
                
                out_store.append('{}/visual'.format(ad_id), images)
                out_store.append('{}/text'.format(ad_id), title_encoded)

        out += '\n\tThere were {} failed reads'.format(fail)
        print out
        
        # Ideally, len(set(good_queries)) == len(features). Either something wrong, or there are duplicate entries in the mapping file. Investigate.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Concatenates feature vectors into HDF5 files, to be used by the rest
    of the training framework.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mapping',
        help='Mapping from ad ids to filenames')
    parser.add_argument(
        'one_hot_encoding',
        help='One hot encoding from thai and latin characters to indices')
    parser.add_argument(
        'title_location',
        help='Where the files with titles for each ad id can be found')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    parser.add_argument(
        'feature_files',
        help='HDF5 files that contain the stored features', 
        nargs='+')
    parser.add_argument(
        '--prefix',
        help='Prefix of the paths in the feature HDF5 index', 
        default='/home/ubuntu/workspace/downloads')
    parser.add_argument(
        '--decoding',
        help='How to decode the letters', 
        default='thai')
    args = parser.parse_args()

    with pd.HDFStore(args.mapping, mode='r') as store:
        keys = store.keys()

    combine(keys[0])

    assert False
        
    pool = mp.Pool()
    pool.map(combine, keys)
