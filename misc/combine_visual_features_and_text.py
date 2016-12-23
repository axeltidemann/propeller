# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
import json

import pandas as pd
import numpy as np

def combine(category):
    with pd.HDFStore(args.mapping, mode='r') as store:

        mapping = store[category]
        entries = mapping.count(axis=1)
        features = pd.read_hdf('{}{}.h5'.format(args.visual_files, category))
        
        out = '{}: {} entries in mapping file, number of images processed: {}.'.format(category, len(mapping), len(features))
        out += ' mean {}, median {}, std {} images per ad.'.format(np.mean(entries), np.median(entries), np.std(entries))

        text = pd.read_hdf('{}{}.h5'.format(args.text_files, category))

        present = np.mean([ row[0] in text.index for row in mapping.itertuples() ])

        if present < .5:
            print '{}: {}% data, skipping'.format(category, 100*present)
            return False

        fail = 0
        h5name = '{}{}.h5'.format(args.store_location, category)

        all_text = []
        all_images = []
        all_index = []

        for row in mapping.itertuples():
            images = []
            ad_id = row[0]
            for fname in row[1:]: # Maybe try out with just first picture also
                if pd.notnull(fname):
                    image_query = '{}{}/{}'.format(args.prefix, category, fname)
                    ad_image = features.query('index == @image_query')
                    if len(ad_image):
                        images.append(ad_image)

            ad_text = text.query('index == @ad_id')

            if len(images) and len(ad_text):
                for i, image in enumerate(images):
                    all_images.append(image)
                    all_text.append(ad_text)
                    all_index.append('{}_{}'.format(ad_id, i))

            else:
                print '{}: {} lacked title or image, skipping'.format(category, ad_id)
                fail += 1

        with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
            text_df = pd.DataFrame(data=np.vstack(all_text), index=all_index, columns=text.columns)
            out_store.append('text', text_df)
        
            image_df = pd.DataFrame(data=np.vstack(all_images), index=all_index, columns=features.columns)
            out_store.append('visual', image_df)

        out += '\n\tThere were {} failed reads'.format(fail)
        print out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Concatenates feature vectors into HDF5 files, to be used by the rest
    of the training framework.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mapping',
        help='Mapping file from ad id to image filenames')
    parser.add_argument(
        'visual_files',
        help='Folder with HDF5 files that contain the visual features')
    parser.add_argument(
        'text_files',
        help='HDF5 files with encoded text for each ad id')
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

    pool = mp.Pool()
    pool.map(combine, keys)
