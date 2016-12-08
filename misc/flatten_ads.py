# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
from keras.preprocessing import sequence

def flatten(h5):
    X = []
    with pd.HDFStore(h5, mode='r') as in_store:
        keys = in_store.keys()

        ad_ids = list(set([ key.split('/')[1] for key in keys ]))

        text = []
        images = []
        index = []
        
        for ad_id in ad_ids:
            try:
                ad_text = in_store['{}/text'.format(ad_id)]
                ad_text = np.squeeze(ad_text)

                ad_images = in_store['{}/visual'.format(ad_id)]

                for i, image in enumerate(ad_images.values):
                    text.append(ad_text)
                    images.append(image)
                    index.append('{}_{}'.format(ad_id, i))

            except Exception as e:
                print '{}: {}'.format(h5, e)

        text = sequence.pad_sequences(text, maxlen=args.seq_len, truncating='post', padding='post')
        text = np.vstack(text)
        images = np.vstack(images)

        assert text.shape[0] == images.shape[0], 'These two must be similar length.'
        
    h5name = os.path.join(args.store_location, os.path.basename(h5))
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
        text_columns = [ 'c{}'.format(i) for i in range(args.seq_len) ]
        text_df = pd.DataFrame(data=text, index=index, columns=text_columns)

        out_store.append('text', text_df)
        
        image_df = pd.DataFrame(data=images, index=index, columns=ad_images.columns)
        out_store.append('visual', image_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Flattens HDF5 files with ad titles and image features.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'ad_files',
        help='HDF5 files that contain both title and visual features.', 
        nargs='+')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    parser.add_argument(
        '--seq_len',
        help='How many characters to store in the flattened file',
        type=int,
        default=100)
    args = parser.parse_args()

    pool = mp.Pool()
    pool.map(flatten, args.ad_files)
