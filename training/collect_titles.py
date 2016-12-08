# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
import json

import pandas as pd
import numpy as np
from keras.preprocessing import sequence

def save_titles(category):
    print 'Processing {}'.format(category)

    one_hot = json.load(open(args.one_hot_encoding))

    titles = pd.read_csv(category,
                         header=None, names=['ad_id', 'title'], index_col=[0],
                         converters={'title': lambda x: x[1:-1].decode(args.decoding, errors='ignore') })

    index = []
    encodings = []
    for row in titles.itertuples():
        title_encoded = [ one_hot[c] for c in row.title ]
        if len(title_encoded):
            index.append(row.Index)
            encodings.append(title_encoded)
            
    encodings = sequence.pad_sequences(encodings, maxlen=args.seq_len, truncating='post', padding='post')
    encodings = np.vstack(encodings)

    columns = [ 'c{}'.format(i) for i in range(args.seq_len) ]
    
    h5name = '{}{}.h5'.format(args.store_location, os.path.splitext(os.path.basename(category))[0])
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
        data = pd.DataFrame(data=encodings, index=index, columns=columns)
        out_store.append('data', data)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Reads files with ad ids and titles, puts them into HDF5 files.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'csv',
        help='CSV files that contain the titles', 
        nargs='+')
    parser.add_argument(
        'one_hot_encoding',
        help='One hot encoding from thai and latin characters to indices')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    parser.add_argument(
        '--decoding',
        help='How to decode the letters', 
        default='thai')
    parser.add_argument(
        '--seq_len',
        help='Length of encoded sequence',
        type=int,
        default=100)
    args = parser.parse_args()

    pool = mp.Pool()
    pool.map(save_titles, args.csv)
