# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import multiprocessing as mp
import json
from string import ascii_lowercase

import pandas as pd
import numpy as np
from keras.preprocessing import sequence
import regex

def encode(x):
    return unicode(x[1:-1], 'utf-8').lower()

def save_text(category):

    if args.with_description:
        titles = pd.read_csv(category,
                             header=None, names=['ad_id', 'title', 'description'], index_col=[0],
                             converters={'title': encode, 'description': encode })
    else:
        titles = pd.read_csv(category,
                             header=None, names=['ad_id', 'title'], index_col=[0],
                             converters={'title': encode })
        

    index = []
    encodings = []
    for row in titles.itertuples():
        
        encoded = [ graphemes_index[c] for c in regex.findall(u'\\X', row.title) if c in graphemes_index ]

        if args.with_description:
            encoded += [ graphemes_index[' '] ] + [ graphemes_index[c] for c in regex.findall(u'\\X', row.description) if c in graphemes_index ]

        if len(encoded) > 1:
            index.append(row.Index)
            encodings.append(encoded)

    stats = [ np.mean(encoded, axis=0), np.median(encoded, axis=0), np.std(encoded, axis=0) ]
    
    encodings = sequence.pad_sequences(encodings, maxlen=args.seq_len, truncating='post', padding='post')
    encodings = np.vstack(encodings)

    columns = [ 'c{}'.format(i) for i in range(args.seq_len) ]
    
    h5name = '{}{}.h5'.format(args.store_location, os.path.splitext(os.path.basename(category))[0])
    
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
        data = pd.DataFrame(data=encodings, index=index, columns=columns)
        out_store.append('data', data)

    print '{}: mean: {}, median: {}, std: {}'.format(category, stats[0], stats[1], stats[2])

    return stats
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Reads files with ad ids, titles and descriptions, puts them into HDF5 files.
    The title and description will be concatenated if --with_description is given.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'csv',
        help='CSV files that contain the titles and descriptions', 
        nargs='+')
    parser.add_argument(
        'grapheme_counts',
        help='Distribution of the graphemes found in the data')
    parser.add_argument(
        'store_location',
        help='Where to store the HDF5 files')
    parser.add_argument(
        '--seq_len',
        help='Length of encoded sequence',
        type=int,
        default=100)
    parser.add_argument(
        '--top_k',
        help='Top K graphemes to use, based on counts of the graphemes',
        type=int,
        default=200) # Top 200 graphemes account for 95% of the data
    parser.add_argument(
        '--with_description',
        help='Includes the description as well as the title',
        action='store_true')
    args = parser.parse_args()

    grapheme_counts = json.load(open(args.grapheme_counts))

    # Filtering out stuff:
    
    # Numbers
    for i in range(10):
        del grapheme_counts[str(i)]

    # Special characters
    unwanted = ['(', ')', '.', '*', '_', '-', '%', '@', '=', ';', '"', ':', '!', '<', '>', '&', '+', '/']
    for u in unwanted:
        del grapheme_counts[u]

    # Any roman characters (we will add these later)
    for c in ascii_lowercase:
        del grapheme_counts[c]

    graphemes, _ = zip(*sorted(grapheme_counts.items(),
                               key=lambda x: x[1], reverse=True))

    graphemes = list(graphemes[:args.top_k]) + [c for c in ascii_lowercase]
    
    graphemes_index = { g: i+1 for i,g in enumerate(graphemes) } # Zero is used for padding.

    pool = mp.Pool()
    results = pool.map(save_text, args.csv)

    print 'Graphemes used for encoding:',
    for g in graphemes:
        print g,
    print ''
    
    print 'Mean of mean: {}, median: {}, std: {} of encodings before clipping/padding.'.format(np.mean([ x[0] for x in results]),
                                                                                               np.median([ x[1] for x in results]),
                                                                                               np.std([ x[2] for x in results]))
