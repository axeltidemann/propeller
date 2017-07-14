# -*- coding: utf-8 -*-

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

def grapheme_map(X):
    return [ graphemes_index[c] for c in regex.findall(u'\\X', X) if c in graphemes_index ]

def create_grapheme_index(grapheme_counts, top_k):
    grapheme_counts = json.load(open(grapheme_counts))

    print 'Total number of graphemes before filtering: {}'.format(len(grapheme_counts))
    
    # Filtering out stuff:
    
    # Numbers
    for i in range(10):
        del grapheme_counts[str(i)]

    # Special characters
    unwanted = ['(', ')', '.', '*', '_', '-', '%', '@', '=', ';', '"', ':', '!', '<', '>', '&', '+', '/', u'…', '#']
    for u in unwanted:
        del grapheme_counts[u]

    # Any roman characters (we will add these later)
    for c in ascii_lowercase:
        del grapheme_counts[c]

    graphemes, _ = zip(*sorted(grapheme_counts.items(),
                               key=lambda x: x[1], reverse=True))


    ratio = 100.0*sum([ grapheme_counts[g] for g in graphemes[:top_k] ])/sum(grapheme_counts.values())
    print '{} top thai graphemes are in {}% of the filtered dataset, which totals {} graphemes'.format(top_k, ratio, len(grapheme_counts))
    
    graphemes = list(graphemes[:top_k]) + [c for c in ascii_lowercase]
    
    graphemes_index = { g: i+1 for i,g in enumerate(graphemes) } # Zero is used for padding.

    return graphemes_index

def save_text(category):

    titles = pd.read_csv(category,
                         header=None, names=['ad_id', 'title', 'description'], index_col=[0],
                         converters={'title': encode, 'description': encode })


    if not len(titles):
        print 'No data in {}'.format(category)
        return False
    
    index = []
    encodings = []
    for row in titles.itertuples():
        encoded = grapheme_map(row.title)

        if args.with_description:
            encoded += [ graphemes_index[' '] ] + grapheme_map(row.description)

        # Filter out words with latin characters less than 3?
            
        if len(encoded) > 1:
            index.append(row.Index)
            encodings.append(encoded)

    encoding_lengths = map(len, encodings)
    stats = [ np.mean(encoding_lengths, axis=0), np.median(encoding_lengths, axis=0), np.std(encoding_lengths, axis=0) ]
    
    encodings = sequence.pad_sequences(encodings, maxlen=args.seq_len, truncating='post', padding='post')
    encodings = np.vstack(encodings)

    columns = [ 'c{}'.format(i) for i in range(args.seq_len) ]
    
    h5name = '{}{}.h5'.format(args.store_location, os.path.splitext(os.path.basename(category))[0])
    
    with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as out_store:
        data = pd.DataFrame(data=encodings, index=index, columns=columns)
        out_store.append('data', data)

    print '{}: mean: {}, median: {}, std: {}'.format(category, stats[0], stats[1], stats[2])

    return encoding_lengths
        
        
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
        help='Top K graphemes to use, based on counts of the graphemes - excluding latin characters',
        type=int,
        default=200) # Top 200 graphemes account for 96.1% of the data
    parser.add_argument(
        '--with_description',
        help='Includes the description as well as the title',
        action='store_true')
    args = parser.parse_args()

    graphemes_index = create_grapheme_index(args.grapheme_counts, args.top_k)
    
    pool = mp.Pool()
    results = pool.map(save_text, args.csv)
    results = filter(lambda x: isinstance(x, list), results)
    flat_results = [ item for sublist in results for item in sublist ]
    
    print 'Graphemes used for encoding:',
    for g in sorted(graphemes_index.keys()):
        print g,
    print ''

    print 'Mean: {}, median: {}, std: {} of encodings before clipping/padding.'.format(np.mean(flat_results),
                                                                                       np.median(flat_results),
                                                                                       np.std(flat_results))
