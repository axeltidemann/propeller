# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import csv
import json
from collections import Counter
import multiprocessing as mp

import numpy as np
import regex

parser = argparse.ArgumentParser(description='''
    Reads files with ad ids, titles and descriptions, and finds
    all the unique graphemes in them. Saves counts and inverse mappings as well.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'csv',
    help='CSV file(s) with ad ids, title and description',
    nargs='+')
parser.add_argument(
    '--mapping_filename',
    help='Filename of the JSON mapping file. _inverse and _counter will be appended to this filename.', 
    default='grapheme_mapping.json')
args = parser.parse_args()

def encode(x):
    return unicode(x[1:-1], 'utf-8').lower()

def traverse(_file):

    letters = set([])
    lengths = []
    grapheme_counter = Counter()

    print 'Opening {}'.format(_file)
    with open(_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for ad_id, title, description in reader:
            title = encode(title)
            description = encode(description)
            graphemes = regex.findall(u'\\X', title+description)
            letters = letters.union(graphemes)
            grapheme_counter.update(graphemes)
            lengths.append(len(graphemes))

    return letters, lengths, grapheme_counter


pool = mp.Pool()
results = pool.map(traverse, args.csv)

letters = set([])
lengths = []
grapheme_counter = Counter()

for lt, ln, gc in results:
    letters.union(lt)
    lengths.extend(ln)
    grapheme_counter.update(gc)
            
mapping = { c: i for i,c in enumerate(letters) }
mapping_inverse = { i: c for i,c in enumerate(letters) }

with open(args.mapping_filename, 'w') as _file:
    json.dump(mapping, _file, sort_keys=True, indent=4)
    
with open('{}_inverse'.format(args.mapping_filename), 'w') as _file:
    json.dump(mapping_inverse, _file, sort_keys=True, indent=4)

with open('{}_counter'.format(args.mapping_filename), 'w') as _file:
    json.dump(grapheme_counter, _file, sort_keys=True, indent=4)
    
print 'Encoding lengths: mean: {}, median: {}, std: {}, max: {}, min: {}'.format(np.mean(lengths), np.median(lengths), np.std(lengths), max(lengths), min(lengths))
