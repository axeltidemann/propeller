# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import csv
import json

import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'csv',
    help='CSV file(s) with ad ids and title',
    nargs='+')
parser.add_argument(
    '--decoding',
    help='How to decode the letters', 
    default='thai')
parser.add_argument(
    '--mapping_filename',
    help='Filename of the JSON mapping file', 
    default='letters_mapping.json')
args = parser.parse_args()

letters = set([])
lengths = []
for _file in args.csv:
    print 'Opening {}'.format(_file)
    with open(_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for ad_id, title in reader:
            title = title[1:-1].decode(args.decoding, errors='ignore')
            letters = letters.union(title)
            lengths.append(len(title))
            
mapping = { c: i for i,c in enumerate(letters) }
mapping_inverse = { i: c for i,c in enumerate(letters) }

with open(args.mapping_filename, 'w') as _file:
    json.dump(mapping, _file, sort_keys=True, indent=4)
    
with open('{}_inverse'.format(args.mapping_filename), 'w') as _file:
    json.dump(mapping_inverse, _file, sort_keys=True, indent=4)
    
print 'Encoding lengths: mean: {}, median: {}, std: {}, max: {}, min: {}'.format(np.mean(lengths), np.median(lengths), np.std(lengths), max(lengths), min(lengths))
