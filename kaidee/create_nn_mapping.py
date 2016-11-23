# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Reads the statistics JSON file, maps the output of the categories trained
on the network to the human readable files. The HDF5 files must have the
same name as the category indices.
'''

import argparse
import glob
import os
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='Folder with Inception states used for training')
parser.add_argument(
    '--categories',
    help='JSON file with categories',
    default='categories.json')
parser.add_argument(
    '--mapping_filename',
    help='Mapping filename', 
    default='nn_mapping.txt')
parser.add_argument(
    '--machine',
    help='Maps to categoy IDs instead of human readable text',
    action='store_true')
args = parser.parse_args()

categories = json.load(open(args.categories))

mapping = {}

for i, h5_file in enumerate(sorted(glob.glob('{}/*.h5'.format(args.source)))):
    category = os.path.basename(h5_file[:h5_file.find('.')])
    mapping[str(i)] = category if args.machine else categories[category]['name']

with open(args.mapping_filename, 'w') as _file:
    json.dump(mapping, _file, sort_keys=True, indent=4)
