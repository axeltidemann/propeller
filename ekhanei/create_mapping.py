# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import glob
import os
import json
import csv

parser = argparse.ArgumentParser(description='''
Reads the provided file, maps it into the output of the
neural network. Assumes that the HDF5 files have the same names
as the categories.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states used for training')
parser.add_argument(
    'categories_file',
    help='File with categories')
parser.add_argument(
    '--mapping_filename',
    help='Mapping filename', 
    default='mapping.txt')
parser.add_argument(
    '--human',
    help='Whether the mapping will be in human readable text, or just category numbers.',
    action='store_true')
args = parser.parse_args()

with open(args.categories_file, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter='=')
    categories = dict(reader)

mapping = {}

for i, h5_file in enumerate(sorted(glob.glob('{}/*.h5'.format(args.data_folder)))):
    
    category = os.path.basename(h5_file[:h5_file.find('.')])
    try:
        mapping[str(i)] = categories[category] if args.human else category
    except Exception as e:
        print e

with open(args.mapping_filename, 'w') as _file:
    json.dump(mapping, _file, sort_keys=True, indent=4)
