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
    help='Mapping filename. If unspecified, language+nn_mapping.txt', 
    default=False)
parser.add_argument(
    '--language',
    help='What should be the output. Can be either english, thai or machine (machine is the category IDs).',
    default='english')
args = parser.parse_args()

args.mapping_filename = args.mapping_filename if args.mapping_filename else '{}_nn_mapping.txt'.format(args.language)

categories = json.load(open(args.categories))

mapping = {}

for i, h5_file in enumerate(sorted(glob.glob('{}/*.h5'.format(args.source)))):
    category, _ = os.path.splitext(os.path.basename(h5_file))
    mapping[str(i)] = category
    if args.language in ['english', 'thai']:
        mapping[str(i)] = categories[category][args.language]

with open(args.mapping_filename, 'w') as _file:
    json.dump(mapping, _file, sort_keys=True, indent=4)
