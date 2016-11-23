# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Reads the canonical JSON file that has everything in it (in list form), 
convert to the JSON hierarchy. This is used for display of statistics and reports,
not for output of the neural network.
'''

import argparse
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'json_file',
    help='JSON file with categories')
parser.add_argument(
    '--out_filename',
    help='Name of JSON file', 
    default='categories.json')
args = parser.parse_args()

categories = {}
for cat in json.load(open(args.json_file), encoding='iso8859_11', strict=False):
    if cat['parent_id']:
        categories[cat['cate_id']] = {'name': cat['title_eng'], 'parent': cat['parent_id']}
    else:
        categories[cat['cate_id']] = {'name': cat['title_eng']}

with open(args.out_filename, 'w') as _file:
    json.dump(categories, _file, sort_keys=True, indent=4)
