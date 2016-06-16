# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Reads the provided Excel file, maps it into the output of the
neural network. Assumes that the HDF5 files have the same names
as the categories. Handles subcategories as well.
'''

import argparse
import glob
import os
import json

from xlrd import open_workbook

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states used for training')
parser.add_argument(
    'excel_file',
    help='Excel file with categories')
parser.add_argument(
    '--mapping_filename',
    help='Mapping filename', 
    default='mapping.txt')
parser.add_argument(
    '--human',
    help='Whether the mapping will be in human readable text, or just categories.',
    action='store_true')
args = parser.parse_args()

book = open_workbook(args.excel_file)
sheet = book.sheet_by_index(0)

ids = map(lambda x: str(int(x)), sheet.col_values(4, start_rowx=1))
names = sheet.col_values(5, start_rowx=1)

categories = dict(zip(ids, names))

mapping = {}

for i, h5_file in enumerate(sorted(glob.glob('{}/*.h5'.format(args.data_folder)))):
    category = os.path.basename(h5_file[:h5_file.find('.')])
    mapping[str(i)] = categories[category] if args.human else category

with open(args.mapping_filename, 'w') as _file:
    json.dump(mapping, _file, sort_keys=True, indent=4)
