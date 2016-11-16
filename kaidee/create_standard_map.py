# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Reads the provided Excel file and maps it into JSON 
hierarchy that will eventually evolve to become the standard.
'''

import argparse
import glob
import os
import json

from xlrd import open_workbook

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'excel_file',
    help='Excel file with categories')
parser.add_argument(
    '--categories_filename',
    help='Name of JSON file', 
    default='categories.json')
args = parser.parse_args()

book = open_workbook(args.excel_file)
sheet = book.sheet_by_index(0)

category_level = map(int, sheet.col_values(6, start_rowx=1))

top_ids = map(int, sheet.col_values(0, start_rowx=1))
top_names = sheet.col_values(1, start_rowx=1)

sub_ids = map(int, sheet.col_values(4, start_rowx=1))
sub_names = sheet.col_values(5, start_rowx=1)

categories = {}
for top_i, top_n, sub_i, sub_n, lvl in zip(top_ids, top_names, sub_ids, sub_names, category_level):
    if lvl == 1:
        categories[top_i] = {"name": top_n, "level": lvl }
    else:
        categories[sub_i] = {"name": sub_n, "level": lvl, "parent": top_i}

with open(args.categories_filename, 'w') as _file:
    json.dump(categories, _file, sort_keys=True, indent=4)
