# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Reads the provided Excel file, maps it into the output of the
neural network. Assumes that the HDF5 files have the same names
as the categories. Handles subcategories as well.
'''

import argparse
import glob

from xlrd import open_workbook

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states used for training')
parser.add_argument(
    'excel_file',
    help='Excel file with categories')
args = parser.parse_args()

book = open_workbook(args.excel_file)
sheet = book.sheet_by_index(0)

ids = map(int, sheet.co_values(4, start_rowx=1))
names = sheet.col_values(5, start_rowx=1)

mapping = {}

for h5_file in sorted(glob.glob('{}/*.h5'.format(args.data_folder))):
    category = h5_file[h5_file.find('.')]
    print category
