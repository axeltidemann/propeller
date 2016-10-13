# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import csv
import os

parser = argparse.ArgumentParser(description='''
Converts Ekhanei flat file structure into folders for further
processing. The folders contain symbolic links to the image files.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--csv_file',
    help='CSV file with mappings of files to categories',
    default='/home/ubuntu/images_for_irs.csv')
parser.add_argument(
    '--target',
    help='Where to put the image folders',
    default='/mnt/ekhanei/images')
args = parser.parse_args()

with open(args.csv_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for image, category in reader:

        folder = os.path.join(args.target, category)
        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            os.symlink(image, os.path.join(folder, os.path.basename(image)))
        except:
            pass # Link already exists

