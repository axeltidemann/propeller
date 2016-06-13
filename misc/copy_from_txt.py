# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''Copies files specfied in text files, puts them in the target folder'''

import argparse
import shutil
import os
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='Folder with plain text file with absolute file paths')
parser.add_argument(
    'target',
    help='Target folder')
args = parser.parse_args()

for filefile in glob.glob('{}/*.txt'.format(args.source)):
    print 'Copying files specified in {}...'.format(filefile)
    dirname = os.path.basename(filefile)
    destination = '{}/{}'.format(args.target, dirname.replace('.txt',''))

    if not os.path.exists(destination):
        os.makedirs(destination)

    with open(filefile) as _file:
        for line in _file:
            shutil.copy(line.strip(), destination)
