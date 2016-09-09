# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import multiprocessing as mp
from random import shuffle
import os
from functools import partial
import time

parser = argparse.ArgumentParser(description='''Creates square images from a folder of folders.
The image is resized so the smallest dimension will be of the size specified. It will then
crop the center to achieve a square. For each folder, create a corresponding folder with the
squared images in the target folder.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'source',
    help='Where to find a folder of folders with images.')
parser.add_argument(
    'target',
    help='Where to put the resulting folders with square images.')
parser.add_argument(
    '--limit',
    help='The maximum number of images.',
    default=10000,
    type=int)
parser.add_argument(
    '--dimension',
    help='Dimension of the square image',
    default=299,
    type=int)
parser.add_argument(
    '--randomize',
    help='Whether to randomize the file list, in case of more than 10 000 images.',
    default=True,
    type=bool)
args = parser.parse_args()

def convert(target, limit, dimension, randomize, source):
    print 'Processing folder {}'.format(source)
    images = os.listdir(source)

    if randomize:
        shuffle(images)

    images = images[:limit]

    target_folder = os.path.join(target, os.path.basename(source))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for img in images:
        in_file = os.path.join(source, img)
        out_file = os.path.join(target_folder, img)
        os.system('convert {0} -resize "{1}x{1}^" -gravity center -crop {1}x{1}+0+0 +repage {2}'.format(in_file, dimension, out_file))
    

unsquared = [ os.path.join(args.source, folder) for folder in os.listdir(args.source) ]

par_convert = partial(convert, args.target, args.limit, args.dimension, args.randomize)

pool = mp.Pool()
pool.map(par_convert, unsquared)
