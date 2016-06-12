# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html

"""
Creates a mass insertion proto.txt from a learned model file by running it on 
all HDF5 files in the folder.
"""

from __future__ import print_function
import argparse
import glob
import os
import uuid

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'model',
    help='Path to trained model')
parser.add_argument(
    'data_folder',
    help='Folder with Inception states')
parser.add_argument(
    'prefix',
    help='Key prefix')
parser.add_argument(
    '--moderate_threshold',
    help='Images with confidence lower than this threshold will be flagged for moderation',
    type=float,
    default=.7)
parser.add_argument(
    '--path',
    help='Where to put proto.txt',
    default='.')

args = parser.parse_args()

def load_graph(path):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with gfile.FastGFile(path, 'r') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    load_graph(args.model)
    transfer_predictor = sess.graph.get_tensor_by_name('output:0')

    with open('{}/proto.txt'.format(args.path),'w') as _file:

        files = sorted(glob.glob('{}/*.h5'.format(args.data_folder)))
        
        for h5_file in files:
            data = pd.read_hdf(h5_file, 'data')

            print('Processing category {}'.format(os.path.basename(h5_file.replace('.h5', ''))))
            predictions = sess.run(transfer_predictor, {'input:0': np.vstack(data.state) })

            for location, row in zip(data.index, predictions):
                identifier = '{}:image:{}'.format(args.prefix, os.path.basename(location))
                index = row.argsort()[-1]
                category = os.path.basename(files[index].replace('.h5', ''))
                confidence = row[index]
                tag = 'tag:{}'.format(category)
                print('HMSET {} location {} {} {}'.format(identifier, location, tag, confidence), file=_file)
                print('ZADD {}:{} {} {}'.format(args.prefix, tag, confidence, identifier), file=_file)

                if confidence < args.moderate_threshold:
                    print('HMSET {}:moderate:images {} {}'.format(args.prefix, identifier, tag), file=_file)

    
