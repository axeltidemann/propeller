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

from training_data import states
from utils import load_graph

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'model',
    help='Path to trained model')
parser.add_argument(
    'data_folder',
    help='Folder with Inception states')
parser.add_argument(
    '--train_ratio',
    help='Train ratio',
    type=float,
    default=.8)
parser.add_argument(
    '--validation_ratio',
    help='Validation ratio',
    type=float,
    default=.1)
parser.add_argument(
    '--test_ratio',
    help='Test ratio',
    type=float,
    default=.1)
parser.add_argument(
    '--top_k',
    help='How many to consider',
    type=int,
    default=3)
args = parser.parse_args()

with tf.Session() as sess:
    _,_,test = states(args.data_folder, args.train_ratio, args.validation_ratio, args.test_ratio)
    load_graph(args.model)
    
    transfer_predictor = sess.graph.get_tensor_by_name('output:0')

    avg_accuracy = []
    avg_top_k_accuracy = []
    
    for category, data in test.iteritems():
        predictions = sess.run(transfer_predictor, { 'input:0': data.x })

        correct = np.argmax(predictions, axis=1) == np.argmax(data.y, axis=1)
        accuracy = np.mean(correct)

        top_k = [ np.argmax(target) in np.argsort(prediction)[-args.top_k:] for target, prediction in zip(data.y, predictions) ]
        top_k_accuracy = np.mean(top_k)

        correct_confidence = np.mean(np.max(predictions[np.where(correct)], axis=1))
        wrong_confidence = np.mean(np.max(predictions[np.where(~correct)], axis=1))

        print('Category {}, {} images: accuracy: {}, top_{} accuracy: {}, '
              'correct confidence: {}, wrong confidence: {}'
              ''.format(category, data.x.shape[0], accuracy, args.top_k,
                        top_k_accuracy, correct_confidence, wrong_confidence))

        avg_accuracy.append(accuracy)
        avg_top_k_accuracy.append(top_k_accuracy)

    print('Average accuracy across categories: {}'.format(np.mean(avg_accuracy)))
    print('Average top_{} accuracy across categories: {}'.format(args.top_k, np.mean(avg_top_k_accuracy)))
