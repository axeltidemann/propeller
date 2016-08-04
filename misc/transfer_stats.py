# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html

from __future__ import print_function
import argparse
import glob
import os
import uuid

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
import numpy as np

from training_data import states
from utils import load_graph

parser = argparse.ArgumentParser(description='''
Performs statistics on a trained model.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states')
parser.add_argument(
    'models',
    help='Models to evaluate',
    nargs='+')
parser.add_argument(
    '--experts',
    help='Whether we are evaluating experts',
    action='store_true')
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

_,_,test = states(args.data_folder, args.train_ratio, args.validation_ratio, args.test_ratio)

if args.experts:
    results = {}
    avg_accuracy = []
    with tf.Session() as sess:
        for category, data in test.iteritems():
            results = {}

            for model in args.models:
                load_graph(model)

                stripped = model[20:]
                h5 = stripped[:stripped.find('_')]# MESSY.
                
                transfer_predictor = sess.graph.get_tensor_by_name('{}output:0'.format(h5))
                predictions = sess.run(transfer_predictor, { '{}input:0'.format(h5): data.x })
                results[model] = predictions

            keys = results.keys()

            expert = np.argmax([ category in model for model in keys ])

            all_predictions = np.hstack([ results[model] for model in keys ])

            correct = np.argmax(all_predictions, axis=1) == expert
            accuracy = np.mean(correct)

            print('{} accuracy: {}'.format(category, accuracy))
            avg_accuracy.append(accuracy)

    print('Average accuracy across categories: {}'.format(np.mean(avg_accuracy)))

else:
    for model in args.models:
        with tf.Session() as sess:
            print('Evaluating {}'.format(model))

            load_graph(model)

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

        tf.reset_default_graph()
