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
    '--model',
    help='Path to trained model',
    default=False)
parser.add_argument(
    '--experts',
    help='Path to a folder with trained experts',
    default=False)
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

assert args.model or args.experts, 'You must specify either a trained model or a folder with experts'

_,_,test = states(args.data_folder, args.train_ratio, args.validation_ratio, args.test_ratio)

if args.experts:
    results = {}
    avg_accuracy = []
    for category, data in test.iteritems():
        print('Evaluating category {}'.format(category))
        results = {}

        for model in glob.glob('{}/*.pb'.format(args.experts)):
            print('Evaluating model {}'.format(model))
            load_graph(model)

            with tf.Session() as sess:

                transfer_predictor = sess.graph.get_tensor_by_name('output:0')
                predictions = sess.run(transfer_predictor, { 'input:0': data.x })

                if category in model:
                    print('This is the expert: {}'.format(predictions[:5]))
                else:
                    print(predictions[:5])
                yes = predictions[:,0]
                yes.shape = (predictions.shape[0], 1)
                results[model] = yes

            tf.reset_default_graph()

        keys = results.keys()

        expert = np.argmax([ category in model for model in keys ])

        all_predictions = np.hstack([ results[model] for model in keys ])

        print(all_predictions.shape)
        print(all_predictions)

        correct = np.argmax(all_predictions, axis=1) == expert
        accuracy = np.mean(correct)

        print('{} accuracy: {}'.format(category, accuracy))
        avg_accuracy.append(accuracy)

        break

    print('Average accuracy across categories: {}'.format(np.mean(avg_accuracy)))


if args.model:
    with tf.Session() as sess:

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
