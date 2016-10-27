# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html

from __future__ import print_function
import argparse
import glob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../misc')))
from itertools import chain

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from training_data import states
from utils import load_graph, pretty_float

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def evaluate(model, h5_files, top_k):
    
    figure = plt.figure()
    
    colormap = plt.cm.spectral
    colors = [colormap(i) for i in np.linspace(0, 1, len(h5_files))]
    colors = chain(*zip(colors, colors))
    plt.gca().set_color_cycle(colors)
    
    with tf.Session() as sess:
        print('Evaluating {}'.format(model))

        print('NOTE: ALL NUMBER TRIPLES ARE ON THE FORM (mean, median, standard deviation)')
        
        load_graph(model)

        transfer_predictor = sess.graph.get_tensor_by_name('output:0')

        all_accuracies = []
        all_top_k_accuracy = []

        for i, h5 in enumerate(h5_files):

            data, _ = states([ h5 ])

            X,_ = data

            Y = np.zeros((len(X), len(h5_files)))
            Y[:,i] = 1

            predictions = sess.run(transfer_predictor, { 'input:0': X })

            correct = np.argmax(predictions, axis=1) == np.argmax(Y, axis=1)
            accuracy = np.mean(correct)

            top_k_accuracy = np.mean([ np.argmax(target) in np.argsort(prediction)[-top_k:]
                                       for target, prediction in zip(Y, predictions) ])

            correct_scores = np.max(predictions[np.where(correct)], axis=1)
            correct_x = np.linspace(0,1, num=len(correct_scores))
            correct_confidence = np.mean(correct_scores)
            correct_confidence_median = np.median(np.max(predictions[np.where(correct)], axis=1))
            correct_confidence_std = np.std(np.max(predictions[np.where(correct)], axis=1))

            plt.plot(correct_x, sorted(correct_scores), label='{} correct'.format(os.path.basename(h5)))

            wrong_scores = np.max(predictions[np.where(~correct)], axis=1)
            wrong_x = np.linspace(0,1, num=len(wrong_scores))
            wrong_confidence = np.mean(wrong_scores)
            wrong_confidence_median = np.median(np.max(predictions[np.where(~correct)], axis=1))
            wrong_confidence_std = np.std(np.max(predictions[np.where(~correct)], axis=1))

            plt.plot(wrong_x, sorted(wrong_scores), '--', label='{} wrong'.format(os.path.basename(h5)))
            
            print('Category {}, {} images. \t accuracy: {} top_{} accuracy: {} '
                  'correct confidence: {}, {}, {} wrong confidence: {}, {}, {}'
                  ''.format(os.path.basename(h5), len(X), pretty_float(accuracy), top_k,
                            pretty_float(top_k_accuracy),
                            pretty_float(correct_confidence),
                            pretty_float(correct_confidence_median),
                            pretty_float(correct_confidence_std),
                            pretty_float(wrong_confidence),
                            pretty_float(wrong_confidence_median),
                            pretty_float(wrong_confidence_std)))

            all_accuracies.append(accuracy)
            all_top_k_accuracy.append(top_k_accuracy)


        mean_accuracy = np.mean(all_accuracies)
        top_k_accuracy = np.mean(all_top_k_accuracy)
        print('Average accuracy across categories: {}'.format(pretty_float(mean_accuracy)))
        print('Average top_{} accuracy across categories: {}'.format(top_k, pretty_float(top_k_accuracy)))

    tf.reset_default_graph()

    plt.ylim([0,1])

    return figure, mean_accuracy, top_k_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
    Performs statistics on a trained model. Does not run on GPU.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'test_folder',
        help='Folder with Inception states for testing')
    parser.add_argument(
        'models',
        help='Models to evaluate',
        nargs='+')
    parser.add_argument(
        '--top_k',
        help='How many to consider',
        type=int,
        default=3)
    args = parser.parse_args()

    h5_files = sorted(glob.glob('{}/*'.format(args.test_folder)))

    for model in args.models:
        evaluate(model, h5_files, args.top_k)
