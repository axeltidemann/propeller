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
import json

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
import numpy as np
import ipdb 

import plotly.graph_objs as go


from training_data import states
from utils import load_graph, pretty_float as pf

def evaluate(model, h5_files, top_k, categories=None):
    h5_files = sorted(h5_files)

    plotly_data = []
    
    with tf.Session() as sess:
        print('Evaluating {}'.format(model))

        print('NOTE: ALL NUMBER TRIPLES ARE ON THE FORM (mean, median, standard deviation)')
        
        load_graph(model)

        transfer_predictor = sess.graph.get_tensor_by_name('output:0')

        all_accuracies = []
        all_top_k_accuracy = []
        stats = []

        for target, h5 in enumerate(h5_files):

            data = pd.read_hdf(h5)

            predictions = sess.run(transfer_predictor, { 'input:0': data })

            correct = np.argmax(predictions, axis=1) == target
            accuracy = np.mean(correct)

            top_k_accuracy = np.mean([ target in np.argsort(prediction)[-top_k:]
                                       for prediction in predictions ])

            correct_scores = np.max(predictions[correct], axis=1)
            correct_x = np.linspace(0,1, num=len(correct_scores))
            correct_confidence = np.mean(correct_scores)
            correct_confidence_median = np.median(np.max(predictions[correct], axis=1))
            correct_confidence_std = np.std(np.max(predictions[correct], axis=1))

            category_i = os.path.basename(h5).replace('.h5','')
            category = categories[category_i]['name'] if categories else category_i

            plotly_data.append(go.Scatter(
                x=correct_x,
                y=sorted(correct_scores),
                mode='lines',
                name=category,
                hoverinfo='name+y',
                text=[ json.dumps({ 'path': path, 'prediction': category })
                       for path in data.index[correct]]))
            
            wrong_scores = np.max(predictions[~correct], axis=1)
            wrong_categories_i = np.argmax(predictions[~correct], axis=1)
            wrong_x = np.linspace(0,1, num=len(wrong_scores))
            wrong_confidence = np.mean(wrong_scores)
            wrong_confidence_median = np.median(np.max(predictions[~correct], axis=1))
            wrong_confidence_std = np.std(np.max(predictions[~correct], axis=1))

            wrong_categories = []

            for i in wrong_categories_i:
                wrong_i = os.path.basename(h5_files[i]).replace('.h5','')
                wrong_categories.append(categories[wrong_i]['name'] if categories else wrong_i)

            plotly_data.append(go.Scatter(
                x=wrong_x,
                y=sorted(wrong_scores),
                mode='markers',
                name=category,
                hoverinfo='name+y',
                text=[ json.dumps({ 'path': path, 'prediction': prediction }) for path, prediction in
                       zip(data.index[~correct], wrong_categories)]))
            
            print('Category {}, {} images. \t accuracy: {} top_{} accuracy: {} '
                  'correct confidence: {}, {}, {} wrong confidence: {}, {}, {} '
                  'diff: {}, {}, {}'
                  ''.format(category_i, len(data), pf(accuracy), top_k,
                            pf(top_k_accuracy),
                            pf(correct_confidence),
                            pf(correct_confidence_median),
                            pf(correct_confidence_std),
                            pf(wrong_confidence),
                            pf(wrong_confidence_median),
                            pf(wrong_confidence_std),
                            pf(correct_confidence - wrong_confidence),
                            pf(correct_confidence_median - wrong_confidence_median),
                            pf(correct_confidence_std - wrong_confidence_std)
                        ))

            all_accuracies.append(accuracy)
            all_top_k_accuracy.append(top_k_accuracy)
            stats.append([ category, pf(accuracy), pf(top_k_accuracy), pf(correct_confidence) ])

        mean_accuracy = pf(np.mean(all_accuracies))
        top_k_accuracy = pf(np.mean(all_top_k_accuracy))
        print('Average accuracy across categories: {}'.format(mean_accuracy))
        print('Average top_{} accuracy across categories: {}'.format(top_k, top_k_accuracy))

    tf.reset_default_graph()

    return plotly_data, mean_accuracy, top_k_accuracy, stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
    Performs statistics on a trained model.
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
    parser.add_argument(
        '--gpu',
        help='Which GPU to use for inference. Empty string means no GPU.',
        default='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    h5_files = sorted(glob.glob('{}/*'.format(args.test_folder)))

    for model in args.models:
        evaluate(model, h5_files, args.top_k)
