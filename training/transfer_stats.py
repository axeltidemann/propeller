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
from collections import defaultdict

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
import numpy as np
import ipdb 
import plotly.graph_objs as go

from training_data import states
from utils import load_graph, pretty_float as pf

def evaluate(model, h5_files, top_k, categories=None, out_file='stats.h5'):
    h5_files = sorted(h5_files)

    plotly_data = []
    
    with tf.Session() as sess:
        print('Evaluating {}'.format(model))

        print('NOTE: ALL NUMBER TRIPLES ARE ON THE FORM (mean, median, standard deviation)')
        
        load_graph(model)

        transfer_predictor = sess.graph.get_tensor_by_name('output:0')

        all_accuracies = []
        all_top_k_accuracy = []
        all_top_level_accuracy = []
        stats = []

        with pd.HDFStore(out_file, mode='w', complevel=9, complib='blosc') as store:
            
            for target, h5 in enumerate(h5_files):

                data = pd.read_hdf(h5)
                category_i = os.path.basename(h5).replace('.h5','')

                predictions = sess.run(transfer_predictor, { 'input:0': data })

                top_level_accuracy = np.mean([ categories[category_i]['parent'] ==
                                               categories[os.path.basename(h5_files[prediction]).replace('.h5','')]['parent']
                                               for prediction in np.argmax(predictions, axis=1) ])

                correct = np.argmax(predictions, axis=1) == target
                accuracy = np.mean(correct)

                top_k_accuracy = np.mean([ target in np.argsort(prediction)[-top_k:]
                                           for prediction in predictions ])

                correct_scores = np.max(predictions[correct], axis=1)
                correct_x = np.linspace(0,1, num=len(correct_scores))
                correct_confidence = np.mean(correct_scores)
                correct_confidence_median = np.median(np.max(predictions[correct], axis=1))
                correct_confidence_std = np.std(np.max(predictions[correct], axis=1))

                category = categories[category_i]['name'] if categories else category_i

                sorted_correct = sorted(zip(correct_scores, data.index[correct]),
                                        key=lambda x: x[0])
                sorted_correct_scores, sorted_correct_paths = zip(*sorted_correct)

                df = pd.DataFrame(data=list(sorted_correct_scores), index=sorted_correct_paths, columns=['score'])
                df.index.name = 'filename'

                store.append('{}/correct'.format(category_i), df)
                
                wrong_scores = np.max(predictions[~correct], axis=1)
                wrong_categories_i = np.argmax(predictions[~correct], axis=1)
                wrong_x = np.linspace(0,1, num=len(wrong_scores))
                wrong_confidence = np.mean(wrong_scores)
                wrong_confidence_median = np.median(np.max(predictions[~correct], axis=1))
                wrong_confidence_std = np.std(np.max(predictions[~correct], axis=1))

                wrong_categories = [ os.path.basename(h5_files[i]).replace('.h5','')
                                     for i in wrong_categories_i ]
                
                sorted_wrong = sorted(zip(wrong_scores, data.index[~correct], wrong_categories),
                                      key=lambda x: x[0])
                sorted_wrong_scores, sorted_wrong_paths, sorted_wrong_categories = zip(*sorted_wrong)

                df = pd.DataFrame(data=zip(sorted_wrong_scores, sorted_wrong_categories),
                                  index=sorted_wrong_paths, columns=['score', 'category'])
                df.index.name='filename'

                store.append('{}/wrong/out'.format(category_i), df)

                spread = defaultdict(list)
                for score, path, category in sorted_wrong:
                    spread[category].append((path, score))

                for category, X in spread.items():
                    paths, scores = zip(*X)
                    
                    df = pd.DataFrame(data=list(scores), index=paths, columns=['score'])
                    df.index.name='filename'

                    store.append('{}/wrong/in'.format(category), df, min_itemsize={'index': 50})
                    
                # plotly_data.append(go.Scatter(
                #     x=wrong_x,
                #     y=sorted_wrong_scores,
                #     mode='markers',
                #     name=category,
                #     hoverinfo='name+y',
                #     text=[ json.dumps({ 'path': path, 'prediction': prediction }) for path, prediction in
                #            zip(sorted_wrong_paths, sorted_wrong_categories)]))

                print('Category {}, {} images. \t accuracy: {} top level accuracy: {} top_{} accuracy: {} '
                      'correct confidence: {}, {}, {} wrong confidence: {}, {}, {} '
                      'diff: {}, {}, {}'
                      ''.format(category_i, len(data), pf(accuracy), pf(top_level_accuracy),
                                top_k,
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
                all_top_level_accuracy.append(top_level_accuracy)
                stats.append([ category, pf(accuracy), pf(top_k_accuracy), pf(correct_confidence),
                               wrong_categories, wrong_scores, data.index[~correct] ])

            mean_accuracy = pf(np.mean(all_accuracies))
            top_k_accuracy = pf(np.mean(all_top_k_accuracy))
            top_level_accuracy = pf(np.mean(all_top_level_accuracy))
            print('Average accuracy across categories: {}'.format(mean_accuracy))
            print('Average top_{} accuracy across categories: {}'.format(top_k, top_k_accuracy))
            print('Average top level accuracy across categories: {}'.format(top_level_accuracy))

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
        'categories',
        help='JSON file with description of categories.')
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

    categories = json.load(open(args.categories))
    
    for model in args.models:
        evaluate(model, h5_files, args.top_k, categories)
