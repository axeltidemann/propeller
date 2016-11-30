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

from training_data import states
from utils import load_graph, pretty_float as pf

def evaluate(model, h5_files, top_k, categories, out_file, special, num_images):
    h5_files = sorted(h5_files)

    try:
        num_images = int(num_images)
    except:
        needle = 'images_'
        substr = model[model.find(needle) + len(needle):]
        num_images = substr[:substr.find('_')]
        num_images = int(num_images)
    
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

                with pd.HDFStore(h5, mode='r') as in_store:
                    keys = sorted(in_store.keys())[:num_images]

                if special:
                    all_predictions = defaultdict(list)
                    for key in keys:
                        X = pd.read_hdf(h5, key)
                        local_predictions = sess.run(transfer_predictor, { 'input:0': X })
                        for i, (prediction, x) in enumerate(zip(local_predictions, X.values)):
                            if sum(x): # No image present equals all 0s
                                all_predictions[i].append(prediction)

                    predictions = np.vstack([ np.mean(x, axis=0) for x in all_predictions.values() ])
                else:
                    X = np.hstack([ pd.read_hdf(h5, key) for key in keys ])
                    predictions = sess.run(transfer_predictor, { 'input:0': X })

                data = pd.read_hdf(h5, key) # We need the index, that is why we load the file again.
                correct = np.argmax(predictions, axis=1) == target
                accuracy = np.mean(correct)

                category_i = os.path.basename(h5).replace('.h5','')
                
                if 'parent' in categories[category_i]:
                    top_level_accuracy = np.mean([ 'parent' in categories[os.path.basename(h5_files[prediction]).replace('.h5','')] and
                                                   categories[category_i]['parent'] == categories[os.path.basename(h5_files[prediction]).replace('.h5','')]['parent']
                                                   for prediction in np.argmax(predictions, axis=1) ])
                else:
                    top_level_accuracy = accuracy
                
                top_k_accuracy = np.mean([ target in np.argsort(prediction)[-top_k:]
                                           for prediction in predictions ])

                correct_scores = np.max(predictions[correct], axis=1)
                correct_x = np.linspace(0,1, num=len(correct_scores))
                correct_confidence = np.mean(correct_scores)
                correct_confidence_median = np.median(np.max(predictions[correct], axis=1))
                correct_confidence_std = np.std(np.max(predictions[correct], axis=1))

                category = categories[category_i]['english']

                sorted_correct = sorted(zip(correct_scores, data.index[correct]),
                                        key=lambda x: x[0])
                sorted_correct_scores, sorted_correct_paths = zip(*sorted_correct)

                df = pd.DataFrame(data=list(sorted_correct_scores), index=sorted_correct_paths, columns=['score'])
                df.index.name = 'ad_id'

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
                df.index.name='ad_id'

                store.append('{}/wrong/out'.format(category_i), df)

                spread = defaultdict(list)
                for score, path, wrong_category in sorted_wrong:
                    spread[wrong_category].append((path, score))

                for wrong_category, X in spread.items():
                    paths, scores = zip(*X)
                    
                    df = pd.DataFrame(data=zip(scores, [category_i]*len(paths)), index=paths, columns=['score', 'category'])
                    df.index.name='ad_id'

                    store.append('{}/wrong/in'.format(wrong_category), df, min_itemsize={'index': 79, 'category': 12})

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

                index = [ 'test_len', 'train_len', 'accuracy', 'top_k_accuracy', 'k', 'num_images']
                values = [ len(data), len(data)*4, accuracy, top_k_accuracy, top_k, num_images ]

                stats = pd.DataFrame(data=values, index=index, columns=[category])
                store.append('{}/stats'.format(category_i), stats)

            mean_accuracy = pf(np.mean(all_accuracies))
            top_k_accuracy = pf(np.mean(all_top_k_accuracy))
            top_level_accuracy = pf(np.mean(all_top_level_accuracy))
            print('Average accuracy across categories: {}'.format(mean_accuracy))
            print('Average top_{} accuracy across categories: {}'.format(top_k, top_k_accuracy))
            print('Average top level accuracy across categories: {}'.format(top_level_accuracy))

    tf.reset_default_graph()

    return mean_accuracy, top_k_accuracy, stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
    Calculates statistics on a trained model, prints the results and saves to HDF5 file.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'test_folder',
        help='Folder with Inception states for testing')
    parser.add_argument(
        'model',
        help='Model to evaluate')
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
    parser.add_argument(
        '--num_images',
        help='How many images to process. If unspecified, will try to guess from the filename.',
        default='')
    parser.add_argument(
        '--out_file',
        help='Name of the HDF5 file with the results. If unspecified, will be model name + report.h5',
        default=False)
    parser.add_argument(
        '--special',
        help='Set this flag if you want to test out a classifier trained on everything, and classify each image separately and sum the result.',
        action='store_true')
    args = parser.parse_args()

    args.out_file = args.out_file if args.out_file else '{}_report.h5'.format(os.path.basename(args.model))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    h5_files = sorted(glob.glob('{}/*'.format(args.test_folder)))

    categories = json.load(open(args.categories))
    
    evaluate(args.model, h5_files, args.top_k, categories, args.out_file, args.special, args.num_images)
    
