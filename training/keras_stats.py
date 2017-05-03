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
import math

from collections import defaultdict


from keras.models import load_model
import keras.backend as K

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
import numpy as np
import ipdb 

from training_data import states
from utils import load_graph, pretty_float as pf

def to_text(numbers, encoding):
    texts = []
    for row in numbers:
        texts.append(','.join([ str(n) for n in row if n > 0 ])) # zero pads

    return texts

def evaluate(model, h5_files, top_k, categories, out_file, index_to_grapheme):
    h5_files = sorted(h5_files)

    print('Evaluating {}'.format(model))
    print('NOTE: ALL NUMBER TRIPLES ARE ON THE FORM (mean, median, standard deviation)')
    model = load_model(model)#, custom_objects={'moe_loss': moe_loss})

    all_accuracies = []
    all_top_k_accuracy = []
    all_top_level_accuracy = []
    stats = []

    with pd.HDFStore(out_file, mode='w', complevel=9, complib='blosc') as store:

        for target, h5 in enumerate(h5_files):

            text_inputs = pd.read_hdf(h5, 'text').values
            text_inputs = text_inputs[:,:args.seq_len]
            visual_inputs = pd.read_hdf(h5, 'visual')

            predictions = model.predict_on_batch([ text_inputs, visual_inputs ])

            index = visual_inputs.index
            
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
            correct_text = text_inputs[correct]

            category = categories[category_i]['english']

            sorted_correct = sorted(zip(correct_scores, index[correct], correct_text),
                                    key=lambda x: x[0])
            sorted_correct_scores, sorted_correct_paths, sorted_correct_text = zip(*sorted_correct)

            sorted_correct_text = to_text(sorted_correct_text, index_to_grapheme)
            
            df = pd.DataFrame(data=zip(sorted_correct_scores, sorted_correct_text), index=sorted_correct_paths, columns=['score', 'text'])
            df.index.name = 'ad_id'

            store.append('{}/correct'.format(category_i), df)

            wrong_scores = np.max(predictions[~correct], axis=1)
            wrong_categories_i = np.argmax(predictions[~correct], axis=1)
            wrong_x = np.linspace(0,1, num=len(wrong_scores))
            wrong_confidence = np.mean(wrong_scores)
            wrong_confidence_median = np.median(np.max(predictions[~correct], axis=1))
            wrong_confidence_std = np.std(np.max(predictions[~correct], axis=1))
            wrong_text = text_inputs[~correct]
            
            wrong_categories = [ os.path.basename(h5_files[i]).replace('.h5','')
                                 for i in wrong_categories_i ]

            sorted_wrong = sorted(zip(wrong_scores, index[~correct], wrong_categories, wrong_text),
                                  key=lambda x: x[0])

            sorted_wrong_scores, sorted_wrong_paths, sorted_wrong_categories, sorted_wrong_text = zip(*sorted_wrong)

            sorted_wrong_text = to_text(sorted_wrong_text, index_to_grapheme)
            
            df = pd.DataFrame(data=zip(sorted_wrong_scores, sorted_wrong_categories, sorted_wrong_text),
                              index=sorted_wrong_paths, columns=['score', 'category', 'text'])
            df.index.name='ad_id'

            store.append('{}/wrong/out'.format(category_i), df)

            spread = defaultdict(list)
            for score, path, wrong_category, wrong_text in zip(sorted_wrong_scores, sorted_wrong_paths, sorted_wrong_categories, sorted_wrong_text):
                spread[wrong_category].append((path, score, wrong_text))

            for wrong_category, X in spread.items():
                paths, scores, text = zip(*X)

                df = pd.DataFrame(data=zip(scores, [category_i]*len(paths), text), index=paths, columns=['score', 'category', 'text'])
                df.index.name='ad_id'

                store.append('{}/wrong/in'.format(wrong_category), df, min_itemsize={'index': 79, 'category': 12, 'text': 193})

            print('Category {}, {} images. \t accuracy: {} top level accuracy: {} top_{} accuracy: {} '
                  'correct confidence: {}, {}, {} wrong confidence: {}, {}, {} '
                  'diff: {}, {}, {}'
                  ''.format(category_i, len(index), pf(accuracy), pf(top_level_accuracy),
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
                           wrong_categories, wrong_scores, index[~correct] ])

            results_index = [ 'test_len', 'train_len', 'accuracy', 'top_k_accuracy', 'k' ]
            values = [ len(index), len(index)*4, accuracy, top_k_accuracy, top_k ]

            stats = pd.DataFrame(data=values, index=results_index, columns=[category])
            store.append('{}/stats'.format(category_i), stats)

        mean_accuracy = pf(np.mean(all_accuracies))
        top_k_accuracy = pf(np.mean(all_top_k_accuracy))
        top_level_accuracy = pf(np.mean(all_top_level_accuracy))
        print('Average accuracy across categories: {}'.format(mean_accuracy))
        print('Average top_{} accuracy across categories: {}'.format(top_k, top_k_accuracy))
        print('Average top level accuracy across categories: {}'.format(top_level_accuracy))

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
        'index_to_grapheme',
        help='JSON file that maps indices to graphemes')
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
        '--out_file',
        help='Name of the HDF5 file with the results. If unspecified, will be model name + report.h5',
        default=False)
    parser.add_argument(
        '--seq_len',
        help='Length of sequences. The sequences are typically longer than what might be needed.',
        type=int,
        default=50)
    args = parser.parse_args()

    args.out_file = args.out_file if args.out_file else '{}_report.h5'.format(os.path.basename(args.model))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    h5_files = sorted(glob.glob('{}/*'.format(args.test_folder)))

    categories = json.load(open(args.categories))
    index_to_grapheme = json.load(open(args.index_to_grapheme))
    
    evaluate(args.model, h5_files, args.top_k, categories, args.out_file, index_to_grapheme)
    
