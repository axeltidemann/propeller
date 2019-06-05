# Copyright 2019 Telenor ASA, Author: Axel Tidemann

import argparse
import time
import random
import os

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import ELU
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pandas as pd
import h5py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data',
    help='Folder with encoded titles + image features.')
parser.add_argument(
    'model',
    help='Saved model file.')
parser.add_argument(
    'graphemes_file',
    help='Graphemes HDF5 file')
parser.add_argument(
    'quantiles',
    help='Quantiles HDF5 file')
parser.add_argument(
    '--test',
    help='Test on a smaller part of the dataset',
    action='store_true')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024)
parser.add_argument(
    '--title_len',
    help='Length of title input',
    type=int,
    default=50)
parser.add_argument(
    '--desc_len',
    help='Length of description input',
    type=int,
    default=50)
parser.add_argument(
    '--graphemes',
    help='The top number of graphemes to use',
    type=int,
    default=200)
args = parser.parse_args()

graphemes = pd.read_hdf(args.graphemes_file)
graphemes_used = graphemes.grapheme[:args.graphemes]
grapheme_map = { g:i for i,g in enumerate(graphemes_used) }

quantiles = pd.read_hdf(args.quantiles)
quantiles = np.squeeze(quantiles.values)

model = keras.models.load_model(args.model)

N = 2048

def value_to_quantile(original_value, quantiles):
    if original_value <= quantiles[0]:
        return 0.0
    if original_value >= quantiles[-1]:
        return 1.0
    n_quantiles = float(len(quantiles) - 1)
    right = np.searchsorted(quantiles, original_value)
    left = right - 1

    interpolated = (left + ((original_value - quantiles[left])
                            / ((quantiles[right] + 1e-6) - quantiles[left]))) / n_quantiles
    return interpolated

with h5py.File(args.data, 'r', libver='latest') as h5_file:
    categories = [ 'categories/{}'.format(c) for c in sorted(h5_file['categories'].keys()) ]
    sizes = [ h5_file['{}/table'.format(c)].shape[0] for c in categories ]

    for category, size in zip(categories, sizes):
        print('{}: {}'.format(os.path.basename(category), size))

    min_size = 100 if args.test else min(sizes)

    print('Using {} samples'.format(min_size))
    
    data = pd.DataFrame()

    for target, category in enumerate(categories):
        _data = pd.read_hdf(args.data, key=category, stop=min_size)
        _data['target'] = target

        for text_field, seq_len in zip(['title', 'description'], [args.title_len, args.desc_len]):
            enc_name = '{}_encoded'.format(text_field)
            _data[enc_name] = _data[text_field].apply(lambda x: [0] if pd.isnull(x)
                                                      else [ grapheme_map[g] for g in x.lower() if g in grapheme_map ])
            padded = pad_sequences(_data[enc_name], maxlen=seq_len, padding='post', truncating='post')
            _data[enc_name] = [ p for p in padded ]
        
        _data['embeddings'] = [ h5_file['images/{}/{}'.format(category, ad_id)][:] for ad_id in _data.index ]
        _data['price_quantile'] = _data.price.apply(lambda x: 0 if pd.isnull(x) else value_to_quantile(x, quantiles)) # should be -1?
        
        data = data.append(_data)


def serve(df):
    vision = np.vstack(df.embeddings.apply(lambda x: x[0] if len(x) else np.zeros(N)))
    title = np.vstack(df['title_encoded'])
    description = np.vstack(df['description_encoded'])
    price = df.price_quantile

    inputs = {'vision': vision, 'title': title, 'description': description, 'price': price}
    outputs = df.target

    return (inputs, outputs)

test_data = serve(data)

loss, accuracy = model.evaluate(test_data[0], test_data[1], batch_size=args.batch_size)
print('Accuracy {}'.format(accuracy))

# predictions = model.predict(test_data[0], batch_size=args.batch_size)
# predictions = np.argmax(predictions, axis=1)

# for col in data.columns:
#     del data[col]

# data['prediction'] = predictions
# data.to_csv('predictions.csv')
