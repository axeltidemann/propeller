# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import time
import glob

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, LSTM, merge, Input, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import ELU
from sklearn.model_selection import train_test_split
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data',
    help='Folder with encoded titles + image features. This will be split into training, validation and test data.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024)
parser.add_argument(
    '--epochs',
    type=int,
    default=10)
parser.add_argument(
    '--hidden_size',
    type=int,
    default=128)
parser.add_argument(
    '--embedding_size',
    type=int,
    default=128)
parser.add_argument(
    '--dropout',
    type=float,
    default=.5)
parser.add_argument(
    '--seq_len',
    help='Length of sequences. The sequences are typically longer than what might be needed.',
    type=int,
    default=50)
parser.add_argument(
    '--graphemes',
    help='The top number of graphemes to use',
    type=int,
    default=200)
parser.add_argument(
    '--filter_value',
    help='Calculate stats for predictions above this threshold',
    type=float,
    default=0.0)
parser.add_argument(
    '--test_validation_ratio',
    help='The ratio to use for validation and testing (50% each)',
    type=float,
    default=0.2)
args = parser.parse_args()

t0 = time.time()

with h5py.File(args.hdf, 'r', libver='latest') as h5_file:
    categories = [ 'categories/{}'.format(c) for c in list(h5_file['categories'].keys()) ]
    sizes = [ h5_file['{}/table'].shape[0] for c in categories ]

min_size = min(sizes)
data = pd.DataFrame()

for target, category in enumerate(categories):
    _data = pd.read_hdf(args.data, key=category, stop=min_size)
    _data['target'] = target

    data = data.append(_data)

train, X = train_test_split(data, shuffle=True, test_size=args.test_validation_ratio)
validation, test = train_test_split(X, test_size=.5)

graphemes = pd.read_hdf(args.data, key='graphemes')
graphemes_used = graphemes.grapheme[:args.graphemes]
grapheme_map = { g:i for i,g in enumerate(graphemes_used) }

###### THIS IS WHERE WE GOT

test_accuracies = []
filter_accuracies = []
filter_lengths = []


vocab_size = np.max(titles)
params = np.eye(vocab_size+1, dtype=np.float32)

# 0 is the padding number, space is 1. However, space is an important character in thai.
# We don't need to confuse the convolution unnecessarily.
params[0][0] = 0 

filter_widths = range(1,7)
nb_filters_coeff = 25

text_inputs = Input(shape=(text_train.shape[1],))
visual_inputs = Input(shape=(visual_train.shape[1],))

one_hot = Embedding(vocab_size+1, vocab_size+1, weights=[params], trainable=False)(text_inputs)

if args.embedding == 'cnn':

    filters = []
    for fw in filter_widths:
        x = Convolution1D(nb_filters_coeff*fw, fw, activation='tanh')(one_hot)
        x = GlobalMaxPooling1D()(x)
        filters.append(x)

    text_embedding = merge(filters, mode='concat')

elif args.embedding == 'lstm':

    text_embedding = Bidirectional(LSTM(args.lstm_size/2, unroll=True, dropout_U=args.dropout))(one_hot)

if args.mode == 'text':
    fusion = text_embedding
if args.mode == 'image':
    fusion = visual_inputs
if args.mode == 'both':
    fusion = merge([text_embedding, visual_inputs], mode='concat')

x = Dense(args.hidden_size)(fusion)
x = ELU()(x) # Alleviates need for batchnorm
x = Dropout(args.dropout)(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=[text_inputs, visual_inputs], output=predictions)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit([ text_train, visual_train ], target_train,
          nb_epoch=args.epochs,
          batch_size=args.batch_size,
          validation_data=([ text_validate, visual_validate ], target_validate))

loss, accuracy = model.evaluate([ text_test, visual_test ], target_test, batch_size=args.batch_size)
print '\nTest accuracy experiment {}: {}'.format(experiment, accuracy)

test_accuracies.append(accuracy)

if args.filter_value > 0:
    predict = model.predict([ text_test, visual_test ], batch_size=args.batch_size)
    results = pd.DataFrame(data=np.hstack([ np.expand_dims(np.max(predict, axis=1),-1),
                                            np.expand_dims(np.argmax(predict, axis=1),-1),
                                            target_test ]), columns=['score', 'predicted', 'target'])
    results_filter = results[ results.score > args.filter_value ]
    filter_accuracy = np.mean(results_filter.predicted == results_filter.target)

    print '\nFiltered accuracy (above {}, {}%) experiment {}: {}'.format(args.filter_value,
                                                                         100.*len(results_filter)/len(results),
                                                                         experiment, filter_accuracy)

    filter_accuracies.append(filter_accuracy)
    filter_lengths.append(float(len(results_filter))/len(results))

print 'Test accuracy over {} experiments: mean: {} std: {}'.format(args.n_experiments, np.mean(test_accuracies),
                                                                   np.std(test_accuracies))

if args.filter_value > 0:
    print 'Filtered accuracy value (above {}) over {} experiments: mean {} std: {}. Average length: {}%'.format(args.filter_value,
                                                                                                                args.n_experiments,
                                                                                                                np.mean(filter_accuracies),
                                                                                                                np.std(filter_accuracies),
                                                                                                                100.*np.mean(filter_lengths))
