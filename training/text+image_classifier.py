# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import time
import glob
import datetime

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, LSTM, merge, Input
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import ELU
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'train_data',
    help='Folder with encoded titles + image features for training')
parser.add_argument(
    'test_data')
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
    '--lstm_size',
    type=int,
    default=128)
parser.add_argument(
    '--seq_len',
    help='Length of sequences. The sequences are typically longer than what might be needed.',
    type=int,
    default=50)
parser.add_argument(
    '--embedding',
    help='cnn or lstm',
    default='cnn')
parser.add_argument(
    '--mode',
    help='text, image or both',
    default='both')
args = parser.parse_args()

def get_data(folder):
    text = []
    visual = []
    target = []
    
    for i,h5 in enumerate(sorted(glob.glob('{}/*'.format(folder)))):
        text.append(pd.read_hdf(h5, 'text'))
        visual.append(pd.read_hdf(h5, 'visual'))
        target.extend([i]*len(text[-1]))

    text = np.vstack(text)
    visual = np.vstack(visual)
    target = np.vstack(target)

    return text, visual, target

t0 = time.time()

text_train, visual_train, target_train = get_data(args.train_data)
text_test, visual_test, target_test = get_data(args.test_data)

print 'Loading data took {} seconds'.format(time.time()-t0)

nb_classes = len(np.unique(target_train))

vocab_size = np.max(text_train)
params = np.eye(vocab_size+1, dtype=np.float32)

# 0 is the padding number, space is 1. However, space is an important character in thai.
# We don't need to confuse the convolution unnecessarily.
params[0][0] = 0 

filter_widths = range(1,7)
nb_filters_coeff = 25

text_train = text_train[:,:args.seq_len]
text_test = text_test[:,:args.seq_len]

text_inputs = Input(shape=(text_train.shape[1],))
visual_inputs = Input(shape=(visual_train.shape[1],))

one_hot = Embedding(vocab_size+1, vocab_size+1, weights=[params], trainable=False)(text_inputs)

if args.embedding == 'cnn':

    filters = []
    for fw in filter_widths:
        x = Convolution1D(nb_filters_coeff*fw, fw, activation='tanh')(one_hot)
        x = GlobalMaxPooling1D()(x)
        filters.append(x)

    text = merge(filters, mode='concat')

elif args.embedding == 'lstm':

    # accuracy 81% after 10 epochs, batch size 32, hidden 512
    text = Bidirectional(LSTM(args.lstm_size/2, unroll=True, dropout_U=.5))(one_hot)

if args.mode == 'text':
    fusion = text
if args.mode == 'image':
    fusion = visual_inputs
if args.mode == 'both':
    fusion = merge([text, visual_inputs], mode='concat')
    
x = Dense(args.hidden_size)(fusion)
x = ELU()(x) # Alleviates need for batchnorm
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=[text_inputs, visual_inputs], output=predictions)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit([ text_train, visual_train ], target_train,
          nb_epoch=args.epochs,
          batch_size=args.batch_size,
          validation_data=([ text_test, visual_test ], target_test))
