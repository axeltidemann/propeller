# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import json
import time
import glob
import datetime
import os

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, LSTM, merge, BatchNormalization, Lambda, Input
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser(description='''
Classifies sequences of graphemes into categories
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'train_data',
    help='Folder with encoded textual descriptions')
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
    '--lstm_size',
    type=int,
    default=64)
parser.add_argument(
    '--hidden_size',
    type=int,
    default=2048)
parser.add_argument(
    '--conv_size',
    type=int,
    default=32)
parser.add_argument(
    '--embedding_size',
    type=int,
    default=100)
parser.add_argument(
    '--filter_length',
    type=int,
    default=3)
parser.add_argument(
    '--dropout',
    type=float,
    default=.5)
parser.add_argument(
    '--checkpoint_dir',
    default='checkpoints/')
parser.add_argument(
    '--filename',
    help='What to call the checkpoint files.',
    default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

def get_data(folder):
    text = []
    target = []

    folder_list = sorted(glob.glob('{}/*'.format(folder)))

    for i,h5 in enumerate(folder_list):
        text.append(pd.read_hdf(h5))#, 'text'))
        target.extend([i]*len(text[-1]))

    text = np.vstack(text)
    target = np.vstack(target)
        
    return text, target

t0 = time.time()

text_train, target_train = get_data(args.train_data)
text_test, target_test = get_data(args.test_data)

print 'Loading data took {} seconds'.format(time.time()-t0)

nb_classes = len(np.unique(target_train))

params = np.eye(np.max(text_train), dtype=np.float32)

# 0 is the padding number, don't need to confuse the convolution unnecessary
params[0][0] = 0 

def embedding(x):
    return tf.nn.embedding_lookup(params, tf.to_int32(x))

def embedding_shape(input_shape):
    return input_shape[0], input_shape[1], params.shape[1]

filter_widths = range(1,11)
nb_filters_coeff = 50
    
inputs = Input(shape=(text_train.shape[1],))

e = Lambda(embedding, output_shape=embedding_shape)(inputs)

# e = Embedding(np.max(text_train)+1, 75, input_length=text_train.shape[1])(inputs)

filters = []
for fw in filter_widths:
    x = Convolution1D(min(200, nb_filters_coeff*fw), fw, activation='tanh')(e)
    x = GlobalMaxPooling1D()(x)
    filters.append(x)

merge = merge(filters, mode='concat')
merge = BatchNormalization()(merge)

x = Dense(args.hidden_size, activation='relu')(merge)
x = BatchNormalization()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

check_cb = keras.callbacks.ModelCheckpoint(args.checkpoint_dir+args.filename+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

model.fit(text_train, target_train, nb_epoch=args.epochs, batch_size=args.batch_size,
          validation_data=(text_test, target_test), callbacks=[check_cb])
