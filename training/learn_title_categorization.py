# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import json
import time
import glob
import datetime
import os

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Merge, BatchNormalization
from keras.preprocessing import sequence
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
    default=100)
parser.add_argument(
    '--lstm_size',
    type=int,
    default=128)
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256)
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
    for i,h5 in enumerate(sorted(glob.glob('{}/*'.format(folder)))):
        text.append(pd.read_hdf(h5))
        target.extend([i]*len(text[-1]))

    text = np.vstack(text)
    target = np.vstack(target)

    # This due to the nature of feeding an LSTM, where the input_dim is 1.
    text = np.expand_dims(text, -1)
        
    return text, target

t0 = time.time()

text_train, target_train = get_data(args.train_data)
text_test, target_test = get_data(args.test_data)

print 'Loading data took {} seconds'.format(time.time()-t0)

nb_classes = len(np.unique(target_train))

forward = Sequential()
forward.add(LSTM(args.lstm_size, input_dim=1, input_length=text_train.shape[1], consume_less='gpu', unroll=True,
               dropout_W=args.dropout, dropout_U=args.dropout))

reverse = Sequential()
reverse.add(LSTM(args.lstm_size, input_dim=1, input_length=text_train.shape[1], consume_less='gpu', unroll=True,
                 dropout_W=args.dropout, dropout_U=args.dropout, go_backwards=True))

model = Sequential()
model.add(Merge([ forward, reverse ], mode='concat', concat_axis=1))

model.add(Dense(args.hidden_size, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

check_cb = keras.callbacks.ModelCheckpoint(args.checkpoint_dir+args.filename+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

model.fit([ text_train, text_train ], target_train, nb_epoch=args.epochs, batch_size=args.batch_size,
          validation_data=([ text_test, text_test ], target_test), callbacks=[check_cb])
