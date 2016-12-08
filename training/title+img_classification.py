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
from keras.layers import Dense, LSTM, Dropout, Merge, InputLayer, BatchNormalization
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
    default=128)
parser.add_argument(
    '--epochs',
    type=int,
    default=100)
parser.add_argument(
    '--lstm_size',
    type=int,
    default=2048)
parser.add_argument(
    '--hidden_size',
    type=int,
    default=2048)
parser.add_argument(
    '--dropout',
    type=float,
    default=.5)
parser.add_argument(
    '--filename',
    help='What to call the checkpoint files.',
    default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
parser.add_argument(
    '--checkpoint_dir',
    default='checkpoints/')
args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

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

    # This due to the nature of feeding an LSTM, where the input_dim is 1.
    text = np.expand_dims(text, -1)
        
    return text, visual, target

t0 = time.time()

text_train, visual_train, target_train = get_data(args.train_data)
text_test, visual_test, target_test = get_data(args.test_data)

print 'Loading data took {} seconds'.format(time.time()-t0)

nb_classes = len(np.unique(target_train))

model_text = Sequential()
model_text.add(LSTM(args.lstm_size, input_dim=1, input_length=text_train.shape[1],
                    consume_less='gpu', unroll=True,
                    dropout_W=args.dropout, dropout_U=args.dropout))

model_text_reverse = Sequential()
model_text_reverse.add(LSTM(args.lstm_size, input_dim=1, input_length=text_train.shape[1],
                            consume_less='gpu', unroll=True,
                            dropout_W=args.dropout, dropout_U=args.dropout,
                            go_backwards=True))

input_visual = Sequential()
input_visual.add(InputLayer(input_shape=(visual_train.shape[1],)))

# model_visual = Sequential()
# model_visual.add(Dense(args.lstm_size*2, input_shape=(visual_train.shape[1],)))

intermodal = Sequential()
#intermodal.add(Merge([ model_text, model_text_reverse, model_visual ], mode='concat', concat_axis=1))
intermodal.add(Merge([ model_text, model_text_reverse, input_visual ], mode='concat', concat_axis=1))

intermodal.add(Dense(args.hidden_size, activation='relu'))
intermodal.add(BatchNormalization())
intermodal.add(Dense(nb_classes, activation='softmax'))

intermodal.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(intermodal.summary())

check_cb = keras.callbacks.ModelCheckpoint(args.checkpoint_dir+args.filename+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

intermodal.fit([ text_train, text_train, visual_train ], target_train,
               validation_data=([text_test, text_test, visual_test], target_test),
               nb_epoch=args.epochs, batch_size=args.batch_size,
               callbacks=[check_cb])

# Final evaluation of the model
scores = intermodal.evaluate([ text_test, text_test, visual_test ], target_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
