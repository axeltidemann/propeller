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
    '--lstm_size',
    type=int,
    default=512)
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
    '--embedding',
    help='cnn or lstm',
    default='cnn')
parser.add_argument(
    '--mode',
    help='text, image or both',
    default='both')
parser.add_argument(
    '--n_experiments',
    help='Number of experiments to run',
    type=int,
    default=1)
args = parser.parse_args()

t0 = time.time()

titles = []
images = []
targets = []

for i,h5 in enumerate(sorted(glob.glob('{}/*'.format(args.data)))):
    titles.append(pd.read_hdf(h5, 'text'))
    images.append(pd.read_hdf(h5, 'visual'))
    targets.extend([i]*len(titles[-1]))

titles = np.vstack(titles)
images = np.vstack(images)
targets = np.vstack(targets)

print 'Loading data took {} seconds'.format(time.time()-t0)

test_accuracies = []

for experiment in range(args.n_experiments):

    idx = np.random.permutation(len(titles))

    train_idx = idx[:int(len(titles)*.8)]
    validate_idx = idx[int(len(titles)*.8):int(len(titles)*.9)]
    test_idx = idx[int(len(titles)*.9):]

    text_train = titles[train_idx, :args.seq_len]
    text_validate = titles[validate_idx, :args.seq_len]
    text_test = titles[test_idx, :args.seq_len]

    visual_train = images[train_idx]
    visual_validate = images[validate_idx]
    visual_test = images[test_idx]

    target_train = targets[train_idx]
    target_validate = targets[validate_idx]
    target_test = targets[test_idx]

    nb_classes = len(np.unique(targets))

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

print 'Test accuracy over {} experiments: mean: {} std: {}'.format(args.n_experiments, np.mean(test_accuracies),
                                                                   np.std(test_accuracies))
