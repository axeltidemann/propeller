# Copyright 2017 Telenor ASA, Author: Axel Tidemann

import argparse
import json
import time
import glob
import datetime
import os
import math

import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, LSTM, merge, BatchNormalization, Lambda, Input
from keras.layers.core import Flatten, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'train_data',
    help='Folder with encoded titles + image features for training')
parser.add_argument(
    'test_data')
parser.add_argument(
    '--batch_size',
    help='Batch size for training',
    type=int,
    default=1024)
parser.add_argument(
    '--epochs',
    help='Epochs to train for',
    type=int,
    default=10)
parser.add_argument(
    '--hidden_size',
    help='Size of hidden layers after visual + text filters',
    type=int,
    default=2048)
parser.add_argument(
    '--seq_len',
    help='Length of sequences. The sequences are typically longer than what might be needed.',
    type=int,
    default=50)
parser.add_argument(
    '--filename',
    help='What to call the checkpoint files.',
    default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
parser.add_argument(
    '--checkpoint_dir',
    default='checkpoints/')
parser.add_argument(
    '--loss',
    help='simple or complex MOE loss function',
    default='complex')
parser.add_argument(
    '--n_experts',
    help='Number of experts',
    type=int,
    default=5)
parser.add_argument(
    '--save_model',
    help='Filename of saved model',
    default=False)
args = parser.parse_args()

assert args.loss in ['simple', 'complex']

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

    return text, visual, target

t0 = time.time()

text_train, visual_train, target_train = get_data(args.train_data)
text_test, visual_test, target_test = get_data(args.test_data)

print 'Loading data took {} seconds'.format(time.time()-t0)

n_classes = len(np.unique(target_train))

# Hours of debugging, and not hot encoding the target was the issue.
# Strange. Maybe to deal with int/float/accuracy problems?
target_train = to_categorical(target_train, n_classes)
target_test = to_categorical(target_test, n_classes)

vocab_size = np.max(text_train)
params = np.eye(vocab_size+1, dtype=np.float32)

# 0 is the padding number from the pre-processing, space is 1.
# However, space is an important character in thai.
# We don't need to confuse the convolution unnecessary.
params[0][0] = 0 

text_train = text_train[:,:args.seq_len]
text_test = text_test[:,:args.seq_len]

text_inputs = Input(shape=(text_train.shape[1],))
visual_inputs = Input(shape=(visual_train.shape[1],))

one_hot = Embedding(vocab_size+1, vocab_size+1, weights=[params], trainable=False)(text_inputs)

# The Convolution1D character embedding is taken from "Character-Aware Neural Language Models", Kim et al., 2016
filter_widths = range(1,7)
nb_filters_coeff = 25

def text_filters():
    filters = []
    for fw in filter_widths:
        x = Convolution1D(nb_filters_coeff*fw, fw, activation='tanh')(one_hot)
        x = GlobalMaxPooling1D()(x)
        filters.append(x)
        
    return filters

experts = []

for i in range(args.n_experts):

    #lstm = Bidirectional(LSTM(args.text_expert_hidden_size/2, unroll=True, dropout_U=.5))(one_hot)
    #x = merge(text_filters() + [ lstm, visual_inputs ], mode='concat')

    x = merge(text_filters() + [ visual_inputs ], mode='concat')
    x = BatchNormalization()(x)

    x = Dense(args.hidden_size, activation='relu')(x)
    x = BatchNormalization()(x)

    experts.append(Dense(n_classes, activation='softmax', name='expert_{}'.format(i))(x))

# Gating network receives the same input as the other experts, and therefore has its own text filters.
multi_modal = merge(text_filters() + [ visual_inputs ], mode='concat')
x = BatchNormalization()(multi_modal)
x = Dense(args.hidden_size, activation='relu')(x)
x = BatchNormalization()(x)

gate = Dense(len(experts), activation='softmax')(x)

def experts_merge(branches):
    _gate = branches[0]
    _experts = branches[1:]

    # First value is batch size. A bit confusing.
    output_shape = K.int_shape(_experts[0])[1]
    out = K.zeros(shape=(output_shape,))

    for i,e in enumerate(_experts):
        out += K.expand_dims(_gate[:,i],1)*e

    return out
    
predictions = merge([gate] + experts, mode=experts_merge, output_shape=(n_classes,))

model = Model(input=[text_inputs, visual_inputs], output=predictions)

k = 1./np.sqrt(2*math.pi)

def simple_moe_loss(y_true, y_pred):
    errors = K.zeros(shape=(n_classes,))

    for i,e in enumerate(experts):
        errors += K.expand_dims(gate[:,i],1)*K.square(y_true - e)

    return K.sum(errors, axis=1)

def moe_loss(y_true, y_pred):
    errors = K.zeros(shape=(1,))

    for i,e in enumerate(experts):
        gt = K.expand_dims(gate[:,i]*k,1)
        err = K.exp(-.5*K.sum(K.square(y_true - e), axis=1))
        errors += gt*err

    return -K.log(errors)

loss_fn = simple_moe_loss if args.loss == 'simple' else moe_loss
    
model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(clipnorm=5.), metrics=['accuracy'])
print(model.summary())

check_cb = keras.callbacks.ModelCheckpoint(args.checkpoint_dir+args.filename+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

# Should not be necessary. Hopefully to be removed in the future.
keras.backend.get_session().run(tf.global_variables_initializer())

model.fit([ text_train, visual_train ], target_train,
          nb_epoch=args.epochs,
          batch_size=args.batch_size,
          validation_data=([ text_test, visual_test ], target_test),
          callbacks=[check_cb])

if args.save_model:
    model.save(args.save_model)
