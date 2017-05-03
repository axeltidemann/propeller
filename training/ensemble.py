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
    type=int,
    default=1024)
parser.add_argument(
    '--epochs',
    type=int,
    default=10)
parser.add_argument(
    '--hidden_size',
    type=int,
    default=2048)
parser.add_argument(
    '--text_expert_hidden_size',
    type=int,
    default=512)
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
    '--mixture_mode',
    help='ensemble or moe',
    default='moe')
parser.add_argument(
    '--loss',
    help='simple or complex MOE loss function',
    default='complex')
parser.add_argument(
    '--n_experts',
    type=int,
    default=5)
args = parser.parse_args()

assert args.loss in ['simple', 'complex']
assert args.mixture_mode in ['ensemble', 'moe']

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


hiddens = []
outputs = []
for _ in range(args.n_experts):

    # Train them separately first, and then combine.
    
    # text experts
    _text_filters = merge(text_filters(), mode='concat')
    x = BatchNormalization()(_text_filters)

    x = Dense(args.text_expert_hidden_size, activation='relu')(x)
    hidden_text = BatchNormalization()(x)

    hiddens.append(hidden_text)

    text_predictions = Dense(n_classes, activation='softmax')(hidden_text)

    outputs.append(text_predictions)

    # text_model = Model(input=text_inputs, output=predictions)
    # text_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(clipnorm=5.), metrics=['accuracy'])

    # print(text_model.summary())
    
    # # Should not be necessary. Hopefully to be removed in the future.
    # keras.backend.get_session().run(tf.global_variables_initializer())

    # text_model.fit(text_train, target_train,
    #                nb_epoch=args.epochs,
    #                batch_size=args.batch_size,
    #                validation_data=(text_test, target_test))

    
    # experts.append(text_model)

    # x = Bidirectional(LSTM(args.text_expert_hidden_size/2, unroll=True, dropout_U=.5))(one_hot)
    # x = BatchNormalization()(x)

    # x = Dense(args.text_expert_hidden_size, activation='relu')(filters)
    # x = BatchNormalization()(x)

    # experts.append(Dense(n_classes, activation='softmax')(x))

    # vision expert
    x = Dense(args.hidden_size, activation='relu')(visual_inputs)
    hidden_vision = BatchNormalization()(x)

    hiddens.append(hidden_vision)
    
    vision_predictions = Dense(n_classes, activation='softmax')(hidden_vision)
    outputs.append(vision_predictions)

    # vision_model = Model(input=visual_inputs, output=predictions)
    # vision_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(clipnorm=5.), metrics=['accuracy'])

    # print(vision_model.summary())
    # # Should not be necessary. Hopefully to be removed in the future.
    # keras.backend.get_session().run(tf.global_variables_initializer())

    # vision_model.fit(visual_train, target_train,
    #                  nb_epoch=args.epochs,
    #                  batch_size=args.batch_size,
    #                  validation_data=(visual_test, target_test))

    
    # experts.append(vision_model)
    

# expert weighting

both = merge(hiddens, mode='concat')
x = Dense(args.hidden_size/2, activation='relu')(both)
x = BatchNormalization()(x)
x = Dense(args.hidden_size/2, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(n_classes, activation='softmax', name='OUTPUT')(x)

experts_out = [ text_predictions, vision_predictions ]*args.n_experts

model = Model(input=[text_inputs, visual_inputs], output=[ predictions ] +  experts_out)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(clipnorm=5.), metrics=['accuracy'])
#              loss_weights=[1., 0.2, 0.2])
print(model.summary())

check_cb = keras.callbacks.ModelCheckpoint(args.checkpoint_dir+args.filename+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

# Should not be necessary. Hopefully to be removed in the future.
keras.backend.get_session().run(tf.global_variables_initializer())



model.fit([ text_train, visual_train ], [ target_train ]*(args.n_experts*2 + 1),
          nb_epoch=args.epochs,
          batch_size=args.batch_size,
          validation_data=([ text_test, visual_test ], [ target_test ]*(args.n_experts*2 + 1)))
#          callbacks=[check_cb])
