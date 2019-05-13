# Copyright 2019 Telenor ASA, Author: Axel Tidemann

import argparse
import time
import glob
import random

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import ELU
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py

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
    help='Length of input sequences.',
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
    default=0)
parser.add_argument(
    '--test_validation_ratio',
    help='The ratio to use for validation and testing (50% each)',
    type=float,
    default=0.2)
parser.add_argument(
    '--test',
    help='Test on a smaller part of the dataset',
    action='store_true')
args = parser.parse_args()

N = 2048

t0 = time.time()

graphemes = pd.read_hdf(args.data, key='graphemes')
graphemes_used = graphemes.grapheme[:args.graphemes]
grapheme_map = { g:i for i,g in enumerate(graphemes_used) }

with h5py.File(args.data, 'r', libver='latest') as h5_file:
    categories = [ 'categories/{}'.format(c) for c in list(h5_file['categories'].keys()) ]
    sizes = [ h5_file['{}/table'.format(c)].shape[0] for c in categories ]

    min_size = 100 if args.test else min(sizes)
    data = pd.DataFrame()

    for target, category in enumerate(categories):
        _data = pd.read_hdf(args.data, key=category, stop=min_size)
        _data['target'] = target
        _data['text'] = _data.title.apply(lambda x: [ grapheme_map[g] for g in x if g in grapheme_map ])
        padded = pad_sequences(_data.text, maxlen=args.seq_len, padding='post', truncating='post')
        _data.text = [ pad for pad in padded ]
        _data['embeddings'] = [ h5_file['images/{}/{}'.format(category, ad_id)] for ad_id in _data.index ]

        data = data.append(_data)

    print('Loading data in {} seconds.'.format(time.time() - t0))

    train, X = train_test_split(data, shuffle=True, test_size=args.test_validation_ratio)

    validation, test = train_test_split(X, test_size=.5)

    filter_widths = range(1,7)
    nb_filters_coeff = 25

    text_inputs = Input(shape=(args.seq_len,), name='text')
    visual_inputs = Input(shape=(N,), name='vision')

    text_embedding = Embedding(args.graphemes + 1, args.embedding_size, input_length=args.seq_len)(text_inputs)

    filters = []
    for fw in filter_widths:
        x = Convolution1D(nb_filters_coeff*fw, fw, activation='tanh')(text_embedding)
        x = GlobalMaxPooling1D()(x)
        filters.append(x)

    fusion = Concatenate()(filters + [visual_inputs])

    x = Dropout(args.dropout)(fusion)
    x = Dense(args.hidden_size)(x)
    x = ELU()(x) # Alleviates need for batchnorm
    x = Dropout(args.dropout)(x)
    predictions = Dense(len(categories), activation='softmax')(x)

    model = Model(inputs=[text_inputs, visual_inputs], outputs=predictions)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    def serve(df):
        vision = np.vstack(df.embeddings.apply(lambda x: random.choice(x) if len(x) else np.zeros(N)))
        text = np.vstack(df.text)
        
        inputs = {'vision': vision, 'text': text}
        outputs = df.target

        return (inputs, outputs)

    def yield_batches(df, batch_size):
        while True:
            batch = df.sample(batch_size)
            yield serve(batch)

    train_data_generator = yield_batches(train, args.batch_size)
    validation_data = serve(validation)
    test_data = serve(test)

    model.fit_generator(train_data_generator,
                        steps_per_epoch=len(train)/args.batch_size,
                        epochs=args.epochs,
                        validation_data=validation_data)

    loss, accuracy = model.evaluate(test_data[0], test_data[1], batch_size=args.batch_size)
    print('Test accuracy {}'.format(accuracy))

    if args.filter_value > 0:
        predict = model.predict(test_data[0], batch_size=args.batch_size)
        results = pd.DataFrame(data=np.hstack([ np.expand_dims(np.max(predict, axis=1),-1),
                                                np.expand_dims(np.argmax(predict, axis=1),-1),
                                                test_data[1] ]), columns=['score', 'predicted', 'target'])
        results_filter = results[ results.score > args.filter_value ]
        filter_accuracy = np.mean(results_filter.predicted == results_filter.target)

        print('Filtered accuracy (above {}, {}%): {}'.format(args.filter_value,
                                                             100.*len(results_filter)/len(results),
                                                             filter_accuracy))
