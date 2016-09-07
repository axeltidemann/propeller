# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py

'''
Holds the data from the Inception states, and returns a random batch
for stochastic gradient descent.

Author: Axel.Tidemann@telenor.com
'''

from __future__ import division
import glob
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf

from clusters import find
from utils import flatten

Data = namedtuple('Data', 'x y')

def states(folder):
    h5_files = sorted(glob.glob('{}/*'.format(folder)))

    data = {}
    for i, h5 in enumerate(h5_files):

        category = os.path.basename(h5) 
        x = np.vstack(pd.read_hdf(h5, 'data').state)

        y = np.zeros((x.shape[0], len(h5_files)))
        y[:,i] = 1
        
        data[category] = Data(x, y)

    sep = '_'

    h5_stripped = map(lambda x: os.path.basename(x[:x.rfind(sep)])
                          if os.path.basename(x).rfind(sep) > -1
                          else os.path.basename(x),
                          h5_files)
    
    pure = sorted(set(h5_stripped))

    filtr = np.zeros((1, len(h5_files), 1, len(pure)))

    if len(pure) < len(h5_files):
        print 'These HDF5 files are part of a clustering operation, creating output filter.'
        
    for i, curated in enumerate(h5_stripped):
        for j, original in enumerate(pure):
            if original == curated:
                filtr[0,i,0,j] = 1

    assert all([ sum(row) == 1 for row in filtr[0,:,0,:] ]), 'The filter is wrong, there is more than one high value per row'

    filtr.astype('float')

    return data, filtr

class DataSet:
    def __init__(self, data):
        self._X = np.vstack([ data[key].x for key in data.keys() ])
        self._Y = np.vstack([ data[key].y for key in data.keys() ])
        
        self._num_examples = self._X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.shuffle()

    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self._X)
        np.random.set_state(rng_state)
        np.random.shuffle(self._Y)
        
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
            
        end = self._index_in_epoch

        return self._X[start:end], self._Y[start:end]
        
    @property
    def epoch(self):
        return self._epochs_completed
    
    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def X_features(self):
        return self._X.shape[1]

    @property
    def Y_features(self):
        return self._Y.shape[1]

        
def read_data(train_folder, test_folder):

    class DataSets:
        pass

    train, output_filter = states(train_folder)
    test, _ = states(test_folder)

    data_sets = DataSets()

    data_sets.output_filter = output_filter
    
    data_sets.train = DataSet(train)
    print 'Training data: {} examples, {} features, {} categories.'.format(data_sets.train.X.shape[0],
                                                                           data_sets.train.X.shape[1],
                                                                           data_sets.train.Y.shape[1])
    
    data_sets.test = DataSet(test)
    print 'Testing data: {} examples, {} features, {} categories.'.format(data_sets.test.X.shape[0],
                                                                          data_sets.test.X.shape[1],
                                                                          data_sets.test.Y.shape[1])
    return data_sets
