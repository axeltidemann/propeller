# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py

'''
Holds the data from the Inception states, and returns a random batch
for stochastic gradient descent. This should be modified to allow
for humongous datasets, i.e. offline data access, using TensorFlow.FixedLengthRecordReader.
Currently, it reads all HDF5 files into memory. 

Author: Axel.Tidemann@telenor.com
'''

import glob
import os
from collections import namedtuple

import numpy as np
import pandas as pd

from clusters import find
from utils import flatten

Data = namedtuple('Data', 'x y')

def states(folder, train_ratio, validation_ratio, test_ratio, expert=False):
    h5_files = sorted(glob.glob('{}/*.h5'.format(folder)))

    train = {}
    validation = {}
    test = {}
    for i, h5 in enumerate(h5_files):

        category = os.path.basename(h5) 
        x = np.vstack(pd.read_hdf(h5, 'data').state)
        
        if expert:
            y = np.zeros((x.shape[0], 2))
            if os.path.basename(h5) == expert:
                y[:,0] = 1
            else:
                y[:,1] = 1
        else:
            y = np.zeros((x.shape[0], len(h5_files)))
            y[:,i] = 1
        
        train_length = int(x.shape[0]*train_ratio)
        validation_length = int(x.shape[0]*(train_ratio+validation_ratio))
        
        train[category] = Data(x[:train_length],y[:train_length])
        validation[category] = Data(x[train_length:validation_length], y[train_length:validation_length])
        test[category] = Data(x[validation_length:],y[validation_length:])

    return train, validation, test

class DataSet:
    def __init__(self, data, expert=False):
        self._X = np.vstack([ data[key].x for key in data.keys() ])
        self._Y = np.vstack([ data[key].y for key in data.keys() ])

        self._expert = expert
        if expert:
            self._expert_x = data[expert].x
            self._expert_y = data[expert].y
        
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

        if self._expert:
            np.random.shuffle(self._expert_x)
            return np.vstack([ self._X[start:end], self._expert_x[:batch_size] ]), \
                np.vstack([ self._Y[start:end], self._expert_y[:batch_size] ])
        else:
            return self._X[start:end], self._Y[start:end]
        
    @property
    def epoch(self):
        return self._epochs_completed
    
    @property
    def X(self):
        if self._expert:
            return np.vstack([ self._expert_x, self._X[:self._expert_x.shape[0]] ])
        else:
            return self._X

    @property
    def Y(self):
        if self._expert:
            return np.vstack([ self._expert_y, self._Y[:self._expert_y.shape[0]] ])
        else:
            return self._Y

    @property
    def X_features(self):
        return self._X.shape[1]

    @property
    def Y_features(self):
        return self._Y.shape[1]

        
def read_data(data_folder, train_ratio, validation_ratio, test_ratio, expert=False):

    class DataSets:
        pass

    train, validation, test = states(data_folder, train_ratio, validation_ratio, test_ratio, expert)

    data_sets = DataSets()
    
    data_sets.train = DataSet(train, expert)
    print 'Training data: {} examples, {} features, {} categories.'.format(data_sets.train.X.shape[0],
                                                                          data_sets.train.X.shape[1],
                                                                          data_sets.train.Y.shape[1])
    data_sets.validation = DataSet(validation, expert)
    print 'Validation data: {} examples.'.format(data_sets.validation.X.shape[0])
    data_sets.test = DataSet(test, expert)
    print 'Testing data: {} examples.'.format(data_sets.test.X.shape[0])
        
    return data_sets
