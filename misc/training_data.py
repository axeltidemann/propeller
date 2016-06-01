# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py

'''
Holds the data from the Inception states, and returns a random batch
for stochastic gradient descent. This should be modified to allow
for humongous datasets, i.e. offline data access, using TensorFlow.FixedLengthRecordReader.

Author: Axel.Tidemann@telenor.com
'''

import glob
import os
from collections import namedtuple

import numpy as np
import h5py

Data = namedtuple('Data', 'x y')

def states(folder, ratio):
    h5_files = sorted(glob.glob('{}/*.h5'.format(folder)))

    train = {}
    test = {}
    for i, h5 in enumerate(h5_files):
        category = os.path.basename(h5) 
        x = h5py.File(h5, 'r')['data']['states'].value
        y = np.zeros((x.shape[0], len(h5_files)))
        y[:,i] = 1
        train_length = int(x.shape[0]*ratio)
        train[category] = Data(x[:train_length],y[:train_length])
        test[category] = Data(x[train_length:],y[train_length:])

    return train, test

class DataSet:
    def __init__(self, data):
        self._X = np.vstack([ data[key].x for key in data.keys() ])
        self._Y = np.vstack([ data[key].y for key in data.keys() ])
        
        self._num_examples = self._X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X = self._X[perm]
            self._Y = self._Y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._Y[start:end]
            
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

        
def read_data(data_folder, ratio):
    class DataSets:
        pass

    train, test = states(data_folder, ratio)

    data_sets = DataSets()
    
    data_sets.train = DataSet(train)
    print 'Training data: {} examples, {} features, {} categories.'.format(data_sets.train.X.shape[0],
                                                                          data_sets.train.X.shape[1],
                                                                          data_sets.train.Y.shape[1])
    data_sets.test = DataSet(test)
    print 'Testing data: {} examples.'.format(data_sets.test.X.shape[0])
        
    return data_sets
