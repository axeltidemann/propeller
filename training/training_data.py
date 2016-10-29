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
from collections import namedtuple, defaultdict
import gc

import numpy as np
import pandas as pd
import dask.array as da
import tensorflow as tf
import h5py
import ipdb

USE_DASK = True

print 'USE_DASK:', USE_DASK

def states(h5_files, separator='_'):
    
    length = 0
    for h5 in h5_files:
        with pd.HDFStore(h5) as store:
            storer = store.get_storer('data')
            length += storer.nrows
            width = storer.ncols

    if USE_DASK:
        datasets = [ h5py.File(fn)['/data/table'] for fn in h5_files ]

        dtype = [('index', 'S52'), ('values_block_0', '<f4', (2048,))]
        arrays = [ da.from_array(data, chunks=1000).astype(dtype) for data in datasets ]

        X = da.concatenate(arrays, axis=0)
        print X
    else:
        X = np.zeros((length, width))
    Y = np.zeros((length, len(h5_files)))

    start_index = 0
    
    for i, h5 in enumerate(h5_files):
        x = pd.read_hdf(h5)
        end_index = start_index + len(x)

        if not USE_DASK:
            X[start_index:end_index] = x
            
        Y[start_index:end_index,i] = 1

        start_index += len(x)

        gc.collect()

    h5_stripped = map(lambda x: os.path.basename(x[:x.rfind(separator)])
                          if os.path.basename(x).rfind(separator) > -1
                          else os.path.basename(x),
                          h5_files)
    
    pure = sorted(set(h5_stripped))

    filtr = np.zeros((1, len(h5_files), 1, len(pure)))

    if len(pure) < len(h5_files):
        print 'These HDF5 files are a result of a data hygiene operation, creating output filter.'
        
    for i, curated in enumerate(h5_stripped):
        for j, original in enumerate(pure):
            if original == curated:
                filtr[0,i,0,j] = 1
                
    assert all([ sum(row) == 1 for row in filtr[0,:,0,:] ]), 'The filter is wrong, there is more than one high value per row'

    filtr.astype('float')

    return (X,Y), filtr

class DataSet:
    def __init__(self, data):
        self._X, self._Y = data
        
        self._num_examples = self._X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if USE_DASK:
            self._indices = np.arange(self._num_examples)

        self.shuffle()

    def shuffle(self):
        if USE_DASK:
            np.random.shuffle(self._indices)
        else:
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

        if USE_DASK:
            
            return np.vstack([ values for index, values in self._X[self._indices[start:end]].compute() ]), \
                self._Y[self._indices[start:end]]
            #    np.vstack([ values for index, values in self._Y[self._indices[start:end]]])
        else:
            return self._X[start:end], self._Y[start:end]

        
    @property
    def epoch(self):
        return self._epochs_completed
    
    @property
    def X(self):
        if USE_DASK:
            return np.vstack([ values for index, values in self._X.compute() ])
        else:
            return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def X_features(self):
        if USE_DASK:
            return self._X.dtype[1].shape[0]
            return self._X[0].compute().shape[1]
        else:
            return self._X.shape[1]

    @property
    def Y_features(self):
        # if USE_DASK:
        #     return self._Y.dtype[1].shape[0]
        #     return self._Y[0].shape[1]
        # else:
        return self._Y.shape[1]

        
def read_data(train_folder, test_folder):

    class DataSets:
        pass

    h5_train = sorted(glob.glob('{}/*'.format(train_folder)))
    assert len(h5_train), 'The HDF5 folder is empty.'
    train, output_filter = states(h5_train)

    h5_test = sorted(glob.glob('{}/*'.format(test_folder)))
    test, _ = states(h5_test)

    data_sets = DataSets()

    data_sets.output_filter = output_filter
    
    data_sets.train = DataSet(train)
    print 'Training data: \t{} examples, {} features, {} categories.'.format(data_sets.train.X.shape[0],
                                                                           data_sets.train.X_features,
                                                                           data_sets.train.Y_features)
    
    data_sets.test = DataSet(test)
    print 'Testing data: \t{} examples, {} features, {} categories.'.format(data_sets.test.X.shape[0],
                                                                          data_sets.test.X_features, 
                                                                          data_sets.test.Y_features)
    return data_sets
