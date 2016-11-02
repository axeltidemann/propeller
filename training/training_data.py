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
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../misc')))


import numpy as np
import pandas as pd
import dask.array as da
import tensorflow as tf
import h5py
import ipdb
from sklearn.utils import shuffle

from utils import pretty_float as pf

def states(h5_files, use_dask, dask_chunksize, separator='_'):
    
    h5_lengths = {}
    for h5 in h5_files:
        with pd.HDFStore(h5) as store:
            storer = store.get_storer('data')
            width = storer.ncols
            
            h5_lengths[h5] = storer.nrows

    length = sum(h5_lengths.values())
    
    if use_dask:
        datasets = [ h5py.File(fn)['/data/table'] for fn in h5_files ]
        X = [ da.from_array(data, chunks=dask_chunksize) for data in datasets ]
        Y = []
    else:
        X = np.zeros((length, width))
        Y = np.zeros((length,))

    start_index = 0
    for i, h5 in enumerate(h5_files):
        end_index = start_index + h5_lengths[h5]

        if use_dask:
            Y.append(np.ones((h5_lengths[h5],))*i)
        else:
            X[start_index:end_index] = pd.read_hdf(h5)
            Y[start_index:end_index] = i            

        start_index = end_index

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
    def __init__(self, data, use_dask, in_memory):
        self._X, self._Y = data
        self._use_dask = use_dask
        
        if self._use_dask:
            self._dask_X = self._X
            self._dask_Y = self._Y

            self._pr_category = int(in_memory/len(self._X))

            print '{} categories, {} vectors pr category will be loaded from disk.'.format(len(self._X), self._pr_category)

            self._indices = [ [] for _ in range(len(self._dask_X)) ]
            self.Y_features = len(self._Y)
        else:
            self.Y_features = len(np.unique(self._Y))
            self._num_examples = self._X.shape[0]

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.shuffle()

        
    def load(self):
        X = []
        Y = []

        t0 = time.time()
        for i, (x, y) in enumerate(zip(self._dask_X, self._dask_Y)):
            if len(self._indices[i]) == 0:
                if x.shape[0] > self._pr_category:
                    start_index = 0 if np.random.choice([True, False]) else x.shape[0] % self._pr_category
                    indices = range(start_index, x.shape[0], self._pr_category)
                    np.random.shuffle(indices)
                    self._indices[i] = indices
                else:
                    self._indices[i] = [0]
            
            j = self._indices[i].pop()
            X.extend([ values for index, values in x[j:j+self._pr_category].compute() ])
            Y.extend(y[j:j+self._pr_category])

        X = np.vstack(X)
        Y = np.squeeze(np.vstack(Y))
        
        print '\t{} vectors read from disk in {} seconds.'.format(X.shape[0], pf(time.time()-t0))

        return X, Y
        
    def shuffle(self):
        if self._use_dask:
            self._X, self._Y = self.load()
            self._num_examples = self._X.shape[0]

        self._X, self._Y = shuffle(self._X, self._Y)
        
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
    def X_len(self):
        if self._use_dask:
            return sum([ x.shape[0] for x in self._dask_X ])
        else:
            return self._X.shape[0]

    @property
    def X_features(self):
        return self._X.shape[1]
            
    @property
    def Y(self):
        return self._Y

        
def read_data(train_folder, test_folder, use_dask, in_memory, dask_chunksize):

    class DataSets:
        pass

    h5_train = sorted(glob.glob('{}/*'.format(train_folder)))
    assert len(h5_train), 'The HDF5 folder is empty.'
    train, output_filter = states(h5_train, use_dask, dask_chunksize)

    h5_test = sorted(glob.glob('{}/*'.format(test_folder)))
    test, _ = states(h5_test, False, dask_chunksize) # We assume testing data can fit in memory.

    data_sets = DataSets()

    data_sets.output_filter = output_filter
    
    data_sets.train = DataSet(train, use_dask, in_memory)
    print 'Training data: \t{} examples, {} features, {} categories.'.format(data_sets.train.X_len,
                                                                           data_sets.train.X_features,
                                                                           data_sets.train.Y_features)
    
    data_sets.test = DataSet(test, False, in_memory) 
    print 'Testing data: \t{} examples, {} features, {} categories.'.format(data_sets.test.X_len,
                                                                          data_sets.test.X_features, 
                                                                          data_sets.test.Y_features)
    return data_sets
