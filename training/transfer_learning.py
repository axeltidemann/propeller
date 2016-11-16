# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

from __future__ import print_function
import argparse
import time
import os
import threading
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../misc')))

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.utils import shuffle

from training_data import read_data
from utils import pretty_float as pf, trueXor, chunks

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def learn(train_states, test_states, learning_rate, save_every, batch_size, hidden_size, dropout, epochs,
          print_every, model_dir, perceptron, dense, lenet, filter_width, depth,
          q_size, use_dask, in_memory, dask_chunksize):

    data = read_data(train_states, test_states, use_dask, in_memory, dask_chunksize)

    model_name = ('''transfer_classifier_epochs_{}_batch_{}_learning_rate_{}'''.format(
        epochs, batch_size, learning_rate))
    
    if perceptron:
        model_name = '{}_perceptron.pb'.format(model_name)
    if dense:
        model_name = '{}_dense_dropout_{}_hidden_size_{}.pb'.format(model_name, dropout, hidden_size)
    if lenet:
        model_name = '{}_lenet_dropout_{}_hidden_size_{}.pb'.format(model_name, dropout, hidden_size)

    with tf.Session() as sess:
        
        q_x_in = tf.placeholder(tf.float32, shape=[batch_size, data.train.X_features])
        q_y_in = tf.placeholder(tf.int32, shape=[batch_size])
        y_real = tf.placeholder(tf.int32, shape=[None])
        keep_prob = tf.placeholder_with_default([1.], shape=None)
        
        q = tf.FIFOQueue(q_size, [tf.float32, tf.int32],
                         shapes=[ q_x_in.get_shape(), q_y_in.get_shape()])

        enqueue_op = q.enqueue([q_x_in, q_y_in])
        q_x_out, q_y_out = q.dequeue()

        x = tf.placeholder_with_default(q_x_out, shape=[None, data.train.X_features], name='input')
        y_ = tf.placeholder_with_default(q_y_out, shape=[None])

        if perceptron:
            W = weight_variable([data.train.X_features, data.train.Y_features])
            b = bias_variable([data.train.Y_features])

            logits = tf.matmul(x, W) + b
        if dense:
            W_in = weight_variable([data.train.X_features, hidden_size])
            b_in = bias_variable([hidden_size])

            hidden = tf.matmul(x, W_in) + b_in
            relu = tf.nn.relu(hidden)
            hidden_dropout = tf.nn.dropout(relu, keep_prob)

            W_out = weight_variable([hidden_size,data.train.Y_features])
            b_out = bias_variable([data.train.Y_features])

            logits = tf.matmul(hidden_dropout, W_out) + b_out
        if lenet:
            w1 = weight_variable([1, filter_width, 1, depth])
            b1 = bias_variable([depth])
            x_4d = tf.expand_dims(tf.expand_dims(x,1),-1) # Singleton dimension height, out_channel
            conv = tf.nn.conv2d(x_4d, w1, strides=[1,1,1,1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b1))
            pool = tf.nn.max_pool(relu, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

            w2 = weight_variable([1, filter_width, depth, depth*2])
            b2 = bias_variable([depth*2])
            conv = tf.nn.conv2d(pool, w2, strides=[1,1,1,1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b2))
            pool = tf.nn.max_pool(relu, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

            pool_shape = tf.shape(pool)
            reshape = tf.reshape(pool,
                                 [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3] ])

            w3 = weight_variable([ (data.train.X_features/4)*2*depth, hidden_size])
            b3 = bias_variable([hidden_size])
            hidden = tf.matmul(reshape, w3) + b3
            relu = tf.nn.relu(hidden)
            hidden_dropout = tf.nn.dropout(relu, keep_prob)

            w4 = weight_variable([hidden_size, data.train.Y_features])
            b4 = bias_variable([data.train.Y_features])
            logits = tf.matmul(hidden_dropout, w4) + b4

        # Loss & train
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        # Evaluation
        y = tf.nn.softmax(logits) 
        train_correct_prediction = tf.equal(tf.to_int32(tf.argmax(y, 1)), y_,)
        train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        # Applying convolutional filter to put subcategories into original categories for testing
        stride = [1,1,1,1]
        _filter = tf.constant(data.output_filter, dtype=tf.float32, shape=data.output_filter.shape)

        conv_in = tf.expand_dims(y, 0)
        conv_in = tf.expand_dims(conv_in,-1)
        conv_out = tf.nn.conv2d(conv_in, _filter, stride, 'VALID') # We don't want zero padding.
        back = tf.squeeze(conv_out, squeeze_dims=[0,2], name='output')

        test_correct_prediction = tf.equal(tf.to_int32(tf.argmax(back, 1)), y_real)
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
        
        sess.run(tf.initialize_all_variables())

        def load_data():
            try:
                while True:
                    next_x, next_y = data.train.next_batch(batch_size)
                    if next_x.shape[0] == batch_size:
                        sess.run(enqueue_op, feed_dict={q_x_in: next_x, q_y_in: next_y})
            except Exception as error:
                print(error)
                print('Stopped streaming of data.')
                
        data_thread = threading.Thread(target=load_data)
        data_thread.daemon = True
        data_thread.start()

        last_epoch = 0
        epoch = 0
        
        t_epoch = time.time()
        t_end = []
        i = 0 
        while epoch <= epochs:

            t_start = time.time()
            sess.run(train_step, feed_dict={keep_prob: dropout})
            t_end.append(time.time() - t_start)

            if epoch > last_epoch:

                if epoch % print_every == 0:
                    batch_x, batch_y = data.train.next_batch(batch_size)
                    
                    train_accuracy_mean = train_accuracy.eval(feed_dict={
                        x: batch_x,
                        y_: batch_y })

                    validation_accuracy_mean = np.mean([ test_accuracy.eval(feed_dict={x: t_x, y_real: t_y })
                                                         for t_x, t_y in zip(chunks(data.test.X, batch_size),
                                                                             chunks(data.test.Y, batch_size)) ])

                    print('''Epoch {} train accuracy: {}, test accuracy: {}. '''
                          '''{} states/sec on average, {} secs/epoch.'''.format(epoch, pf(train_accuracy_mean),
                                                                                pf(validation_accuracy_mean),
                                                                                pf(batch_size/np.mean(t_end)),
                                                                                pf(time.time() - t_epoch)))
                if epoch % save_every == 0 or epoch == epochs:
                    output_graph_def = graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(), ['input', 'output'])

                    with gfile.FastGFile(os.path.join(model_dir, model_name), 'w') as f:
                        f.write(output_graph_def.SerializeToString())

                t_epoch = time.time()
                t_end = []
                last_epoch = epoch

            
            i += 1
                
            if batch_size*i > data.train.X_len:
                epoch += 1
                i = 0

        q.close(cancel_pending_enqueues=True)
        
        print('Trained model saved to {}'.format(os.path.join(model_dir, model_name)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
    '''
    Trains the final layer of the Inception model. You must have
    collected the next to last layer states beforehand.
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'train_states',
        help='Folder with Inception states for training')
    parser.add_argument(
        'test_states',
        help='Folder with Inception states for testing')
    parser.add_argument(
        '--learning_rate',
        help='Learning rate',
        type=float,
        default=.0001)
    parser.add_argument(
        '--save_every',
        help='How often (in epochs) to save checkpoints',
        type=int,
        default=10)
    parser.add_argument(
        '--batch_size',
        help='Batch size for training',
        type=int,
        default=2048)
    parser.add_argument(
        '--hidden_size',
        help='Size of the ReLU hidden layer',
        type=int,
        default=2048)
    parser.add_argument(
        '--dropout',
        help='The probability to drop neurons, helps against overfitting',
        type=float,
        default=0.5)
    parser.add_argument(
        '--epochs',
        help='Maximum number of epochs before ending the training',
        type=int,
        default=100)
    parser.add_argument(
        '--print_every',
        help='Print training and validation accuracy every X steps',
        type=int,
        default=1)
    parser.add_argument(
        '--model_dir',
        help='Where to save the transfer learned model',
        default='.')
    parser.add_argument(
        '--perceptron',
        action='store_true',
        help='Train a perceptron as the last layer.')
    parser.add_argument(
        '--dense',
        action='store_true',
        help='Train a layer with hidden connections and dropout as the last layer.')
    parser.add_argument(
        '--lenet',
        action='store_true',
        help='5-LeNet as final layer.')
    parser.add_argument(
        '--filter_width',
        help='Width of convolutional filter',
        type=int,
        default=32)
    parser.add_argument(
        '--depth',
        help='Convnet depth',
        type=int,
        default=16)
    parser.add_argument(
        '--q_size',
        help='Capacity of FIFOQueue for loading training data.',
        type=int,
        default=100)
    parser.add_argument(
        '--use_dask',
        action='store_true',
        help='Read data from disk. Use if you cannot store everything in memory.')
    parser.add_argument(
        '--in_memory',
        help='How many vectors to hold in memory when using dask.',
        type=int,
        default=200000)
    parser.add_argument(
        '--dask_chunksize',
        help='The size of dask chunks on disk.',
        type=int,
        default=8*1024)

    args = parser.parse_args()

    assert trueXor(args.perceptron, args.dense, args.lenet), 'Specify one of perceptron, dense or lenet'
    
    learn(args.train_states, args.test_states, args.learning_rate,
          args.save_every, args.batch_size, args.hidden_size, args.dropout,
          args.epochs, args.print_every, args.model_dir, args.perceptron,
          args.dense, args.lenet, args.filter_width, args.depth,
          args.q_size, args.use_dask, args.in_memory, args.dask_chunksize)
