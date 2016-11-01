# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

from __future__ import print_function
import argparse
import time
import os
import threading

import numpy as np
import tensorflow as tf
from tensorflow.python.client import graph_util
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from training_data import read_data
    
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def learn(train_states, test_states, learning_rate=.0001, save_every=10, batch_size=2048, hidden_size=2048, dropout=.5,
          epochs=100, print_every=1, model_dir='.', perceptron=False, mem_ratio=.95, q_size=100, use_dask=False):

    data = read_data(train_states, test_states, use_dask)

    model_name = ('''transfer_classifier_epochs_{}_batch_{}_learning_rate_{}'''.format(
        epochs, batch_size, learning_rate))
    
    if perceptron:
        model_name = '{}_perceptron.pb'.format(model_name)
    else:
        model_name = '{}_dropout_{}_hidden_size_{}.pb'.format(model_name, dropout, hidden_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        q_x_in = tf.placeholder(tf.float32, shape=[batch_size, data.train.X_features])
        q_y_in = tf.placeholder(tf.int32, shape=[batch_size])
        y_real = tf.placeholder(tf.int32, shape=[None])
        keep_prob = tf.placeholder_with_default([1.], shape=None)
        
        q = tf.FIFOQueue(q_size, [tf.float32, tf.int32],
                         shapes=[ q_x_in.get_shape(), q_y_in.get_shape()])

        # q = tf.RandomShuffleQueue(q_size, int(q_size*.1), [tf.float32, tf.int32],
        #                  shapes=[ q_x_in.get_shape(), q_y_in.get_shape()])

        enqueue_op = q.enqueue([q_x_in, q_y_in])
        q_x_out, q_y_out = q.dequeue()

        x = tf.placeholder_with_default(q_x_out, shape=[None, data.train.X_features], name='input')
        y_ = tf.placeholder_with_default(q_y_out, shape=[None])

        if perceptron:
            W = weight_variable([data.train.X_features, data.train.Y_features])
            b = bias_variable([data.train.Y_features])

            logits = tf.matmul(x, W) + b
        else:
            W_in = weight_variable([data.train.X_features, hidden_size])
            b_in = bias_variable([hidden_size])

            hidden = tf.matmul(x, W_in) + b_in
            relu = tf.nn.relu(hidden)

            hidden_dropout = tf.nn.dropout(relu, keep_prob)

            W_out = weight_variable([hidden_size,data.train.Y_features])
            b_out = bias_variable([data.train.Y_features])

            logits = tf.matmul(relu, W_out) + b_out

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
        conv_out = tf.nn.conv2d(conv_in, _filter, stride, 'VALID')
        back = tf.squeeze(conv_out, squeeze_dims=[0,2], name='output')

        test_correct_prediction = tf.equal(tf.to_int32(tf.argmax(back, 1)), y_real)
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
        
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()

        def put():
            next_x, next_y = data.train.next_batch(batch_size)
            sess.run(enqueue_op, feed_dict={q_x_in: next_x, q_y_in: next_y})
        
        def putting():
            while not coord.should_stop():
                put()

        t0 = time.time()
        for _ in range(q_size): put()
        print('Pre-filling the queue with {} batches took {} seconds.'.format(q_size, time.time()-t0))
            
        data_thread = threading.Thread(target=putting)
        data_thread.start()

        last_epoch = 0
        epoch = 0
        
        t_epoch = time.time()
        i = 0 
        while epoch <= epochs:
            if batch_size*i > data.train.X_len:
                epoch += 1
                i = 0

            t_start = time.time()
            sess.run(train_step, feed_dict={keep_prob: dropout})
            t_end = time.time() - t_start

            if epoch > last_epoch:

                if epoch % print_every == 0:
                    batch_x, batch_y = data.train.next_batch(batch_size)
                    
                    train_accuracy_mean = train_accuracy.eval(feed_dict={
                        x: batch_x,
                        y_: batch_y })

                    validation_accuracy_mean = test_accuracy.eval(feed_dict={
                        x: data.test.X,
                        y_real: data.test.Y })

                    print('''Epoch {} train accuracy: {}, test accuracy: {}. '''
                          '''{} states/sec, {} secs/epoch.'''.format(epoch, train_accuracy_mean,
                                                                     validation_accuracy_mean, batch_size/t_end,
                                                                     time.time() - t_epoch))
                if epoch % save_every == 0 or epoch == epochs:
                    output_graph_def = graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(), ['input', 'output'])

                    with gfile.FastGFile(os.path.join(model_dir, model_name), 'w') as f:
                        f.write(output_graph_def.SerializeToString())

                t_epoch = time.time()
                last_epoch = epoch

            i += 1
            
        coord.request_stop()
        coord.join([data_thread])

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
        help='Train a perceptron instead of a network with a hidden layer.')
    parser.add_argument(
        '--mem_ratio',
        help='Ratio of memory to reserve on the GPU instance',
        type=float,
        default=.95)
    parser.add_argument(
        '--q_size',
        help='Capacity of FIFOQueue for loading training data.',
        type=int,
        default=100)
    parser.add_argument(
        '--use_dask',
        action='store_true',
        help='Read data from disk. Use if you cannot store everything in memory.')


    args = parser.parse_args()

    learn(args.train_states, args.test_states, args.learning_rate,
          args.save_every, args.batch_size, args.hidden_size, args.dropout,
          args.epochs, args.print_every, args.model_dir, args.perceptron, args.mem_ratio,
          args.q_size, args.use_dask)
