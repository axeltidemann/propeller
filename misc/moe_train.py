# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

from __future__ import print_function
import argparse
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import graph_util
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from training_data import read_data
from utils import load_graph, chunks
    
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)
        
def flow(sess, models, x):
    return np.hstack([ sess.run(model, {'{}input:0'.format(h5): x}) for h5, model in models.iteritems() ])

def learn(data_folder, experts, learning_rate=.001, train_ratio=.8, validation_ratio=.1, test_ratio=.1, save_every=10, batch_size=2048, hidden_size=1024, dropout=.5, epochs=500, print_every=1, model_dir='.', perceptron=False, mem_ratio=.95):
    
    assert train_ratio + validation_ratio + test_ratio == 1, 'Train/validation/test ratios must sum up to 1'

    data = read_data(data_folder, train_ratio, validation_ratio, test_ratio)

    model_name = ('''transfer_classifier_moe_epochs_{}_batch_{}_ratios_{}_{}_{}_'''
                  '''learning_rate_{}'''.format(
                      epochs, batch_size,
                      train_ratio, validation_ratio,
                      test_ratio, learning_rate))
    if perceptron:
        model_name = '{}_perceptron.pb'.format(model_name)
    else:
        model_name = '{}_dropout_{}_hidden_size_{}.pb'.format(model_name, dropout, hidden_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        local_experts = {}
        
        for model in os.listdir(experts):
            print('Loading {}'.format(model))
            load_graph(os.path.join(args.experts, model))
            stripped = model[20:]
            h5 = stripped[:stripped.find('_')]# MESSY.
            local_experts[h5] = sess.graph.get_tensor_by_name('{}output:0'.format(h5))

        data.train._X = np.vstack([ flow(sess, local_experts, x) for x in chunks(data.train.X, batch_size) ])
        data.validation._X = flow(sess, local_experts, data.validation.X)
        data.test._X = flow(sess, local_experts, data.test.X)
            
        x = tf.placeholder('float', shape=[None, len(local_experts)*2], name='input')
        y_ = tf.placeholder('float', shape=[None, data.train.Y_features], name='target')
            
        if perceptron:
            W = weight_variable([len(local_experts)*2, data.train.Y_features], name='weights')
            b = bias_variable([data.train.Y_features], name='bias')

            logits = tf.matmul(x,W) + b
        else:
            W_in = weight_variable([len(local_experts)*2, hidden_size], name='weights_in')
            b_in = bias_variable([hidden_size], name='bias_in')

            hidden = tf.matmul(x,W_in) + b_in
            relu = tf.nn.relu(hidden)
            
            # is_training = tf.placeholder_with_default(False, shape=None)
            # bn = batch_norm(hidden, is_training=is_training, updates_collections=None)
            # relu = tf.nn.relu(bn)

            keep_prob = tf.placeholder_with_default([1.], shape=None)
            hidden_dropout = tf.nn.dropout(relu, keep_prob)

            W_out = weight_variable([hidden_size,data.train.Y_features], name='weights_out')
            b_out = bias_variable([data.train.Y_features], name='bias_out')

            logits = tf.matmul(relu,W_out) + b_out
            # logits = batch_norm(tf.matmul(relu,W_out) + b_out, is_training=is_training, updates_collections=None)

        y = tf.nn.softmax(logits, name='output')

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
        
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        sess.run(tf.initialize_all_variables())

        last_epoch = 0

        t_epoch = time.time()
        while data.train.epoch <= epochs:
            epoch = data.train.epoch
            batch_x, batch_y = data.train.next_batch(batch_size)
            
            t_start = time.time()
            feed_dict = {x: batch_x, y_: batch_y } if perceptron else {x: batch_x, y_: batch_y, keep_prob: dropout}
            # feed_dict = {x: batch_x, y_: batch_y } if perceptron else {x: batch_x, y_: batch_y, is_training: True }
            train_step.run(feed_dict=feed_dict)
            t_end = time.time() - t_start

            if epoch > last_epoch:

                if epoch % print_every == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_x,
                        y_: batch_y })

                    validation_accuracy = accuracy.eval(feed_dict={
                        x: data.validation.X,
                        y_: data.validation.Y })

                    print('''Epoch {} train accuracy: {}, validation accuracy: {}. '''
                          '''{} states/sec, {} secs/epoch.'''.format(epoch, train_accuracy,
                                                                     validation_accuracy, batch_size/t_end,
                                                                     time.time() - t_epoch))
                if epoch % save_every == 0 or epoch == epochs:
                    output_graph_def = graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(), ['input', 'output'])

                    with gfile.FastGFile(os.path.join(model_dir, model_name), 'w') as f:
                        f.write(output_graph_def.SerializeToString())

                t_epoch = time.time()
                last_epoch = epoch

        print('Trained model saved to {}'.format(os.path.join(model_dir, model_name)))

        if test_ratio > 0:
            test_accuracy = accuracy.eval(feed_dict={x: data.test.X, y_: data.test.Y })
            print('Evaluation on testing data: {}'.format(test_accuracy))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
    '''
    Trains the final layer of the Inception model. You must have
    collected the next to last layer states beforehand.
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_folder',
        help='Folder with Inception states for training')
    parser.add_argument(
        'experts',
        help='Folder with trained experts')
    parser.add_argument(
        '--learning_rate',
        help='Learning rate',
        type=float,
        default=.001)
    parser.add_argument(
        '--train_ratio',
        help='Train ratio',
        type=float,
        default=.8)
    parser.add_argument(
        '--validation_ratio',
        help='Validation ratio',
        type=float,
        default=.1)
    parser.add_argument(
        '--test_ratio',
        help='Test ratio',
        type=float,
        default=.1)
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
        default=1024)
    parser.add_argument(
        '--dropout',
        help='The probability to drop neurons, helps against overfitting',
        type=float,
        default=0.5)
    parser.add_argument(
        '--epochs',
        help='Maximum number of epochs before ending the training',
        type=int,
        default=500)
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
        action='store_true')
    parser.add_argument(
        '--mem_ratio',
        help='Ratio of memory to reserve on the GPU instance',
        type=float,
        default=.95)
    args = parser.parse_args()

    learn(args.data_folder, args.experts, args.learning_rate, args.train_ratio,
          args.validation_ratio, args.test_ratio, args.save_every, args.batch_size,
          args.hidden_size, args.dropout, args.epochs, args.print_every, args.model_dir,
          args.perceptron, args.mem_ratio)
