# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

'''
Trains the final layer of the Inception model. You must have
collected the next to last layer states beforehand.
'''

from __future__ import print_function
import argparse
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import graph_util
from tensorflow.python.platform import gfile

from training_data import read_data

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states for training')
parser.add_argument(
    '--learning_rate',
    help='Learning rate',
    type=float,
    default=.0001)
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
    '--checkpoint_dir',
    help='Where to save the checkpoints',
    default='cv/')
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
args = parser.parse_args()

assert args.train_ratio + args.validation_ratio + args.test_ratio == 1, 'Train/validation/test ratios must sum up to 1'

data = read_data(args.data_folder, args.train_ratio, args.validation_ratio, args.test_ratio)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

model_name = ('''transfer_classifier_epochs_{}_batch_{}_ratios_{}_{}_{}_'''
              '''learning_rate_{}_dropout_{}_hidden_size_{}.pb'''.format(args.epochs, args.batch_size,
                                                              args.train_ratio, args.validation_ratio,
                                                              args.test_ratio, args.learning_rate,
                                                                         args.dropout, args.hidden_size))
    
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

    
with tf.Session() as sess:
    x = tf.placeholder('float', shape=[None, data.train.X_features], name='input')
    y_ = tf.placeholder('float', shape=[None, data.train.Y_features], name='target')

    W_in = weight_variable([data.train.X_features, args.hidden_size], name='weights_in')
    b_in = bias_variable([args.hidden_size], name='bias_in')

    hidden = tf.nn.relu(tf.matmul(x,W_in) + b_in)

    keep_prob = tf.placeholder_with_default([1.], shape=None)
    hidden_dropout = tf.nn.dropout(hidden, keep_prob)

    W_out = weight_variable([args.hidden_size,data.train.Y_features], name='weights_out')
    b_out = bias_variable([data.train.Y_features], name='bias_out')

    logits = tf.matmul(hidden_dropout,W_out) + b_out
    
    y = tf.nn.softmax(logits, name='output')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)

    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver()
    
    sess.run(tf.initialize_all_variables())

    last_i = 0

    t_epoch = time.time()
    while data.train.epoch <= args.epochs:
        i = data.train.epoch
        batch_x, batch_y = data.train.next_batch(args.batch_size)
        
        t_start = time.time()
        train_step.run(feed_dict={x: batch_x,
                                  y_: batch_y,
                                  keep_prob: args.dropout})
        t_end = time.time() - t_start
        
        if i % args.save_every == 0 and last_i != i:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), ['input', 'output'])

            with gfile.FastGFile(os.path.join(args.model_dir, model_name), 'w') as f:
                f.write(output_graph_def.SerializeToString())

            saver.save(sess, args.checkpoint_dir + 'model.ckpt',
                       global_step=i+1)
        
        if i % args.print_every == 0 and last_i != i:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_x,
                y_: batch_y })

            validation_accuracy = accuracy.eval(feed_dict={
                x: data.validation.X,
                y_: data.validation.Y })
            
            print('''Epoch {} train accuracy: {}, validation accuracy: {}. '''
                  '''{} states/sec, {} secs/epoch.'''.format(i, train_accuracy,
                                                             validation_accuracy, args.batch_size/t_end,
                                                             time.time() - t_epoch))
            t_epoch = time.time()
            last_i = i

    print('Trained model saved to {}'.format(os.path.join(args.model_dir, model_name)))

    if args.test_ratio > 0:
        test_accuracy = accuracy.eval(feed_dict={x: data.test.X, y_: data.test.Y })
        print('Evaluation on testing data: {}'.format(test_accuracy))
