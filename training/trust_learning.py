# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

from __future__ import print_function
import argparse
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../misc')))

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

def learn(train_states, test_states, model, learning_rate=.0001, save_every=10, batch_size=2048, hidden_size=2048, dropout=.5,
          epochs=500, print_every=1, model_dir='.', perceptron=False, mem_ratio=.95):

    data = read_data(train_states, test_states)

    model_name = ('''trust_classifier_epochs_{}_batch_{}_learning_rate_{}'''.format(
        epochs, batch_size, learning_rate))
    
    if perceptron:
        model_name = '{}_perceptron.pb'.format(model_name)
    else:
        model_name = '{}_dropout_{}_hidden_size_{}.pb'.format(model_name, dropout, hidden_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        load_graph(model)
        transfer_predictor = sess.graph.get_tensor_by_name('output:0')

        # Evaluate the model on the training and test data, for training and testing.
        data.train._X = np.vstack([ sess.run(transfer_predictor, {'input:0': chunk }) for chunk in chunks(data.train.X, 10) ])
        answer = tf.equal(tf.argmax(data.train.X, 1), tf.argmax(data.train.Y,1))
        data.train._Y = tf.one_hot(tf.to_int32(answer), depth=2).eval()

        data.test._X = sess.run(transfer_predictor, {'input:0': data.test.X })
        answer = tf.equal(tf.argmax(data.test.X, 1), tf.argmax(data.test.Y,1))
        data.test._Y = tf.one_hot(tf.to_int32(answer), depth=2).eval()

        x = tf.placeholder('float', shape=[None, data.train.X_features], name='input_b')
        y_ = tf.placeholder('float', shape=[None, data.train.Y_features], name='target')

        if perceptron:
            W = weight_variable([data.train.X_features, data.train.Y_features], name='weights')
            b = bias_variable([data.train.Y_features], name='bias')

            logits = tf.matmul(x,W) + b
        else:
            W_in = weight_variable([data.train.X_features, hidden_size], name='weights_in')
            b_in = bias_variable([hidden_size], name='bias_in')

            hidden = tf.matmul(x,W_in) + b_in
            relu = tf.nn.relu(hidden)

            keep_prob = tf.placeholder_with_default([1.], shape=None)
            hidden_dropout = tf.nn.dropout(relu, keep_prob)

            W_out = weight_variable([hidden_size,data.train.Y_features], name='weights_out')
            b_out = bias_variable([data.train.Y_features], name='bias_out')

            logits = tf.matmul(relu,W_out) + b_out

        # Loss & train
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        # Evaluation
        y = tf.nn.softmax(logits, name='output_b')
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
        sess.run(tf.initialize_all_variables())

        last_epoch = 0

        t_epoch = time.time()
        while data.train.epoch <= epochs:
            epoch = data.train.epoch
            batch_x, batch_y = data.train.next_batch(batch_size)

            t_start = time.time()
            feed_dict = {x: batch_x, y_: batch_y } if perceptron else {x: batch_x, y_: batch_y, keep_prob: dropout}
            train_step.run(feed_dict=feed_dict)
            t_end = time.time() - t_start

            if epoch > last_epoch:

                if epoch % print_every == 0:
                    train_accuracy_mean = accuracy.eval(feed_dict={
                        x: batch_x,
                        y_: batch_y })

                    validation_accuracy_mean = accuracy.eval(feed_dict={
                        x: data.test.X,
                        y_: data.test.Y })

                    print('''Epoch {} train accuracy: {}, test accuracy: {}. '''
                          '''{} states/sec, {} secs/epoch.'''.format(epoch, train_accuracy_mean,
                                                                     validation_accuracy_mean, batch_size/t_end,
                                                                     time.time() - t_epoch))
                if epoch % save_every == 0 or epoch == epochs:
                    output_graph_def = graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(), ['input_b', 'output_b'])

                    with gfile.FastGFile(os.path.join(model_dir, model_name), 'w') as f:
                        f.write(output_graph_def.SerializeToString())

                t_epoch = time.time()
                last_epoch = epoch

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
        'model',
        help='The trained model')
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
        action='store_true',
        help='Train a perceptron instead of a network with a hidden layer.')
    parser.add_argument(
        '--mem_ratio',
        help='Ratio of memory to reserve on the GPU instance',
        type=float,
        default=.95)
    args = parser.parse_args()

    learn(args.train_states, args.test_states, args.model, args.learning_rate,
          args.save_every, args.batch_size, args.hidden_size, args.dropout,
          args.epochs, args.print_every, args.model_dir, args.perceptron, args.mem_ratio)
