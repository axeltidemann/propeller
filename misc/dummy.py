import os

import tensorflow as tf
from keras.applications import Xception


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    net = Xception(include_top=False, weights='imagenet', pooling='max')
    
