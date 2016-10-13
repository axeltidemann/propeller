# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import time
import random

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

from collect_states import transformations, adjustments, random_crop

parser = argparse.ArgumentParser(description='''
Applies a range of parameter values to different image adjustments. To see
the effects of the default values set for data augmentation.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'jpg',
    help='File to apply image adjustments to.')
parser.add_argument(
    'target',
    help='Where to put the resulting files.')
parser.add_argument(
    '--brightness_lower',
    default=.02,
    type=float)
parser.add_argument(
    '--brightness_upper',
    default=0.3,
    type=float)
parser.add_argument(
    '--saturation_lower',
    default=0,
    type=float)
parser.add_argument(
    '--saturation_upper',
    default=5,
    type=float)
parser.add_argument(
    '--hue_lower',
    default=-1,
    type=float)
parser.add_argument(
    '--hue_upper',
    default=1,
    type=float)
parser.add_argument(
    '--contrast_lower',
    default=.2,
    type=float)
parser.add_argument(
    '--contrast_upper',
    default=2,
    type=float)
parser.add_argument(
    '--n',
    default=20,
    help='Number of samples to generate.',
    type=int)
args = parser.parse_args()

def encode(x):
    return tf.image.encode_jpeg(x).eval()

with tf.Session() as sess:
    raw = gfile.FastGFile(args.jpg).read()
    jpg = tf.image.decode_jpeg(raw, channels=3)

    n = args.n

    times = []
    for i in np.linspace(args.brightness_lower,args.brightness_upper,n):
        with open('bright_{}.jpg'.format(i), 'w') as _file:
            t0 = time.time()
            bright = tf.image.adjust_brightness(jpg, i)
            back = encode(bright)
            times.append(time.time()-t0)
            _file.write(back)
    print 'Brightness average time: {} seconds'.format(np.mean(times))

    times = []            
    for i in np.linspace(args.saturation_lower,args.saturation_upper,n):
        with open('saturation_{}.jpg'.format(i), 'w') as _file:
            t0 = time.time()
            saturation = tf.image.adjust_saturation(jpg, i)
            back = encode(saturation)
            times.append(time.time()-t0)
            _file.write(back)
    print 'Saturation average time: {} seconds'.format(np.mean(times))

    times = []            
    for i in np.linspace(args.hue_lower,args.hue_upper,n):
        with open('hue_{}.jpg'.format(i), 'w') as _file:
            t0 = time.time()
            hue = tf.image.adjust_hue(jpg, i)
            back = encode(hue)
            times.append(time.time()-t0)
            _file.write(back)
    print 'Hue average time: {} seconds'.format(np.mean(times))

    times = []            
    for i in np.linspace(args.contrast_lower,args.contrast_upper,n):
        with open('contrast_{}.jpg'.format(i), 'w') as _file:
            t0 = time.time()
            contrast = tf.image.adjust_contrast(jpg, i)
            back = encode(contrast)
            times.append(time.time()-t0)
            _file.write(back)
    print 'Contrast average time: {} seconds'.format(np.mean(times))

    with open('leftright.jpg', 'w') as _file:
        t0 = time.time()
        leftright = tf.image.flip_left_right(jpg)
        back = encode(leftright)
        print 'Flip left right time: {} seconds'.format(time.time()-t0)
        _file.write(back)
        
    times = []
    for i in range(n):
        with open('random_crop_{}.jpg'.format(i), 'w') as _file:
            t0 = time.time()
            crop = random_crop(jpg)
            back = encode(crop)
            times.append(time.time() - t0)
            _file.write(back)
    print 'Random cropping average time: {} seconds'.format(np.mean(times))

    times = []
    for i in range(n):
        with open('augmented_{}.jpg'.format(i), 'w') as _file:
            t0 = time.time()
            augmented = random.choice(adjustments)(transformations(jpg))
            back = encode(augmented)
            times.append(time.time() - t0)
            _file.write(back)
    print 'Data augmentation average time: {} seconds'.format(np.mean(times))
