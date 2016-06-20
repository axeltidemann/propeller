# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

"""
This collects the next to last layer states from the Inception model in order to do
transfer learning.
"""

import os.path
import sys
import tarfile
import logging
import os
import glob
from random import shuffle

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
import pandas as pd

# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.

# this is the same as namedtuple
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

tf.app.flags.DEFINE_string('source', '',
                           """Folder with images""")
tf.app.flags.DEFINE_string('target', '',
                           """Where to put the states file""")
tf.app.flags.DEFINE_integer('limit', 10000,
                           """Maximum amount of images to process per folder""")
tf.app.flags.DEFINE_integer('mem_ratio', 1,
                           """1/x ratio of memory to reserve on the GPU instance""")


# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

logging.getLogger().setLevel(logging.INFO)

def create_graph():
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
def save_states(source, target, limit, mem_ratio):
  create_graph()

  # 3.95G GPU RAM actually free on AWS, reports as 4G. A little legroom is needed.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1./(mem_ratio + .5)) 
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')
    states = []

    images = glob.glob('{}/*.jpg'.format(source))
    shuffle(images)
    images = images[:limit]
    
    for jpg in images:
      image_data = gfile.FastGFile(jpg).read()
      hidden_layer = sess.run(next_last_layer,
                              {'DecodeJpeg/contents:0': image_data})
      hidden_layer = np.squeeze(hidden_layer)
      states.append(hidden_layer)

      # Experimental data augmentation below - shitty conversion troubles. Should be an input option, as it will
      # drastically increase the amount of data.
      
      # flipped = tf.image.encode_jpeg(tf.image.flip_left_right(tf.convert_to_tensor(image_data)), format='rgb')
      
      # for img in [ image_data, flipped ]:
      #   hidden_layer = sess.run(next_last_layer,
      #                           {'DecodeJpeg/contents:0': img})
      #   hidden_layer = np.squeeze(hidden_layer)
      #   states.append(hidden_layer)

      #   # Brightness
      #   hidden_layer = sess.run(next_last_layer,
      #                           {'DecodeJpeg/contents:0': tf.image.encode_jpeg(tf.image.random_brightness(tf.image.decode_jpeg(img, channels=3), max_delta=63), format="rgb")})
      #   hidden_layer = np.squeeze(hidden_layer)
      #   states.append(hidden_layer)

      #   # Contrast
      #   hidden_layer = sess.run(next_last_layer,
      #                           {'DecodeJpeg/contents:0': tf.image.random_contrast(img, lower=.2, upper=1.8)})
      #   hidden_layer = np.squeeze(hidden_layer)
      #   states.append(hidden_layer)

      #   # Saturation
      #   hidden_layer = sess.run(next_last_layer,
      #                           {'DecodeJpeg/contents:0': tf.image.random_saturation(img, lower=0, upper=1.)})
      #   hidden_layer = np.squeeze(hidden_layer)
      #   states.append(hidden_layer)

    df = pd.DataFrame(data={'state': states}, index=images)
    df.index.name='filename'

    h5name = '{}/{}.h5'.format(target, os.path.basename(source))
    with pd.HDFStore(h5name, 'w') as store:
      store['data'] = df

def main(_):
  maybe_download_and_extract()
  save_states(FLAGS.source, FLAGS.target, FLAGS.limit, FLAGS.mem_ratio)

if __name__ == '__main__':
  tf.app.run()
