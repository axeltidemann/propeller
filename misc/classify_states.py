# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.

"""
Classifies a folder with images, creates a histogram.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from six.moves import urllib
import tensorflow as tf
import pandas as pd
import ipdb

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('states', '',
                           """Absolute path to HDF5 file with states""")
tf.app.flags.DEFINE_integer('num_top_classes', 5,
                            """Print these top classes.""")
tf.app.flags.DEFINE_string('target', '',
                            """Where to put the resulting files.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_states(h5_file, top_k, target):

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    categories = defaultdict(int)
    confidences = defaultdict(list)
    files = defaultdict(list)
    states = defaultdict(list)
    
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    data = pd.read_hdf(h5_file, 'data')

    predictions = []
    for state, filename in zip(data.state, data.index):
      state.shape = (1, 1, 1, 2048)
      predictions = sess.run(softmax_tensor,
                                  {'pool_3:0': state})
      predictions = np.squeeze(predictions)
      category = predictions.argsort()[-1]
      categories[category] += 1
      files[category].append(filename)
      states[category].append(np.squeeze(state))
    
    category_name = os.path.basename(h5_file).replace('.h5','')
      
    node_lookup = NodeLookup()
    top_index = np.array(categories.values()).argsort()[-top_k:]

    labels = ['']*len(categories)

    keys = categories.keys()
    
    for t in top_index:
      key = keys[t]
      labels[t] = node_lookup.id_to_string(key).split(',')[0]

      with open('{}/{}.{}.txt'.format(target, category_name, key), 'w') as _file:
        for image in files[key]:
          print(image, file=_file)

      df = pd.DataFrame(data={'state': states[key]}, index=files[key])
      df.index.name='filename'
      
      h5name = '{}/{}.{}.h5'.format(target, category_name, key)
      with pd.HDFStore(h5name, 'w') as store:
        store['data'] = df
    
    sns.barplot(range(len(categories)), categories.values())

    plt.xticks(range(len(categories)), labels)
    plt.title('Category {}'.format(category_name))
    plt.savefig('{}/{}.png'.format(target, category_name), dpi=300)

    
      # # Creates node ID --> English string lookup.
      # node_lookup = NodeLookup()

      # top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
      # for node_id in top_k:
      #   human_string = node_lookup.id_to_string(node_id)
      #   score = predictions[node_id]
      #   print('%s (score = %.5f)' % (human_string, score))


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
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  run_inference_on_states(FLAGS.states, FLAGS.num_top_classes, FLAGS.target)

if __name__ == '__main__':
  tf.app.run()
