# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified by Axel.Tidemann@telenor.com. This inherits the original Google license.

"""
This collects the next to last layer states from the Inception model in order to do
transfer learning.
"""

import os.path
import re
import sys
import tarfile
import cStringIO as StringIO
import logging
import cPickle as pickle
import os
import time
import glob

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf

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

tf.app.flags.DEFINE_string('data_folder', '',
                           """Data folder""")

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

def save_states(folder):
  create_graph()
  with tf.Session() as sess:
    next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')
    states = []
    
    for jpg in glob.glob('{}/*.jpg'.format(folder)):
        image_data = gfile.FastGFile(jpg).read()
        hidden_layer = sess.run(next_last_layer,
                                {'DecodeJpeg/contents:0': image_data})
        hidden_layer = np.squeeze(hidden_layer)
        states.append(hidden_layer)

    states = np.vstack(states)
    np.save('{}/states'.format(folder), states)
          
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


def main(_):
  maybe_download_and_extract()
  save_states(FLAGS.data_folder)

if __name__ == '__main__':
  tf.app.run()
