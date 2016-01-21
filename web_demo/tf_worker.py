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

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

import os.path
import re
import sys
import tarfile
#import argparse
from collections import namedtuple
import cStringIO as StringIO
import logging
import cPickle as pickle
import os
import tempfile
from contextlib import contextmanager
import time

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
import redis
import requests
from wand.image import Image
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
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

tf.app.flags.DEFINE_string('redis_server', '', 
                           """Redis server address""")
tf.app.flags.DEFINE_integer('redis_port', 6379,
                            """Redis server port""")
tf.app.flags.DEFINE_string('redis_queue', 'classify',
                           """Redis queue to read images from""")

Task = namedtuple('Task', 'queue value')
Specs = namedtuple('Specs', 'group path')
Result = namedtuple('Result', 'OK predictions computation_time')

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')

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
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.iteritems():
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
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

@contextmanager
def convert_to_jpg(data):
  tmp = tempfile.NamedTemporaryFile(delete=False)

  with Image(file=StringIO.StringIO(data)) as img:
    if img.format != 'JPEG':
      logging.info('Converting {} to JPEG.'.format(img.format))
      img.format = 'JPEG'
    img.save(tmp)
    
  tmp.close()
  yield tmp.name
  os.remove(tmp.name)

def classify_images():
  create_graph()
  node_lookup = NodeLookup()
  # 4 instances running in parallel on g2.2xlarge seems to be the magic number.
  # If running more instances, memcpy errors will be thrown after some time.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1./4) 

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    r_server = redis.StrictRedis(FLAGS.redis_server, FLAGS.redis_port)
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    while True:
      task = Task(*r_server.brpop(FLAGS.redis_queue))
      specs = Specs(**pickle.loads(task.value))
      logging.info(specs)
      result_key = 'archive:{}:{}'.format(specs.group, specs.path)

      try: 
        response = requests.get(specs.path, timeout=10)
        with convert_to_jpg(response.content) as jpg:
          image_data = gfile.FastGFile(jpg).read()

        starttime = time.time()
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
        endtime = time.time()

        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        result = Result(True, 
                        [ (node_lookup.id_to_string(node_id), predictions[node_id]) for node_id in top_k ], 
                        endtime - starttime)

        r_server.hmset(result_key, result._asdict())
        r_server.zadd('archive:{}:category:{}'.format(specs.group, result.predictions[0][0]),
                      result.predictions[0][1], specs.path)
        # The publishing was only added since AWS ElastiCache does not support subscribing to keyspace notifications.
        r_server.publish('latest', pickle.dumps({'path': specs.path, 'group': specs.group, 
                                                 'category': result.predictions[0][0], 'value': float(result.predictions[0][1])}))
        logging.info(result)
      except Exception as e:
        logging.error('Something went wrong when classifying the image: {}'.format(e))
        r_server.hmset(result_key, {'OK': False})
          
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
  classify_images()

if __name__ == '__main__':
  tf.app.run()
