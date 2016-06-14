# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html

"""
This uses the Inception model, reads from the next-to-last layer and uses this as input
to a trained classifier on top.
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
import json
from collections import namedtuple
import tempfile
from contextlib import contextmanager
import time
import math

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
from sklearn import svm
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

tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# Axel defined input arguments
tf.app.flags.DEFINE_string('redis_server', 'localhost', 
                           """Redis server address""")
tf.app.flags.DEFINE_integer('redis_port', 6379,
                            """Redis server port""")
tf.app.flags.DEFINE_string('redis_queue', 'classify',
                           """Redis queue to read images from""")

tf.app.flags.DEFINE_string('transfer_dir', '.',
                           """Path to transfer_classifier.pb, which contains the transfer learned model.""")
tf.app.flags.DEFINE_string('mapping', 'mapping.txt',
                           """A mapping from node IDs to readable text""")


Task = namedtuple('Task', 'queue value')
Specs = namedtuple('Specs', 'group path')
Result = namedtuple('Result', 'OK predictions computation_time')

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

logging.getLogger().setLevel(logging.INFO)

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

def load_graph(path, filename):
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join(
      path, filename), 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    
def classify_images(mapping):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1./4)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    load_graph(FLAGS.model_dir, 'classify_image_graph_def.pb')
    inception_next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')
    
    load_graph(FLAGS.transfer_dir, 'transfer_classifier.pb')
    transfer_predictor = sess.graph.get_tensor_by_name('output:0')
    
    r_server = redis.StrictRedis(FLAGS.redis_server, FLAGS.redis_port)
    
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
        hidden_layer = sess.run(inception_next_last_layer,
                                {'DecodeJpeg/contents:0': image_data})
        
        predictions = sess.run(transfer_predictor, {'input:0': np.atleast_2d(np.squeeze(hidden_layer)) })
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        
        endtime = time.time()

        result = Result(True, 
                        [ (mapping[str(node_id)], predictions[node_id]) for node_id in top_k ], 
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
  with open(FLAGS.mapping) as f:
    mapping = json.load(f)
  classify_images(mapping)

if __name__ == '__main__':
  tf.app.run()
