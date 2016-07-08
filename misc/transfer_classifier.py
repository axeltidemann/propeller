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

from ast import literal_eval as make_tuple
from tensorflow.python.platform import gfile

import blosc
from aqbc_utils import hash_bottlenecks

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

tf.app.flags.DEFINE_string('classifier', '',
                           """Path to the transfer learned model.""")
tf.app.flags.DEFINE_string('mapping', 'mapping.txt',
                           """A mapping from node IDs to readable text""")

tf.app.flags.DEFINE_string('hashing', True,
                           """whether to hash or not in addition to classification""")


Task = namedtuple('Task', 'queue value')
Specs = namedtuple('Specs', 'group path res_q')
Result = namedtuple('Result', 'OK predictions computation_time path')

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

def load_graph(path):
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(path, 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def classify_images(mapping):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1./4)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    load_graph(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'))
    inception_next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')

    load_graph(FLAGS.classifier)
    transfer_predictor = sess.graph.get_tensor_by_name('output:0')

    r_server = redis.StrictRedis(FLAGS.redis_server, FLAGS.redis_port)

    if FLAGS.hashing:
      R_c = r_server.get('hashing:R')
      if R_c == None:
        logging.info('Did not find any Rotation matrix in redis at key: <hashing:R>. Continuing without hashing.')
        FLAGS.hashing = False
      else:
        R_u = blosc.decompress(R_c)
        R = np.fromstring(R_u, dtype=np.float64)
        bits = R.shape[0]/2048
        R = R.reshape(2048, bits)
        R = np.transpose(R)
    
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
                        endtime - starttime,
                        specs.path)
        
        value = result._asdict()

        hidden_layer = hidden_layer.reshape(2048,1)
       
        if FLAGS.hashing:
          _, c = hash_bottlenecks(R, hidden_layer)
          r_server.sadd("hashing:codes:" + c[0].bin, result_key)
          r_server.hmset(result_key, {'hash': c[0].bin})
          value['hash'] = c[0].bin

        r_server.hmset(result_key, value)

        # for demo
        last_key = 'archive:{}:{}'.format(specs.group, 'lastprediction')
        r_server.hmset(last_key, result._asdict())

        h_s_packed = blosc.compress(hidden_layer.tostring(), typesize=8, cname='zlib')

        r_server.hset('archive:{}:category:{}'.format(specs.group, result.predictions[0][0]),
                      specs.path, h_s_packed)

        # push result on result queue
        preds = {}
        for node_id in top_k:
          preds[str(mapping[str(node_id)])] = str(predictions[node_id])

        json_blob = {
          'predictions':preds,
          'path':specs.path,
          'hidden_states':h_s_packed
        }

        if specs.res_q != "":
          r_server.rpush(specs.res_q, json.dumps(json_blob, ensure_ascii=False, encoding="utf-8"))

        logging.info(result)
      except Exception as e:
        print "exception*****************", e
        logging.error('Something went wrong when classifying the image: {}'.format(e))
        r_server.hmset(result_key, {'OK': False})

def send_kaidee_data(r_server, specs, result):

    # create redis key from image path
    full_url = specs.path.split('//')
    url_path = len(full_url)>1 and full_url[1] or full_url[0]
    kaidee_result_key = url_path.split('/', 1)[1]

    # Set to result to Redis with key from image path
    r_server.hmset(kaidee_result_key, result._asdict())

    # Publish predictions result to classify channel via Redis PubSub
    predictions_dict = dict((x, y) for x, y in result.predictions)
    r_server.publish('classify', pickle.dumps({'path': specs.path, 'group': specs.group,
                                             'predictions': predictions_dict}))


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
