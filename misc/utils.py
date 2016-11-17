import datetime
import subprocess
import os
import tarfile
import sys

from six.moves import urllib
import pandas as pd

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz'
# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def save_df(name, df):
    df.to_hdf('{}_{}.h5'.format(name, datetime.datetime.now().isoformat()),
              key='result', mode='w', format='table', complib='blosc', complevel=9)


def trueXor(*args):
    return sum(args) == 1
    
def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l. """
    for i in xrange(0, len(chunkable), n):
        yield chunkable[i:i+n]
        
def flatten(lst):
    result = []
    for element in lst:
        if hasattr(element, '__iter__'):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result
        
def safe_filename(filename):
    import base64 # With multiprocessing, this needs to be imported here.
    """ Base64 encodes the string, so you can safely use is as a filename. """
    return base64.urlsafe_b64encode(filename)


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def pretty_float(f):
    return '{0:.2f}'.format(f)


def load_graph(path):
    from tensorflow.python.platform import gfile
    import tensorflow as tf
    
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with gfile.FastGFile(path, 'r') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract(model_dir):
    """Download and extract model tar file."""
    dest_directory = model_dir
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
