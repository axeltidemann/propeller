''' 
Downloads image files from URLs.

Author: Axel.Tidemann@telenor.com
'''

import requests
import glob
import multiprocessing as mp
import argparse
import cStringIO as StringIO
import os
import sys
import uuid
import logging
from functools import partial
import os

from wand.image import Image
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='Folder with URL text files')
parser.add_argument(
    'target',
    help='Where to put the images, a folder will be created for URL text file.')
parser.add_argument(
    '--limit',
    help='Limit the number of files to download',
    type=float,
    default=np.inf)
args = parser.parse_args()

def save_to_jpg(data, folder_name):
    filename = '{}/{}.jpg'.format(folder_name, uuid.uuid4())
    with Image(file=StringIO.StringIO(data)) as img:
        if img.format != 'JPEG':
            logging.info('Converting {} to JPEG.'.format(img.format))
            img.format = 'JPEG'
        img.save(filename=filename)
            
def get(target_folder, limit, source_filename):
    print 'Downloading from {}'.format(source_filename)
    category_folder = '{}/{}'.format(target_folder, os.path.basename(os.path.normpath(source_filename)).replace('.txt',''))
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    with open(source_filename) as _file:
        i = 0
        for line in _file:
            try:
                response = requests.get(line.rstrip(), timeout=10)
                save_to_jpg(response.content, category_folder)
                i += 1

                if i > limit:
                    break
                    
            except Exception as e:
                pass
                
    print '{} done, {} files downloaded.'.format(source_filename, i)

pool = mp.Pool()
par_get = partial(get, args.target, args.limit)
filenames = glob.glob('{}/*.txt'.format(args.source))
pool.map(par_get, filenames)
