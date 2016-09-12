''' 
Downloads image files from URLs.

Author: Axel.Tidemann@telenor.com
'''

import requests
import glob
import multiprocessing as mp
import argparse
import cStringIO as StringIO
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
    help='Where to put the images, a folder will be created for each category.')
parser.add_argument(
    '--limit',
    help='Limit the number of files to download',
    type=int,
    default=10000)
parser.add_argument(
    '--timeout',
    help='Number of seconds to wait for the server',
    type=int,
    default=10000)
args = parser.parse_args()

def save_to_jpg(url, data, folder_name):
    #filename = '{}/{}.jpg'.format(folder_name, uuid.uuid4())
    filename = '{}/{}.jpg'.format(folder_name, url.split("/")[-1])
    with Image(file=StringIO.StringIO(data)) as img:
        if img.format != 'JPEG':
            logging.info('Converting {} to JPEG.'.format(img.format))
            img.format = 'JPEG'
        img.save(filename=filename)
            
def get(target_folder, limit, timeout, source_filename):
    
    print 'Downloading from {}'.format(source_filename)
    category_name = os.path.basename(source_filename).replace('.txt', '')
    category_folder = '{}/{}'.format(target_folder, os.path.basename(os.path.normpath(source_filename)).replace('.txt',''))
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

    already_stored = len(os.listdir(category_folder))
    print 'Category {} has {} files, aiming for {} more'.format(category_name, already_stored, limit - already_stored)

    if already_stored == limit:
        return
    
    with open(source_filename) as _file:
        download_counter = already_stored
        line_counter = 0
        
        for line in _file:
            if line_counter >= already_stored:
                try:
                    response = requests.get(line.rstrip(), timeout=timeout)
                    save_to_jpg(line.rstrip(), response.content, category_folder)
                    
                    download_counter += 1

                    if download_counter == limit:
                        break

                except Exception as e:
                    print e
                    pass
                    
            line_counter += 1
                
    print 'Category {} done, {} files downloaded, {} in total.'.format(category_name, download_counter - already_stored, download_counter)

pool = mp.Pool()
par_get = partial(get, args.target, args.limit, args.timeout)
filenames = glob.glob('{}/*.txt'.format(args.source))
pool.map(par_get, filenames)
