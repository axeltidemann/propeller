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

from wand.image import Image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--URL_folder',
    help='Folder with URL text files')
args = parser.parse_args()

def save_to_jpg(data, folder_name):
    filename = '{}/{}.jpg'.format(folder_name, uuid.uuid4())
    with Image(file=StringIO.StringIO(data)) as img:
        if img.format != 'JPEG':
            logging.info('Converting {} to JPEG.'.format(img.format))
            img.format = 'JPEG'
        img.save(filename=filename)
            
def get(filename):
    print 'Downloading from {}'.format(filename)
    folder_name = filename.replace('.txt','')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(filename) as f:
        for line in f:
            try:
                response = requests.get(line, timeout=10)
                save_to_jpg(response.content, folder_name)
            except Exception as e:
                print e
                
    print '{} done.'.format(filename)

pool = mp.Pool()
filenames = glob.glob('{}/*.txt'.format(args.URL_folder))
pool.map(get, filenames)
