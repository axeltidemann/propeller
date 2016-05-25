'''
Parses the Kaidee gzipped JSON files and writes a corresponding file with URLs.

Author: Axel.Tidemann@telenor.com
'''

from __future__ import print_function
import argparse
import glob
import gzip
import json
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='Folder with folders of gzipped JSON files.')
parser.add_argument(
    'target',
    help='Where to put the URL files.')
args = parser.parse_args()

for folder in filter(lambda x: os.path.isdir(os.path.join(args.source, x)), os.listdir(args.source)):
    print('Parsing folder {}'.format(folder))
    for gz_file in glob.glob('{}/*.gz'.format(os.path.join(args.source, folder))):
        with gzip.open(gz_file) as _file:
            for line in _file:
                data = json.loads(line)
                category = data['category']['id']
                try:
                    with open('{}/{}.txt'.format(args.target, category), 'a+') as URL_file:
                        for photo in data['photos']:
                            print(photo['large'], file=URL_file)
                except:
                    pass
