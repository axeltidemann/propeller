# Copyright 2016 Telenor ASA, Author: Axel Tidemann

from __future__ import print_function
import argparse

import ipdb
from pymongo import MongoClient

parser = argparse.ArgumentParser(description=
'''
Downloads stuff marked as "sex product".
''',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--limit',
    help='Number of files to download',
    type=int,
    default=10000)
parser.add_argument(
    '--filename',
    help='Name of file',
    default='snakes.txt')
args = parser.parse_args()

client = MongoClient("mongodb://hackathon:spun6-Dundee@ds017620-a0.mlab.com:17620,ds017620-a1.mlab.com:17620/ims-hackathon2?replicaSet=rs-ds017620")
db = client['ims-hackathon2']
adCollection = db['Ad']

i = 0
with open(args.filename, 'w') as _file:
    for ad in adCollection.find({"statusDetails": "sex product"}):
        if 'medium_view' in ad.keys():
            print('https://cdn.shoppingkaki.com/{}'.format(ad['medium_view']), file=_file)
            i += 1

        if i == args.limit:
            break

print('{} images downloaded.'.format(i))
