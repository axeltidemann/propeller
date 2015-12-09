'''
Client that pushes an image to be processed by the Caffe framework, waits for the result.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import os.path
from cStringIO import StringIO

import requests

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'img',
    help='URL to an image, or a local text file with URLs to images')
parser.add_argument(
    '-s', '--server',
    help='the server address',
    default='localhost')
parser.add_argument(
    '-u', '--user',
    help='user for the image',
    default='web')
args = parser.parse_args()

URL = not os.path.isfile(args.img)

if URL:
    my_file = StringIO(args.img)
else:
    my_file = open(args.img, 'r')   

r = requests.post('http://{}/images/classify'.format(args.server),
                  data={'user': args.user}, files={'file': my_file})

if URL:
    print requests.get('http://{}/images/prediction/{}/{}'.format(args.server, args.user, args.img)).text
else:
    print r.text
