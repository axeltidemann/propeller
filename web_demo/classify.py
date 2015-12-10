'''
Client that pushes an image to be processed by the Caffe framework, waits for the result.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import os.path
from cStringIO import StringIO
import getpass

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
parser.add_argument(
    '-un', '--username',
    help='username for the service')
parser.add_argument(
    '-p', '--password',
    help='password for the service')
args = parser.parse_args()

if args.username is None:
    args.username = raw_input('Username: ')
if args.password is None:
    args.password = getpass.getpass('Password: ')

URL = not os.path.isfile(args.img)

if URL:
    my_file = StringIO(args.img)
else:
    my_file = open(args.img, 'r')   

r = requests.post('http://{}/images/classify'.format(args.server),
                  data={'user': args.user}, files={'file': my_file},
                  auth=(args.username, args.password))

if URL:
    print requests.get('http://{}/images/prediction/{}/{}'.format(args.server, args.user, args.img),
                       auth=(args.username, args.password)).text
else:
    print r.text
