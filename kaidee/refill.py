'''
Refills the images folder, based on how many images there already exists in the states folder.
'''

from __future__ import print_function
import argparse
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'states',
    help='Location of states files.')
parser.add_argument(
    'URLs',
    help='Location of URL files')
parser.add_argument(
    '--limit',
    help='Limit the number of files to download',
    type=float,
    default=np.inf)
args = parser.parse_args()

for h5_file in glob.glob('{}/*.h5'.format(args.states)):
    data = pd.read_hdf(h5_file, 'data')
    print('{}: {} images'.format(os.path.basename(h5_file), len(data)))
