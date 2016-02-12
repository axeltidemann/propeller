'''
Starts the collection of states in parallel. The easiest way to launch TensorFlow
with limited GPU/memory resources is to use CUDA_VISIBLE_DEVICES=X on the command line,
and wait for this to complete.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import multiprocessing as mp
import glob
import subprocess

from split import chop
import numpy as np

def launch_tensorflow(info):
    cuda_device, folders = info
    for folder in folders:
        command = 'CUDA_VISIBLE_DEVICES={} python collect_states.py --data_folder {}'.format(cuda_device, folder)
        print command
        subprocess.call(command, shell=True)
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--folder',
    help='Folder with image folders',
    default='/mnt/data/images/')
parser.add_argument(
    '--gpus',
    help='How many GPUs to use',
    default=4,
    type=int)
args = parser.parse_args()

pool = mp.Pool(processes=args.gpus)
folders = glob.glob('{}/*'.format(args.folder))
n = int(np.ceil(1.0*len(folders)/args.gpus))

pool.map(launch_tensorflow, enumerate(chop(n, folders)))
