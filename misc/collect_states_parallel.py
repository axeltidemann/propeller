# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import multiprocessing as mp
import glob
import subprocess
import os

from utils import maybe_download_and_extract
from collect_states import save_states

KILL = 'POISON PILL'
def launch_tensorflow(q, cuda_device, mem_ratio, target, limit, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    print 'Using CUDA device {}'.format(cuda_device)
    while True:
        source = q.get()
        if source == KILL:
            break
        print 'Processing folder {}'.format(source)
        save_states(source, target, limit, mem_ratio, model_dir)

parser = argparse.ArgumentParser(description='''
Starts the collection of states in parallel, limits memory consumption and visible
CUDA devices to TensorFlow via environment variables. Given that this uses a queue 
for the workers to read from,
this automatically balances the load between the GPU processes.''', 
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'folder',
    help='Folder with image folders')
parser.add_argument(
    'target',
    help='Where to put the states')
parser.add_argument(
    '--limit',
    help='Maximum amount of images to process',
    type=int,
    default=10000)
parser.add_argument(
    '--gpus',
    help='How many GPUs to use',
    default=4,
    type=int)
parser.add_argument(
    '--threads',
    help='How many threads to use pr GPU',
    default=3,
    type=int)
parser.add_argument(
    '--model_dir',
    help='Path to Inception files', 
    default='/tmp/imagenet')
parser.add_argument(
    '--loop',
    help='How many times to loop the computation of a folder, typically used for testing performance.',
    default=1,
    type=int)
args = parser.parse_args()

maybe_download_and_extract(args.model_dir)

q = mp.Queue()

for gpu in range(args.gpus):
    for _ in range(args.threads):
        mp.Process(target=launch_tensorflow, args=(q, gpu, .9/args.threads, args.target, args.limit, args.model_dir)).start()

for folder in glob.glob('{}/*'.format(args.folder)):
    for _ in range(args.loop):
        q.put(folder)

for _ in range(args.gpus*args.threads):
    q.put(KILL)
