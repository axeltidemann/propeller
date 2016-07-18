# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import multiprocessing as mp
import glob
import subprocess

KILL = 'POISON PILL'
def launch_tensorflow(q, cuda_device, mem_ratio, target, limit):
    while True:
        source = q.get()
        if source == KILL:
            break
        command = 'CUDA_VISIBLE_DEVICES={} python collect_states.py --source {} --target {} --limit {} --mem_ratio {}'.format(cuda_device, source, target, limit, mem_ratio)
        print command
        subprocess.call(command, shell=True)

parser = argparse.ArgumentParser(description='''
Starts the collection of states in parallel. The easiest way to launch TensorFlow
with limited GPU/memory resources is to use CUDA_VISIBLE_DEVICES=X on the command line,
and wait for this to complete. Given that this uses a queue for the workers to read from,
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
    default=2,
    type=int)
parser.add_argument(
    '--loop',
    help='How many times to loop the computation of a folder, typically used for testing performance.',
    default=1,
    type=int)
args = parser.parse_args()

q = mp.Queue()

for gpu in range(args.gpus):
    for _ in range(args.threads):
        mp.Process(target=launch_tensorflow, args=(q, gpu, args.threads, args.target, args.limit)).start()

for folder in glob.glob('{}/*'.format(args.folder)):
    for _ in range(args.loop):
        q.put(folder)

for _ in range(args.gpus*args.threads):
    q.put(KILL)
