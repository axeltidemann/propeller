# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Starts the collection of states in parallel. The easiest way to launch TensorFlow
with limited GPU/memory resources is to use CUDA_VISIBLE_DEVICES=X on the command line,
and wait for this to complete. Given that this uses a queue for the workers to read from,
this automatically balances the load between the GPU processes.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import multiprocessing as mp
import glob
import subprocess

KILL = 'POISON PILL'

def launch_tensorflow(q, cuda_device):
    while True:
        folder = q.get()
        if folder == KILL:
            break
        command = 'CUDA_VISIBLE_DEVICES={} python collect_states.py --data_folder {}'.format(cuda_device, folder)
        print command
        subprocess.call(command, shell=True)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--folder',
    help='Folder with image folders')
parser.add_argument(
    '--gpus',
    help='How many GPUs to use',
    default=4,
    type=int)
args = parser.parse_args()

q = mp.Queue()

for i in range(args.gpus):
    mp.Process(target=launch_tensorflow, args=(q, i,)).start()

for folder in glob.glob('{}/*'.format(args.folder)):
    q.put(folder)

for _ in range(args.gpus):
    q.put(KILL)
