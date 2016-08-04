# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import multiprocessing as mp
import glob
import subprocess
import os

from transfer_learning import learn

KILL = 'POISON PILL'

def launch_tensorflow(q, cuda_device, states, mem_ratio):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    print 'Using CUDA device {}'.format(cuda_device)
    while True:
        expert = q.get()
        if expert == KILL:
            break
        print 'Training {} expert'.format(expert)
        learn(states, expert=expert, epochs=20, mem_ratio=mem_ratio, hidden_size=1024, learning_rate=.0001)

parser = argparse.ArgumentParser(description='''
Trains a collection of experts.''', 
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'states',
    help='A folder with Inception states to train invidiual experts')
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
args = parser.parse_args()

q = mp.Queue()

for gpu in range(args.gpus):
    for _ in range(args.threads):
        mp.Process(target=launch_tensorflow, args=(q, gpu, args.states, .9/args.threads)).start()

for h5 in glob.glob('{}/*.h5'.format(args.states)):
    q.put(os.path.basename(h5))

for _ in range(args.gpus*args.threads):
    q.put(KILL)
