'''
Validates the categories, typically found in a folder named "test". Calculates the average.
The correct answer depends on the enumerated ID of the folder names, so these must be the same 
as for the training. This should already be taken care of by the create_test_train_set.py program.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import multiprocessing as mp
import glob
import subprocess

from split import chop
import numpy as np

def launch_tensorflow(info):
    cuda_device, specs = info
    for answer, folder, classifier in specs:
        command = 'CUDA_VISIBLE_DEVICES={} python validate.py --data_folder {} --classifier {} --answer {}'.format(cuda_device, folder, classifier, answer)
        print command
        subprocess.call(command, shell=True)
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--folder',
    help='Folder with image folders for testing',
    default='/mnt/data/images/')
parser.add_argument(
    '--gpus',
    help='How many GPUs to use',
    default=4,
    type=int)
parser.add_argument(
    '--classifier',
    help='Path to classifier',
    default='classifier.svm')

args = parser.parse_args()

pool = mp.Pool(processes=args.gpus)
folders = sorted(glob.glob('{}/*'.format(args.folder)))
items_per_gpu = int(np.ceil(1.0*len(folders)/args.gpus))

specs = [ (answer, folder, args.classifier) for answer, folder in enumerate(folders) ]

cuda_device_and_specs = enumerate(chop(items_per_gpu, specs))
pool.map(launch_tensorflow, cuda_device_and_specs)

top1 = np.mean([ np.load('{}/top1.npy'.format(folder)) for folder in folders ])
top5 = np.mean([ np.load('{}/top5.npy'.format(folder)) for folder in folders ])
    
print 'Top 1: {} Top 5: {}'.format(top1, top5)
