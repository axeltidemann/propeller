'''
Splits data files into train and test sets.

Author: Axel.Tidemann@telenor.com
'''

import shutil
import glob
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--folder',
    help='Folder with image folders',
    default='/mnt/data/images/')
parser.add_argument(
    '--ratio',
    help='Train/test split ratio',
    default=0.75,
    type=float)
parser.add_argument(
    '--target',
    help='Where to put the train/test folders',
    default='/mnt/data/')

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

args = parser.parse_args()

folders = sorted(glob.glob('{}/*'.format(args.folder)))

train_folder = '{}/train'.format(args.target)
test_folder = '{}/test'.format(args.target)
create_folder(train_folder)
create_folder(test_folder)

for folder in folders:
    train_category_folder = '{}/{}'.format(train_folder, folder.split('/')[-1])
    create_folder(train_category_folder)

    test_category_folder = '{}/{}'.format(test_folder, folder.split('/')[-1])
    create_folder(test_category_folder)
    
    files = glob.glob('{}/*.jpg'.format(folder))

    for train_file in files[:int(len(files)*args.ratio)]:
        shutil.copy(train_file, train_category_folder)

    for test_file in files[int(len(files)*args.ratio):]:
        shutil.copy(test_file, test_category_folder)
