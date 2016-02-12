'''
Uses the saved arrays to train an SVM that will be the new classifier
for the Inception model.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import glob
from collections import namedtuple
import cPickle as pickle

from sklearn import svm
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--folder',
    help='Folder with image folders',
    default='/mnt/data/images/')
args = parser.parse_args()

folders = sorted(glob.glob('{}/*'.format(args.folder)))

Data = namedtuple('Data', 'x y')

states = {}
for i, folder in enumerate(folders):
    category = folder.split('/')[-1]
    x = np.load('{}/states.npy'.format(folder))
    y = np.ones((x.shape[0],1))*i
    states[category] = Data(x,y)
    print '{}: {}'.format(i, category)

X = np.vstack([ states[key].x for key in states.keys() ])
Y = np.vstack([ states[key].y for key in states.keys() ])
Y = np.ravel(Y)

classifier = svm.LinearSVC()
classifier.fit(X,Y)

with open('classifier.svm', 'w') as f:
    f.write(pickle.dumps(classifier))
