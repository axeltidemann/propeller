# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse

import numpy as np

parser = argparse.ArgumentParser(description='''
Finds the highest validation accuracy in the input files.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'log_files',
    nargs='+')
parser.add_argument(
    '--top_n',
    help='Print out top N results.',
    type=int,
    default=5)
args = parser.parse_args()

needle = 'val_acc: '

accuracies = []
filenames = []

for log in args.log_files:
    with open(log) as _file:
        top_acc = 0
        for line in _file:
            if needle in line:
                _, rest = line.split(needle)
                candidate = float(rest.split()[0])
                if candidate > top_acc:
                    top_acc = candidate
            
        accuracies.append(top_acc)
                
mix = zip(accuracies, args.log_files)

mix_sorted = sorted(mix, key=lambda x: x[0], reverse=True)

for i in range(min(args.top_n, len(mix_sorted))):
    print '{} {}'.format(mix_sorted[i][0], mix_sorted[i][1])
    
