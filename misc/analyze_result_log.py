# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse

import numpy as np

parser = argparse.ArgumentParser(description='''
Prints out mean and standard deviation of TOTAL TIME in the
result log file provided as input
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'log_file',
    help='Log file from running SLURM job')
args = parser.parse_args()

needle = 'TOTAL TIME:'

with open(args.log_file) as _file:
    times = []
    for line in _file:
        if needle in line:
            times.append(float(line.split(needle)[-1]))

print '{} data points; mean: {} std: {}'.format(len(times), np.mean(times), np.std(times))
