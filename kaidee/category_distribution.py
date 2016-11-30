# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='''
    Calculates the distribution of ad categories.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'in_file',
    help='CSV file with counts for each of the categories')
parser.add_argument(
    '--cutoff',
    help='What percentage to to include',
    type=float,
    default=.9)

args = parser.parse_args()

data = pd.read_csv(args.in_file, header=None, names=['category_id', 'value'], index_col=[0])

total = data.sum().value

top_sum = 0
top_dogs = []
for index, value in data.iterrows():
    top_sum += value.value
    if top_sum < total*args.cutoff:
        top_dogs.append(value.name)

print ' '.join(map(str, top_dogs))
