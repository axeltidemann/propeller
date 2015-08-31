'''
Calculates the mean and std dev of the number of data points for the globul dataset.

python globul_user_stats.py /path/to/ggsn.h5 /path/to/msc.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

ggsn_path = sys.argv[1]
msc_path = sys.argv[2]

