'''
Convert .rpt file to .csv format. Utility already written to import CSV files to pandas.

python tacos_rpt_to_csv.py /path/to/file.rpt

Author: Axel.Tidemann@telenor.com
'''

from __future__ import print_function
import sys
import codecs
import re
import subprocess

import ipdb

def split_and_format(string, indices, remove_datetime_zeros=True):
    data =  [ string[i:j].strip().replace(',', ' ').encode('ascii', 'ignore').decode('ascii')
              for i,j in zip( [0] + indices, indices + [None]) ]

    # The first three dates have 7 (!) trailing digits.
    if remove_datetime_zeros:
        for i in range(3):
            data[i] = data[i][:-4]

    return ','.join(data)

# The last three lines tell how many rows were printed
last = subprocess.check_output(['tail', '-3', sys.argv[1]])
n_lines = int(re.findall(r'\d+', last)[0])
print('File contains {} lines.'.format(n_lines))

with codecs.open(sys.argv[1], 'r', 'utf-8-sig') as input_file:
    header = input_file.readline()
    separator = input_file.readline()
    indices = [ m.start() for m in re.finditer(' ', separator) ]

    with open('{}.csv'.format(sys.argv[1]), 'w') as output_file:
        print(split_and_format(header, indices, False), file=output_file)

        i = 0
        while i < n_lines:
            print(split_and_format(input_file.readline(), indices), file=output_file)
            i += 1
