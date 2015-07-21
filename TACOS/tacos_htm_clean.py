'''
Cleaning badly formatted HTML files.

Author: Axel.Tidemann@telenor.com
'''

from __future__ import print_function
import sys

for input_file in sys.argv[1:]:
    garbage_counter = 0
    with open(input_file, 'rb') as dirty:
        with open('{}_cleaned'.format(input_file), 'w') as clean:
            for i, line in enumerate(dirty):
                if '<TABLE>' in line or '1970-01-01' in line:
                    garbage_counter += 1
                else:
                    print(line, file=clean)
    print('{}: removed {} lines, which is {}%'.format(input_file, garbage_counter, 100*float(garbage_counter)/i))

