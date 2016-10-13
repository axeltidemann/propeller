import os
import argparse
import json
import glob

parser = argparse.ArgumentParser(description='''Prompts
to input the corresponding class. Used until a better mapping
file is produced.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'source',
    help='HDF5 file(s)',
    nargs='+')
args = parser.parse_args()

mapping = {}

for i, h5 in enumerate(sorted(args.source)):
    print '{}: {}'.format(i, os.path.basename(h5))
    mapping[i] = raw_input('common category: ')

with open('mapping.txt','w') as _file:
    json.dump(mapping, _file)

print mapping
