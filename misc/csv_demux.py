'''
Reads two csv files, these are:

1) containing sequential data. The file must have the following form:

timestamp, source, event

Please include a header for manual validation, but take note - the column names will not be used, instead 
it is assumed that the data is in this format.

2) The other file will have the events of interest on each line, like this:

event0
event1
event2
... 
eventN

The outputs are:

- a folder with all the sources, each source in one file
- a folder with all the events, where each file lists the sources that have this interesting event
- a file with a set of all the events

Author: Axel.Tidemann@telenor.com
'''

from __future__ import print_function
import argparse
import multiprocessing as mp
import os
import shutil
from functools import partial

import pandas as pd

from utils import chunks, safe_filename

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--data',
    help='The data csv file.')
parser.add_argument(
    '--eoi',
    help='The events of interest csv file.')
parser.add_argument(
    '--source_dir',
    help='''Destination of source files. If not specified, "[data]_sources" created where --data csv is.
Note: this folder will be deleted when the script is run!''',
    default=False)
parser.add_argument(
    '--eoi_dir',
    help='''Destination of events of interest files. If not specified, "[data]_eoi" created where --data csv is.
Note: this folder will be deleted when the script is run!''',
    default=False)
parser.add_argument(
    '--events_filename',
    help='''Destination of the set of events file. If not specified, "[data]_events.csv" created where --data csv is.
Note: this file will be deleted when the script is run!''',
    default=False)
parser.add_argument(
    '--max',
    help='Maximum number of files per processing unit, split evenly across CPUs (i.e. there might be less).',
    type=int,
    default=1000)
parser.add_argument(
    '--chunksize',
    help='Chunksize for the csv file iterator.',
    type=int,
    default=50000)

args = parser.parse_args()

args.source_dir = args.source_dir or '{}_sources'.format(args.data)
args.eoi_dir = args.eoi_dir or '{}_eoi'.format(args.data)

shutil.rmtree(args.source_dir, ignore_errors=True)
shutil.rmtree(args.eoi_dir, ignore_errors=True)
os.makedirs(args.source_dir)
os.makedirs(args.eoi_dir)

def sort_eoi(eoi, eoi_dir, files):
    for csv in files:
        data = pd.read_csv(csv,
                           names=['timestamp', 'event'],
                           dtype={'event': str},
                           parse_dates=[0],
                           index_col=0)
        data.sort_index(inplace=True)
        data.to_csv(csv, mode='w')

        local_events = data.event.unique()
        for event in eoi.event:
            if event in local_events:
                with open('{}/{}'.format(eoi_dir, safe_filename(event)), 'a+') as _file:
                    print(os.path.basename(os.path.normpath(csv)), file=_file)
        
def split_sources(source_dir, chunk):
    for source in filter(pd.notnull, chunk.source.unique()):
        data = chunk[ chunk.source == source ]
        data.to_csv('{}/{}'.format(source_dir, safe_filename(source)), # We can get sources that are invalid filenames
                    mode='a', # Should manage concurrent writes on proper filesystems
                    header=False, # So we can append various times without messing up the file
                    index=False, # This because the query appends a new index
                    columns=['timestamp', 'event'])

csv = pd.read_csv(args.data,
                  header=0,
                  names=['timestamp', 'source', 'event'],
                  dtype={'source': str, 'event': str},
                  parse_dates=[0],
                  chunksize=args.chunksize)

par_split = partial(split_sources, args.source_dir)

pool = mp.Pool()
pool.map(par_split, csv)

source_files = [ os.path.join(args.source_dir, f) for f in os.listdir(args.source_dir) ]
n = min(args.max, len(source_files)/mp.cpu_count()) or 1
eoi = pd.read_csv(args.eoi,
                  header=None,
                  names=['event'],
                  dtype=str)

par_proc = partial(sort_eoi, eoi, args.eoi_dir)
data = chunks(source_files, n)
pool.map(par_proc, data)
