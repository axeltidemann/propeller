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
import time
from collections import defaultdict

import pandas as pd

from utils import chunks, safe_filename, file_len, pretty_float

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--data',
    help='The data csv file.')
parser.add_argument(
    '--eoi',
    help='The events of interest csv file. If omitted, just csv demultiplexing is performed.',
    default=False)
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
    '--chunksize',
    help='Chunksize for the csv file iterator.',
    type=int,
    default=50000)
parser.add_argument(
    '--max',
    help='Maximum number of files per processing unit, split evenly across CPUs (i.e. there might be less).',
    type=int,
    default=1000)
parser.add_argument(
    '--cores',
    help='Number of CPU cores to use.',
    type=int,
    default=mp.cpu_count())

args = parser.parse_args()

args.source_dir = args.source_dir or '{}_sources'.format(args.data)
args.eoi_dir = args.eoi_dir or '{}_eoi'.format(args.data)
args.events_filename = args.events_filename or '{}_events'.format(args.data)

shutil.rmtree(args.source_dir, ignore_errors=True)
shutil.rmtree(args.eoi_dir, ignore_errors=True)

os.makedirs(args.source_dir)
os.makedirs(args.eoi_dir)

def sort_eoi(eoi, eoi_dir, files):
    unique_events = set()
    events_to_sources = defaultdict(list)
    
    for csv in files:
        data = pd.read_csv(csv,
                           names=['timestamp', 'event'],
                           dtype={'event': str},
                           parse_dates=[0],
                           index_col=0)
        data.sort_index(inplace=True)
        data.to_csv(csv, mode='w')

        local_events = data.event.unique()
        unique_events.update(local_events)
        
        for event in eoi.event:
            if event in local_events:
                events_to_sources[safe_filename(event)].append(os.path.basename(os.path.normpath(csv)))
                # with open('{}/{}'.format(eoi_dir, safe_filename(event)), 'a+') as _file:
                #     print(os.path.basename(os.path.normpath(csv)), file=_file)
                    
    return unique_events, events_to_sources
                    
csv = pd.read_csv(args.data,
                  header=0,
                  names=['timestamp', 'source', 'event'],
                  dtype={'source': str, 'event': str},
                  parse_dates=[0],
                  chunksize=args.chunksize)

length = file_len(args.data)
print('There are {} lines of data in {}.'.format(length, args.data))

t_start = time.time()

for i, chunk in enumerate(csv):
    data = defaultdict(list)

    t0 = time.time()

    for row in chunk.itertuples(index=False):
        timestamp, source, event = row
        if pd.notnull(source):
            data[source].append((timestamp, event))

    print('Demuxing chunk: {} seconds.'.format(pretty_float(time.time()-t0)), end=' ')

    t0 = time.time()

    for source in data.keys():
        with open('{}/{}'.format(args.source_dir, safe_filename(source)), 'a+') as _file:
            for timestamp, event in data[source]:
                print('{},{}'.format(timestamp, event), file=_file)

    print('Writing {} source csv files to disk: {} seconds.'.format(len(data.keys()), pretty_float(time.time()-t0)), end=' ')

    print('{}% done.'.format(100*(i+1)*args.chunksize/length))
    
t_end = time.time()-t_start

source_files = [ os.path.join(args.source_dir, f) for f in os.listdir(args.source_dir) ]

print('{} sources demultiplexed in {} seconds.'.format(len(source_files), pretty_float(t_end)))

n = min(args.max, len(source_files)/args.cores) or 1
eoi = pd.read_csv(args.eoi,
                  header=None,
                  names=['event'],
                  dtype=str) if args.eoi else pd.DataFrame(data=[], columns=['event'])

par_proc = partial(sort_eoi, eoi, args.eoi_dir)
data = chunks(source_files, n)

unique_events = set()

t0 = time.time()

pool = mp.Pool(processes=args.cores)
for subset_unique_events, subset_events_to_sources in pool.map(par_proc, data):
    unique_events.update(subset_unique_events)
    for event in subset_events_to_sources.keys():
        with open('{}/{}'.format(eoi_dir, event), 'a+') as _file:
            print(os.path.basename(os.path.normpath(csv)), file=_file)


# for csv in source_files:
#     data = pd.read_csv(csv,
#                        names=['timestamp', 'event'],
#                        dtype={'event': str},
#                        parse_dates=[0],
#                        index_col=0)
#     data.sort_index(inplace=True)
#     data.to_csv(csv, mode='w')

#     local_events = data.event.unique()
#     unique_events.update(local_events)

#     for event in eoi.event:
#         if event in local_events:
#             with open('{}/{}'.format(args.eoi_dir, safe_filename(event)), 'a+') as _file:
#                 print(os.path.basename(os.path.normpath(csv)), file=_file)

print('Sorting and determining events of interest done in {} seconds.'.format(pretty_float(time.time()-t0)))
    
with open(args.events_filename, 'w') as _file:
    for event in unique_events:
        print(event, file=_file)
