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

from utils import chunks, safe_filename

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
    '--max',
    help='Maximum number of files per processing unit, split evenly across CPUs (i.e. there might be less).',
    type=int,
    default=1000)
parser.add_argument(
    '--chunksize',
    help='Chunksize for the csv file iterator.',
    type=int,
    default=50000)
parser.add_argument(
    '--cores',
    help='Number of CPU cores to use.',
    type=int,
    default=mp.cpu_count())
parser.add_argument(
    '--flush_queue',
    help='Whether to successively put items on the queue, instead of all at once. On Mac OS X, this needs to be set.',
    action='store_true',
    default=False)
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
                with open('{}/{}'.format(eoi_dir, safe_filename(event)), 'a+') as _file:
                    print(os.path.basename(os.path.normpath(csv)), file=_file)
                    
    return unique_events


def split_sources(source_dir, q):
    while True:
        chunk = q.get()

        if not isinstance(chunk, pd.DataFrame):
            break

        data = defaultdict(list)

        t0 = time.time()

        for row in chunk.itertuples(index=False):
            timestamp, source, event = row
            data[source].append((timestamp, event))

        print('Demuxing chunk: {} seconds'.format(time.time()-t0))

        t0 = time.time()

        for source in filter(pd.notnull, data.keys()):
            with open('{}/{}'.format(source_dir, safe_filename(source)), 'a+') as _file:
                for timestamp, event in data[source]:
                    print('{},{}'.format(timestamp, event), file=_file)

        print('Writing source csv files to disk: {} seconds.'.format(time.time()-t0))
                    
csv = pd.read_csv(args.data,
                  header=0,
                  names=['timestamp', 'source', 'event'],
                  dtype={'source': str, 'event': str},
                  parse_dates=[0],
                  chunksize=args.chunksize)

t_start = time.time()

for chunk in csv:
    data = defaultdict(list)

    t0 = time.time()

    for row in chunk.itertuples(index=False):
        timestamp, source, event = row
        data[source].append((timestamp, event))

    print('Demuxing chunk: {} seconds'.format(time.time()-t0))

    t0 = time.time()

    sources = filter(pd.notnull, data.keys())
    
    for source in sources:
        with open('{}/{}'.format(args.source_dir, safe_filename(source)), 'a+') as _file:
            for timestamp, event in data[source]:
                print('{},{}'.format(timestamp, event), file=_file)

    print('Writing {} source csv files to disk: {} seconds.'.format(len(sources), time.time()-t0))

# q = mp.Queue()

# t0 = time.time()

# processes = [ mp.Process(target=split_sources, args=(args.source_dir, q)) for _ in range(args.cores) ]

# for p in processes:
#     p.start()

# for chunk in csv:
#     while not args.flush_queue and q.qsize() > 2*args.cores:
#         time.sleep(1)
#     q.put(chunk)

# for _ in range(args.cores):
#     q.put('DIE!')

# for p in processes:
#     p.join()
    
# t1 = time.time()-t0

t_end = time.time()-t_start

source_files = [ os.path.join(args.source_dir, f) for f in os.listdir(args.source_dir) ]

print('{} sources demultiplexed in {} seconds.'.format(len(source_files), t_end))

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
for subset in pool.map(par_proc, data):
    unique_events.update(subset)

print('Sorting and determining events of interest done in {} seconds.'.format(time.time()-t0))
    
with open(args.events_filename, 'w') as _file:
    for event in unique_events:
        print(event, file=_file)
