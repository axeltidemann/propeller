# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
from collections import defaultdict
import json

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.datastructures import ImmutableTypeConversionDict

parser = argparse.ArgumentParser(description='''Finds 
clusters of pictures based on the cosine similarity between
them.''')

parser.add_argument(
    'h5',
    help='Path to HDF5 file with states')
parser.add_argument(
    '--table',
    help='HDF5',
    default='data')
parser.add_argument(
    '--cutoff',
    help='At which level of cosine similarity similar images should be cut off',
    default=.9,
    type=float)
parser.add_argument(
    '--min_cluster_size',
    help='The minimum size of clusters',
    default=1000,
    type=int)
parser.add_argument(
    '--filename',
    help='Filename for the cluster JSON file.',
    default='clusters.json')

args = parser.parse_args()

def depth_first(graph, start, path=[]):
    path = path + [start]
    for node in graph[start]:
        if not node in path:
            path = depth_first(graph, node, path)
    return path

data = pd.read_hdf(args.h5, args.table)

X = np.vstack(data.state)

similarity = cosine_similarity(X)
graph = defaultdict(list)

for i, row in enumerate(similarity):
    for j, candidate in enumerate(row):
        if j < i and candidate > args.cutoff:
            graph[data.index[i]].append(data.index[j])

# For some weird reason, the depth_first function modifies the graph dictionary.
# If you make it immutable, an error will be thrown. However, what happens is the addition of
# keys with [] as values, so upon removal of these, the dicts are the same (this has been verified
# by copying the dict and comparing it to the dict stripped from these key, value pairs).
# graph = ImmutableTypeConversionDict(graph)

clusters = {}
visited = []

for node in graph.keys():
    if node not in visited:
        cluster = depth_first(graph, node)
        if len(cluster) > args.min_cluster_size:
            clusters[node] = cluster
            visited.extend(clusters[node])

print 'Cluster sizes: ', [ len(clusters[node]) for node in clusters ]
print '{} clusters of size > {}, cosine similarity cutoff at {}.'.format(len(clusters), args.min_cluster_size, args.cutoff)

with open(args.filename, 'w') as _file:
    json.dump(clusters, _file)

print 'Clusters saved to {}'.format(args.filename)
