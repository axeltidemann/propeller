# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
from collections import defaultdict
import json
import time
import random

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

parser = argparse.ArgumentParser(description='''Finds 
clusters of pictures based on the cosine similarity between
them.''',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    default=.8,
    type=float)
parser.add_argument(
    '--min_cluster_size',
    help='The ratio of minimum size of clusters/total size',
    default=.01,
    type=float)
parser.add_argument(
    '--filename',
    help='Filename for the cluster JSON file.',
    default='clusters.json')
args = parser.parse_args()

data = pd.read_hdf(args.h5, args.table)
X = np.vstack(data.state)

similarity = cosine_similarity(X)
similarity = np.tril(similarity, -1)
similarity[ similarity < args.cutoff ] = 0

graph = nx.from_numpy_matrix(similarity)

results = []
while graph.number_of_nodes():
    node = random.choice(graph.nodes())
    edges = list(nx.bfs_tree(graph, node))
    results.append((node, edges))
    graph.remove_nodes_from(edges)

results = [ (node, edges) for node, edges in results if len(edges) > similarity.shape[0]*args.min_cluster_size ]
    
clusters = { data.index[node]: [ data.index[e] for e in edges ] for node, edges in results }

print 'Cluster sizes: ', [ len(clusters[node]) for node in clusters ]
print '{} clusters of size > {}, cosine similarity cutoff at {}.'.format(len(clusters),
                                                                         int(args.min_cluster_size*similarity.shape[0]),
                                                                         args.cutoff)
total_nodes = sum(map(len,clusters.values()))
print 'Total nodes in clusters: {}, {}% of total nodes.'.format(total_nodes, 100.*total_nodes/similarity.shape[0])

with open(args.filename, 'w') as _file:
    json.dump(clusters, _file)

print 'Clusters saved to {}'.format(args.filename)
