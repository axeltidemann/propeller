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

from utils import flatten

def find(X, lower_bound, upper_bound, min_cluster_size):

    similarity = cosine_similarity(X)
    similarity = np.tril(similarity, -1)

    similarity[ similarity < lower_bound ] = 0
    similarity[ similarity > upper_bound ] = 0

    graph = nx.from_numpy_matrix(similarity)

    results = []
    while graph.number_of_nodes():
        node = random.choice(graph.nodes())
        edges = list(nx.bfs_tree(graph, node))
        results.append((node, edges))
        graph.remove_nodes_from(edges)

    included = [ (node, edges) for node, edges in results if len(edges) > similarity.shape[0]*min_cluster_size ]

    clusters = { node: edges for node, edges in included }

    clusters['rejected'] = list(set(range(X.shape[0])).difference(flatten(included)))
    
    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Finds 
    clusters of pictures based on the cosine similarity between
    them.''',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'h5',
        help='HDF5 file(s) with states',
        nargs='+')
    parser.add_argument(
        '--lower_bound',
        help='Lower level of cosine similarity',
        default=.8,
        type=float)
    parser.add_argument(
        '--upper_bound',
        help='Upper level of cosine similarity',
        default=.95,
        type=float)
    parser.add_argument(
        '--min_cluster_size',
        help='The ratio of minimum size of clusters/total size',
        default=.01,
        type=float)
    parser.add_argument(
        '--filename',
        default=False,
        help='Filename for the cluster JSON file. If unspecified, will be h5 filename + .json')
    parser.add_argument(
        '--include_rejected',
        action='store_true',
        help='Whether to include the rejected images')
    args = parser.parse_args()

    for h5 in args.h5:

        X = []
        index = []
        with pd.HDFStore(h5) as store:
            for key in store.keys():
                data = store[key]
                if len(X):
                    X = np.vstack([X, np.vstack(data.state)])
                    index.extend(data.index)
                else:
                    X = np.vstack(data.state)
                    index = list(data.index)

        print 'Done gathering data, X.shape = {}'.format(X.shape)

        clusters = find(X, args.lower_bound, args.upper_bound, args.min_cluster_size)
        keys = clusters.keys()
        keys.remove('rejected')

        print 'Cluster sizes: ', [ len(clusters[node]) for node in keys ]
        print '{} clusters of size > {}, cosine similarity in range ({},{}).'.format(len(keys),
                                                                                     int(args.min_cluster_size*X.shape[0]),
                                                                                     args.lower_bound, args.upper_bound)
        total_nodes = sum([ len(clusters[node]) for node in keys ])
        print 'Total nodes in clusters: {}, {}% of total nodes.'.format(total_nodes, 100.*total_nodes/X.shape[0])

        clusters_with_filenames = { index[node]: [ index[edge] for edge in clusters[node] ] for node in keys }

        if args.include_rejected:
            clusters_with_filenames['rejected'] = [ index[edge] for edge in clusters['rejected'] ]

        args.filename = args.filename if args.filename else '{}.json'.format(h5)
            
        with open(args.filename, 'w') as _file:
            json.dump(clusters_with_filenames, _file)

        print 'Clusters saved to {}'.format(args.filename)

        args.filename = False
