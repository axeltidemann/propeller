# Implementation of aqbc paper
# https://papers.nips.cc/paper/4831-angular-quantization-based-binary-codes-for-fast-similarity-search.pdf

import argparse
import pandas as pd
import numpy as np
import os
#import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from progressbar import ProgressBar
import redis
from collections import defaultdict
from sets import Set
from bitstring import Bits, BitArray
import blosc

from aqbc_utils import nearest_binary_landmark, hamming_search

def optimize(c, X):
    print "Optimizing Q(B,R) for", c, "bits..."
    n = X.shape[0]
    d = X.shape[1]

    X = np.transpose(X)
    R=scipy.linalg.orth(np.random.binomial(1, 0.5, (d, c)))

    for _ in range(int(args.iter)):
        R = np.transpose(R)

        print "iter", _, ") Computing Nearest Binary Landmarks"
        B, m, y = nearest_binary_landmark(R, X)

        B_n = normalize(B, copy=True)
        
        QBR = 0
        for i in range(n):
            QBR = QBR + np.dot(B_n[i], y[i])
        print "Q(B,R)=", QBR/n, "m=", m

        XB = np.dot(X,B_n)
        Uc, s, V = LA.svd(XB, full_matrices=False)
        R = np.dot(Uc, V)
        print "R type", R.dtype, R.shape
    return R, B
    

def index_Q_X_similarity(similars, D):
    res = defaultdict(set)
    for t in zip(similars[0], similars[1]):
        res[t[0]].add(t[1])
    return res
            

def compute_precision_recall(X, queries, similar, codes, inv_codes):
    print "Computing precision/recall/hits statistics..."
    n = X.shape[0]
    precision = defaultdict(list)
    recall = defaultdict(list)
    hits = defaultdict(list)

    pbar = ProgressBar()
    for q in pbar(range(len(queries))):
        relevant = similar[q]
        if len(relevant) == 0:
            continue
        c = inv_codes[queries[q]]

        H = defaultdict(list)
        hamming_search(BitArray(c), -1, 3, 0, H)
        H[0].append(c)
        retrieved = set()
        for radius in H:
            for h in H[radius]:
                c1 = Bits(bin=h.bin) 
                if c1 in codes:
                    retrieved = retrieved.union(set(codes[c1]))

            tp = relevant.intersection(retrieved)
            fp = retrieved.difference(relevant)
            fn = relevant.difference(retrieved)
            p = len(tp)/float(len(tp)+len(fp))
            r = len(tp)/float(len(tp)+len(fn))
            #print "radius", radius, p, r, len(retrieved), "\n"
            precision[radius].append(p)
            recall[radius].append(r)
            hits[radius].append(len(retrieved))

    for i in precision:
        precision[i] = np.mean(precision[i])
    for i in recall:
        recall[i] = np.mean(recall[i])
    for i in hits:
        hits[i] = np.mean(hits[i])


    return precision, recall, hits

def generate_binary_codes_indices(B, n):
    codes = defaultdict(list)
    inv_codes = dict()
    for i in range(n):
	s="0b"
	for b in B[i]:
	    if b:
	        s = s + "1"
	    else:
	        s = s + "0"
	a = Bits(bin=s)
	codes[a].append(i)
	inv_codes[i] = a
    return codes, inv_codes
	
def benchmark(X):
    n = X.shape[0]

    #sample a few points
    print "Sampling", args.nsamples, "points ..."
    queries = sorted(np.random.choice(n, int(args.nsamples)))
    Q = []
    for q in queries: 
        Q.append(X[q])
    Q = np.array(Q)
    
    D = cosine_similarity(Q,X)
    D_0_9 = np.where( D > 0.9)
    similar = index_Q_X_similarity(D_0_9, D)
    
    numbits = int(args.bits)
    
    f = open(args.out,'w', 0)
    
    for b in range(2, numbits, 1):
        print "\n--------------------", b, " BITS -----------------------"
        R, B = optimize(b, X)
        codes, inv_codes = generate_binary_codes_indices(B, n)
        p, r, h = compute_precision_recall(X, queries, similar, codes, inv_codes)
        for i in range(len(p)):
            text = "{} {} {} {} {}\n".format(b,i,p[i],r[i], h[i])
            print text
            f.write(text)

    #plt.plot(range(2,numbits+1), precision, label="precision")
    #plt.plot(range(2,numbits+1), recall, label="recall")
    #plt.xlabel('#bits')
    #plt.ylabel('score')
    #plt.legend(loc='lower right')
    #plt.show()

def run():
    X = []
    cnt = 0
    for category in os.listdir(args.h5):
        cnt = cnt + 1
        print "Loading category", category
        h5_file = args.h5 + "/" + category
        data = pd.read_hdf(h5_file, 'data')
        array = np.vstack(data.state)
        X.append(array)
    
    X = np.vstack(X)

    if args.bench != False:
        benchmark(X)
    else:
        R, B = optimize(int(args.bits), X)
        R_c = blosc.compress(R.tostring(), typesize=8, cname='zlib')
        red.set('hashing:R', R_c)
    
        
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--h5',
    help="the directory containg the h5 files"
    )
parser.add_argument(
    '--bits',
    help="the number of bits to use for hashing",
    default=64
    )
parser.add_argument(
    '--nsamples',
    help="the number of samples to use for assessing precision-recall",
    default=2000
    )
parser.add_argument(
    '--iter',
    help="the number of iterations when optimizing",
    default=5
    )
parser.add_argument(
    '--bench',
    help="benchmark to find the sweet spot (number of bits to use)",
    default=False
    )
parser.add_argument(
    '--redis',
    help="redis server",
    default="localhost"
    )
parser.add_argument(
    '--port',
    help="redis server port",
    default=6379
    )
parser.add_argument(
    '--out',
    help="output file of benchmark",
    default="precision_recall.txt"
    )

args = parser.parse_args()
red = redis.StrictRedis(args.redis, args.port)
print args

run()
