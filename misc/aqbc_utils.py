import numpy as np
import math
from collections import defaultdict
from sklearn.preprocessing import normalize
from bitstring import Bits, BitArray

#########################################################################################################
# Helpers for implementation of aqbc paper
# https://papers.nips.cc/paper/4831-angular-quantization-based-binary-codes-for-fast-similarity-search.pdf

# cache a few square roots for speed
sqrts = {}
for i in range(1,10000):
    sqrts[i] = math.sqrt(i)

def nearest_binary_landmark(R, X):
    y = np.dot(R, X)
    y = np.transpose(y)
    normalize(y, copy=False)

    n = y.shape[0]
    c = y.shape[1]
   
    idx = np.argsort(y, axis=1)
    bs = []
    ms = []

    for i in range(n):
        b = np.zeros(c)
        s = 0
        max_psi = 0
        max_b = b
        m=1
        for k in range(c-1, -1, -1):
            if y[i][idx[i][k]] <= 0:
                break
            b[idx[i][k]] = 1
            s = s + y[i][idx[i][k]]
            psi = s / math.sqrt(c-k)

            if psi > max_psi:
                max_psi = psi
                max_b = np.copy(b)
                m = float(c-k)

        bs.append(max_b)
        ms.append(m)
    bs = np.array(bs)
    return bs, np.mean(ms), y

def binary_landmark_similarity(b_i, b): #computes similarity relative to query vector b_i
    return (b_i & b).count(1) / sqrts[b.count(1)]

def get_similarity_binary_landmarks(b_i, B):
    res = defaultdict(list)
    for b in B:
        cos_sim = binary_landmark_similarity(b_i, b)
        res[cos_sim].append(b)
    return res

def hamming_search(b, bit, radius, current_radius, H):
    if current_radius == radius:
        return
    for i in range(bit+1, b.length):
        b2 = BitArray(b.copy())
        b2.invert(i)
        H[current_radius+1].append(b2.copy())
        hamming_search(b2, i, radius, current_radius+1, H)
    

def generate_binary_codes_indices(B):
    n = B.shape[0]
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
        
# R is the rotation matrix, X contains the bottlenecks (n, 2048)
# returns the hash codes and the associated inverted index
def hash_bottlenecks(R, X):
    B, m, y = nearest_binary_landmark(R, X)
    return generate_binary_codes_indices(B)

