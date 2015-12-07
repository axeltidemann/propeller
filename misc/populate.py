''' 
Populates the redis database in parallel using the knn.py file in data/wow.

Author: Axel.Tidemann@telenor.com
'''


import glob
import multiprocessing as mp
import subprocess

def populate(filename):
    bu = filename.replace('vectors_', '').replace('_wow.txt','')
    subprocess.call(['python', 'knn.py', filename, bu])

pool = mp.Pool()
pool.map(populate, glob.glob('vectors_*txt'))
