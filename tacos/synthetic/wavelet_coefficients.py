'''
Plot wavelet coefficients for different random boolean vectors.

python wavelet_coefficients.py

Author: Axel.Tidemann@telenor.com
'''

import pywt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

distributions = [.2, .4, .6, .8, .99]
colors = ['b', 'r', 'g', 'm', 'c']

for d,c in zip(distributions, colors):
    x, y, z = [], [], []
    for i in range(int(1e3)):
        signal = np.random.rand(1e5) > d
        (cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(signal, 'haar', level=3)
        z.append(np.mean(cA3))
        x.append(np.mean(cA2))
        y.append(np.mean(cA1))
    plt.scatter(x, y, z, c=c, depthshade=False, label='rand(1e5) > {}'.format(d), linewidth=0, alpha=1)

ax.set_ylabel('mean(cA1)')
ax.set_xlabel('mean(cA2)')
ax.set_zlabel('mean(cA3)')

plt.legend(loc=2)
plt.show()
        

