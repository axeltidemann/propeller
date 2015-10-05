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
    x_wavelet, y_wavelet, z_wavelet, x_fourier, y_fourier, z_fourier = [[] for _ in range(6)]
    for i in range(int(1e3)):
        signal = np.random.rand(1e5) > d
        (cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(signal, 'haar', level=3)
        x_wavelet.append(np.mean(cA1))
        y_wavelet.append(np.mean(cA2))
        z_wavelet.append(np.mean(cA3))

        fourier = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(fourier))
        idx = np.argsort(np.abs(fourier))[::-1]
        freq = freqs[idx[:3]] # Three most powerful frequencies
        x_fourier.append(freq[0])
        y_fourier.append(freq[1])
        z_fourier.append(freq[2])
    plt.scatter(x_wavelet, y_wavelet, z_wavelet, c=c, depthshade=False, label='wavelet: rand(1e5) > {}'.format(d), linewidth=0, alpha=1)
    plt.scatter(x_fourier, y_fourier, z_fourier, c=c, depthshade=False, label='fourier: rand(1e5) > {}'.format(d), marker='+', linewidth=0, alpha=1)

ax.set_ylabel('mean(cA1)')
ax.set_xlabel('mean(cA2)')
ax.set_zlabel('mean(cA3)')

plt.legend(loc=2)
plt.show()
        

