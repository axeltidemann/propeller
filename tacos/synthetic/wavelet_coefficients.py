'''
Plot wavelet coefficients for different random boolean vectors.

python wavelet_coefficients.py

Author: Axel.Tidemann@telenor.com
'''

import pywt
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

def random():
    fig, (ax_w, ax_f) = plt.subplots(2)

    distributions = [.2, .4, .6, .8, .99]
    colors = ['b', 'r', 'g', 'm', 'c']

    for d,c in zip(distributions, colors):
        x_wavelet, y_wavelet, x_fourier, y_fourier = [],[],[],[]
        for i in range(int(1e3)):
            signal = np.random.rand(1e5) > d
            (cA2, cD2), (cA1, cD1) = pywt.swt(signal, 'haar', level=2)
            x_wavelet.append(np.mean(cA1))
            y_wavelet.append(np.mean(cA2))

            fourier = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(fourier))
            peaks = argrelextrema(np.abs(fourier), np.greater)

            x_fourier.append(freqs[peaks[0][0]])
            y_fourier.append(freqs[peaks[0][1]])
            
        ax_w.scatter(x_wavelet, y_wavelet, marker='o', c=c, label='w: rand(1e5) > {}'.format(d), linewidth=0)
        ax_f.scatter(x_fourier, y_fourier, marker='+', c=c, label='f: rand(1e5) > {}'.format(d))

    ax_w.set_title('Random signals')
    ax_w.set_xlabel('Wavelet transform')
    ax_f.set_xlabel('Fourier transform')
    #plt.legend()

    plt.show()


def periodic(): # Perfect for Fourier
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 #/ 800.0
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = np.fft.fft(y)
    idx = np.argsort(yf)[::-1]
    freqs = np.fft.fftfreq(N, T)
    print freqs[idx[:4]]

    peaks = argrelextrema(np.abs(yf), np.greater)
    print peaks, freqs[peaks]
    
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    
    plt.axvline(x=freqs[peaks[0][0]], color='r')
    plt.axvline(x=freqs[peaks[0][1]], color='r')

    plt.show()

def bursts():
    pass



if __name__ == '__main__':
    #random()
    periodic()
    
