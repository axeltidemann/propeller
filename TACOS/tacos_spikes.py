'''
Analyzing priority reports as spike trains. The spike trains are
formatted as in the tacos_hdf5_to_spikes.py file.

python tacos_spikes.py /path/to/spike_trains*txt

Author: Axel.Tidemann@telenor.com
'''

import sys
import ast
import itertools

import numpy as np
import pyspike as spk
import matplotlib.pyplot as plt

def spk_plot(func, spike_trains, title, labels):
    plt.figure()
    result = func(spike_trains)
    plt.imshow(result, interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)

for input_file in sys.argv[1:]:
    with open(input_file) as file:
        header = file.readline().translate(None, '# ')
        edges = ast.literal_eval(header)
        labels = [ line.translate(None, '# \n') for line in itertools.islice(file, 0, None, 2) ]

    spike_trains = spk.load_spike_trains_from_txt(input_file, is_sorted=True, edges=edges)

    # spk_plot(spk.isi_distance_matrix, spike_trains, "ISI-distance", labels)
    # spk_plot(spk.spike_distance_matrix, spike_trains, "SPIKE-distance", labels)
    place = input_file.split('spike_trains_',1)[1].split('.txt',1)[0]
    spk_plot(spk.spike_sync_matrix, spike_trains, '{} SPIKE sync'.format(place), labels)

    plt.savefig('{}.sync.png'.format(input_file))
    plt.clf()
