'''
Plot location of the cell towers in Sofia.

Author: Axel.Tidemann@telenor.com

'''

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import ipdb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('meshgrid_file', help='Basemap meshgrid shape file')
parser.add_argument('h5_file', help='HDF5 file')
parser.add_argument('--frame_table',
                    help='Frame table to read from HDF5 file',
                    default='site_info')
parser.add_argument('--output_destination',
                    help='Where to write the image',
                    default='.')

args = parser.parse_args()

plt.figure()

site_info = pd.read_hdf(args.h5_file, args.frame_table)
sofia = site_info.query("Region == 'SOFIA_CITY'")

m = Basemap(projection='ortho',lon_0=min(sofia.Longitude),lon_1=max(sofia.Longitude),
            lat_0=min(sofia.Latitude), lat_1=max(sofia.Latitude), resolution='l')

bulgaria = m.readshapefile(args.meshgrid_file, 'bulgaria', linewidth=1)

xmin, ymin = m(min(sofia.Longitude), min(sofia.Latitude))
xmax, ymax = m(max(sofia.Longitude), max(sofia.Latitude))

ax = plt.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

m.plot(sofia.Longitude.values, sofia.Latitude.values, 'b.', latlon=True)

for lon,lat,site_id in zip(sofia.Longitude.values, sofia.Latitude.values, sofia.index):
    x,y = m(lon, lat)
    plt.text(x,y,site_id, size=1)

plt.tight_layout()
plt.plot()
plt.savefig('{}/sofia.png'.format(args.output_destination), dpi=900)
