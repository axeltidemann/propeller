'''
Plot location of the cell towers as well as some statistics related to the users, namely the
histogram of the duration of the calls as well as histogram of calling/SMSing frequency.

python bulgaria_plot_user.py /path/to/data.h5 /path/to/meshgrid/file /path/to/figures/

Author: Axel.Tidemann@telenor.com

'''

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import ipdb

plt.figure()

m = Basemap(projection='ortho',lon_0=25,lat_0=42.5,resolution='l')

bulgaria = m.readshapefile(sys.argv[2], 'bulgaria', linewidth=1)

xmin, ymin = m(*bulgaria[2][:2])
xmax, ymax = m(*bulgaria[3][:2])
ax = plt.gca()
ax.set_xlim(xmin, xmax+10000)
ax.set_ylim(ymin-10000, ymax)
m.drawlsmask(land_color='w')
m.drawmapboundary(fill_color='w')

site_info = pd.read_hdf(sys.argv[1], 'site_info')
m.plot(site_info.Longitude.values, site_info.Latitude.values, 'b.', latlon=True)
plt.tight_layout()
plt.savefig('{}/bulgaria_cells.png'.format(sys.argv[3]), dpi=300)
plt.clf()

with pd.get_store(sys.argv[1]) as store:
    duration = store.select_column('msc', 'chargeableDuration')
    sns.distplot(duration.dropna(), kde=False)
    plt.xlabel('Duration (seconds)')
    plt.tight_layout()
    plt.savefig('{}/duration_histogram.png'.format(sys.argv[3]), dpi=300)
    
    calling = store.select_column('msc', 'callingNumber')
    calling_counts = pd.value_counts(calling)
    called =  store.select_column('msc', 'calledNumber')
    called_counts = pd.value_counts(calling)

    for index in calling_counts.index:
        calling_counts.ix[index] += called_counts.ix[index]

    print '{}% have used the phone to receive/send calls/SMS less than 20 times in the period.'.\
        format(100.*len(calling_counts[ calling_counts < 20 ])/len(calling_counts))
    sns.distplot(calling_counts, kde=False, color='b')
    plt.tight_layout()
    plt.savefig('{}/usage_times_histogram.png'.format(sys.argv[3]), dpi=300)
    
    #     lon,lat = [],[]
    #     errors = False
    #     for cell_ID in person.cell_ID:
    #         try:
    #             position = site_info.ix[int(str(cell_ID)[:-1])] # Messy way of coding sites
    #         except KeyError:
    #             errors = True
    #         lon.append(position.Longitude)
    #         lat.append(position.Latitude)
    #     if errors:
    #         print 'There were some KeyErrors with phone number {}'.format(ph)
    #     m.plot(lon, lat, '.', latlon=True)
    # plt.tight_layout()
    # plt.savefig('{}/user.png'.format(sys.argv[3]), dpi=300)
