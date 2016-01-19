'''
Plot location of the cell towers in Bulgaria.

Author: Axel.Tidemann@telenor.com

'''

import argparse

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pykml.factory import KML_ElementMaker as KML
from lxml import etree

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('h5_file', help='HDF5 file')
parser.add_argument('--frame_table',
                    help='Frame table to read from HDF5 file',
                    default='site_info')
parser.add_argument('--output_file',
                    help='KML filename',
                    default='sofia.kml')
parser.add_argument('--region',
                    help='What region to select',
                    default='SOFIA_CITY')
args = parser.parse_args()

site_info = pd.read_hdf(args.h5_file, args.frame_table)
region = site_info.query("Region == '{}'".format(args.region))

kml = KML.Document()
kml.append(KML.Folder())

for lon,lat,site_id in zip(region.Longitude.values, region.Latitude.values, region.index):
    kml.Folder.append(KML.Placemark(KML.name(site_id), KML.Point(KML.coordinates('{},{}'.format(lon, lat)))))

with open(args.output_file, 'w') as f:
    f.write(etree.tostring(kml))
