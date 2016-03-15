'''
MSC and GGSN data are merged into one canonical HDF5 file with
IMSI and site_ID. Empty records are thrown away. This replaces older
globul*merge*.py files.

IMSI is taken from  outgoing/incoming typecodes, since you could not
rely on the assumption that they would always be exclusive. site_ID
is converted from cell_ID. Some SACs can not be converted to cell_IDs
either, these are discarded.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from functools import partial

import pandas as pd
import ipdb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--ggsn',
    help='Path to the GGSN HDF5 file')
parser.add_argument(
    '--msc',
    help='Path to the MSC HDF5 file')
parser.add_argument(
    '--meta',
    help='Path to the meta HDF5 file')
parser.add_argument(
    '--output',
    help='Path to the resulting HDF5 file',
    default='file.h5')
parser.add_argument(
    '--interval_length',
    help='Time interval to chunk over files',
    default='hour')
args = parser.parse_args()

def from_to(start_date, end_date):
    return 'index >= "{}" & index < "{}"'.format(start_date, end_date)

def cell_to_site(mapping, cell_ID):
    # Standard cell_IDs from CDR data
    if len(cell_ID) == 5: 
        return int(cell_ID)/10
    try:
        # 4 numbers need SAC->cell_ID conversion based on Huawei dump
        return int(mapping[int(cell_ID)])/10 
    except:
        print 'Could not convert cell_ID {}. -1 returned, handle appropriately.'.format(cell_ID)
        return -1

# These are the type codes that belong to callingSubscriberIMSI (A) and calledSubscriberIMSI (B),
# taken from file MSC CDRs Description.xlsx.
A = ['01', '04']
B = ['04', '07']
    
with pd.get_store(args.ggsn) as ggsn_store, \
     pd.get_store(args.msc) as msc_store, \
     pd.get_store(args.meta) as meta_store, \
     pd.HDFStore(args.output, 'w', complevel=9, complib='blosc') as merged_store:

    ggsn_nrows = ggsn_store.get_storer('data').nrows
    msc_nrows = msc_store.get_storer('data').nrows

    # Find first and last dates, we assume these will be in the first/last 10 000 rows of each file.
    first = min( min(ggsn_store.select('data', stop=int(1e4)).index),
                 min(msc_store.select('data', stop=int(1e4)).index) )
    
    last = max( max(ggsn_store.select('data', start=ggsn_nrows - int(1e4)).index),
                 max(msc_store.select('data', start=msc_nrows - int(1e4)).index) )
    print 'The data runs from {} to {}. GGSN: {} rows, ' \
    'MSC: {} rows. We loop on {} intervals.'.format(first, last, ggsn_nrows, msc_nrows, args.interval_length)

    interval = pd.Timedelta('1 {}'.format(args.interval_length))
    
    start_date = first
    end_date = first + interval

    huawei = meta_store['huawei']

    mapping = {}
    for sac, cell_id in zip(huawei.sac, huawei.index):
        mapping[sac] = cell_id

    par_cell_to_site = partial(cell_to_site, mapping)

    missing_IMSI = 0

    while start_date < last:
        print '{} -> {}'.format(start_date, end_date)
        
        ggsn_data = ggsn_store.select('data', where=from_to(start_date, end_date))
        
        msc_data = msc_store.select('data', where=from_to(start_date, end_date))
        msc_data['in_A'] = msc_data.typeCode.apply(lambda x: x in A)
        A_data = pd.DataFrame(msc_data.query('in_A == True'))
        A_data['IMSI'] = A_data.callingSubscriberIMSI

        msc_data['in_B'] = msc_data.typeCode.apply(lambda x: x in B)
        B_data = pd.DataFrame(msc_data.query('in_B == True'))
        B_data['IMSI'] = B_data.calledSubscriberIMSI

        data = ggsn_data.append(A_data).append(B_data).sort()

        assert len(data) == (len(ggsn_data) + len(A_data) + len(B_data)), 'Error in data length.'

        del data['recordType']
        del data['typeCode']
        del data['callingSubscriberIMSI']
        del data['calledSubscriberIMSI']
        del data['in_A']
        del data['in_B']

        data['length'] = data.IMSI.apply(len)
        data = pd.DataFrame(data.query('length > 0'))
        del data['length']

        missing_IMSI += len(ggsn_data) + len(A_data) + len(B_data) - len(data)

        data['site_ID'] = data.cell_ID.apply(par_cell_to_site)
        del data['cell_ID']
        data = pd.DataFrame(data.query('site_ID > -1'))
        
        start_date = end_date
        end_date = start_date + interval

        merged_store.append('data', data, data_columns=True)

    print 'There are {} ({}%) rows with missing IMSI'.format(missing_IMSI, 100.0*missing_IMSI/(ggsn_nrows+msc_nrows))
    print '{} and {} merged and stored in {}'.format(args.ggsn, args.msc, args.output)
