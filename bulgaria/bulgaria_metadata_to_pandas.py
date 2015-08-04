'''
Import metadata stored in EXCEL files to pandas.

python bulgaria_metadata_to_pandas.py /path/to/data.h5 /path/to/various/files/

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import namedtuple

import pandas as pd

M = namedtuple('Meta', ['site_info', 'msc_description', 'ggsn_acr_dob', 'ggsn_description'])
meta = M('{}/siteInfo.xlsx'.format(sys.argv[2]),
         '{}/MSC CDRs Description.xlsx'.format(sys.argv[2]),
         '{}/GGSN-ACRs15-DOB.xlsx'.format(sys.argv[2]),
         '{}/GGSN xDRs.xlsx'.format(sys.argv[2]))

with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
    data = { 'site_info': pd.read_excel(meta.site_info, sheetname=0, index_col=0),
             'huawei': pd.read_excel(meta.site_info, sheetname=1, index_col=0),
             'msc_description': pd.read_excel(meta.msc_description, sheetname=0, parse_cols=[1,2]),
             'msc_type_code': pd.read_excel(meta.msc_description, sheetname=1, index_col=0),
             'ggsn_UP-RG-SID': pd.read_excel(meta.ggsn_acr_dob, sheetname=0),
             'ggsn_services': pd.read_excel(meta.ggsn_acr_dob, sheetname=1),
             'ggsn_description': pd.read_excel(meta.ggsn_description, sheetname=0, parse_cols=[1,3]) }

    for key in data.keys():
        store.put(key, data[key])

    print '{} stored in HDF5.'.format(meta)
