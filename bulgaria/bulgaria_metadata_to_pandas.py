'''
Import metadata stored in EXCEL files to pandas.

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import namedtuple

import pandas as pd

M = namedtuple('Meta', ['site_info', 'msc_description', 'ggsn_acr_dob', 'ggsn_description'])
meta = M('siteInfo.xlsx', 'MSC CDRs Description.xlsx', 'GGSN-ACRs15-DOB.xlsx', 'GGSN xDRs.xlsx')

with pd.HDFStore('data.h5', 'a', complevel=9, complib='blosc') as store:
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
