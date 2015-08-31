'''
Import GGSN CSV data to pandas. The third input parameter is the output of `wc -l ggsn.csv`, this is
needed to print out the progress. 

python bulgaria_ggsn_to_pandas.py /path/to/data.h5 /path/to/ggsn.csv /path/to/line_count.txt

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy
from numpy import dtype

with open(sys.argv[3], 'r') as line_count_file:
     line = line_count_file.readline()
     num_lines = int(line.split()[0])

input_file = sys.argv[2]
print 'Reading {}, contains {} lines.'.format(input_file, num_lines)
     
with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
     kwargs = {'parse_dates': { 'timestamp': ['recordOpeningDate', 'recordOpeningTime'], 
                                'report_date': ['dateOfReport', 'timeOfReport'],
                                'change_date': ['changeDate', 'changeTime'],
                                'load_date': ['loadDate'] },
               'date_parser': lambda x: pd.to_datetime(x, coerce=True),
               'index_col': 'timestamp',
               'chunksize': 50000,
               'dtype': {'IMSI': dtype('float64'),
                          'MSISDN': dtype('float64'),
                          'Morphology': dtype('O'),
                          'Place_1': dtype('O'),
                          'Place_2': dtype('O'),
                          'Type_1': dtype('O'),
                          'Type_2': dtype('O'),
                          'aPNSelectionMode': dtype('float64'),
                          'anonymousAccessId': dtype('float64'),
                          'apnNI': dtype('O'),
                          'apnOI': dtype('float64'),
                          'callDuration': dtype('float64'),
                          'causeForRecordClosing': dtype('float64'),
                          'cell_ID': dtype('float64'),
                          'changeCondition': dtype('float64'),
                          'changeOffset': dtype('float64'),
                          'chargingCharacteristics': dtype('float64'),
                          'chargingID': dtype('float64'),
                          'diagnostics': dtype('float64'),
                          'downlink': dtype('float64'),
                          'dynamicAddrFlag': dtype('bool'),
                          'fileName': dtype('O'),
                          'ggsnAddr': dtype('O'),
                          'localSequenceNumber': dtype('float64'),
                          'locationAreaCode': dtype('float64'),
                          'mSNetworkCapability': dtype('float64'),
                          'networkInitiation': dtype('float64'),
                          'nodeId': dtype('O'),
                          'pdpAddr': dtype('O'),
                          'pdpType': dtype('float64'),
                          'qoSNegotiatedDelay': dtype('float64'),
                          'qoSNegotiatedMean': dtype('float64'),
                          'qoSNegotiatedPeak': dtype('float64'),
                          'qoSNegotiatedPrecedence': dtype('float64'),
                          'qoSNegotiatedReliability': dtype('float64'),
                          'qoSRequestedDelay': dtype('float64'),
                          'qoSRequestedMean': dtype('float64'),
                          'qoSRequestedPeak': dtype('float64'),
                          'qoSRequestedPrecedence': dtype('float64'),
                          'qoSRequestedReliability': dtype('float64'),
                          'rATType': dtype('float64'),
                          'rNCUnsentDownlinkVolume': dtype('float64'),
                          'ratingGroup': dtype('float64'),
                          'recordExtension': dtype('float64'),
                          'recordOpeningOffset': dtype('float64'),
                          'recordSequenceNumber': dtype('float64'),
                          'recordType': dtype('int64'),
                          'remotePDPAddr': dtype('float64'),
                          'routingArea': dtype('float64'),
                          'servedIMEI': dtype('float64'),
                          'serviceConditionChange': dtype('float64'),
                          'sgsnAddr': dtype('O'),
                          'sgsnChange': dtype('float64'),
                          'systemType': dtype('float64'),
                          'uplink': dtype('float64')}}

     # The dtype dict was found by examining a sample file (about 3GB), and calling utils.determine_dtypes(input_file, **kwargs)
     # where **kwargs is the same as above. Three dtypes were removed: load_date, report_date and change_date. Not sure if that
     # is necessary, though.

     csv = pd.read_csv(input_file, **kwargs)

     dropped = []
     for chunk in csv:
          dropped.append(numpy.mean(pd.isnull(chunk.index)))
          chunk.drop(chunk.index[pd.isnull(chunk.index)], inplace=True) # NaT as index
          chunk.dynamicAddrFlag = chunk.dynamicAddrFlag.astype(pd.np.bool) # Somehow escapes the dtype settings
          store.append('ggsn', chunk, data_columns=True)
          if len(dropped) % 10 == 0:
               print '{0:.2f}%'.format(100.*len(dropped)/num_lines)

     print '{} stored in HDF5. {}% was dropped since NaT was used as an index.'.format(input_file, 100*numpy.mean(dropped))
