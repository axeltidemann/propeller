#!/bin/bash

rm data.h5

time python bulgaria_metadata_to_pandas.py
time python bulgaria_ggsn_to_pandas.py ggsn.csv
time python bulgaria_msc_to_pandas.py msc.csv
