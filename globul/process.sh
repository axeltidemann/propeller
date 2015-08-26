#!/bin/bash

export GLOBUL=/mnt/data/FromTnBul

rm $GLOBUL/data.h5

#time python globul_metadata_to_pandas.py $GLOBUL/data.h5 $GLOBUL/Description
time python globul_ggsn_to_pandas.py $GLOBUL/data.h5 $GLOBUL/OriginalData/ggsn.csv
time python globul_msc_to_pandas.py $GLOBUL/data.h5 $GLOBUL/OriginalData/msc.csv
