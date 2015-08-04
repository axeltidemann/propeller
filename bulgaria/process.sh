#!/bin/bash

export BULGARIA=/Users/tidemann/Documents/data/bulgaria

rm $BULGARIA/data.h5

time python bulgaria_metadata_to_pandas.py $BULGARIA/data.h5 $BULGARIA
time python bulgaria_ggsn_to_pandas.py $BULGARIA/data.h5 $BULGARIA/ggsn.csv
time python bulgaria_msc_to_pandas.py $BULGARIA/data.h5 $BULGARIA/msc.csv
