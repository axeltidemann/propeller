#!/bin/bash

python tacos_htm_clean.py $DATA/TACOS/tacos*htm
python tacos_htm_to_csv.py $DATA/TACOS/tacos*cleaned
python tacos_csv_to_pandas.py $DATA/TACOS/data.h5 $DATA/TACOS/tacos*csv
python tacos_xlsx_to_pandas.py $DATA/TACOS/*xlsx
