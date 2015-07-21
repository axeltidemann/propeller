#!/bin/bash

rm data.h5

python tacos_htm_clean.py tacos*htm
python tacos_htm_to_csv.py tacos*cleaned
python tacos_csv_to_pandas.py tacos*csv
python tacos_xlsx_to_pandas.py *xlsx