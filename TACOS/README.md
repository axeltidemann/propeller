The data from TACOS is quite messy. The HTML export is broken, since there in many instances is a <TABLE> element within the summary field. This causes all kinds of havoc for HTML parsers. Furthermore, the date 1970-01-01 appear in many fields. These data points are also
thrown away, although it could be argued that they might indicate that something is wrong on a higher level.

The data is stored in HDF5 format, to make it accessible without any form of RAM limitations. To transfer the raw data from the HTML files, a three-stage process needs to be done:

```
python tacos_htm_clean.py tacos*htm
python tacos_htm_to_csv.py tacos*cleaned
python tacos_csv_to_pandas.py tacos*csv
```

For convenience, this is also written in the process.sh file.