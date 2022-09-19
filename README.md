# jwst_tools

Helper functions for analyzing JWST data

## Authors
- [Jonathan Aguilar](https://www.github.com/aggle)

## Installation

Install jwst_tools with pip

```bash
  cd /path/to/jwst_tools
  pip install .
```

## Requirements
* numpy
* pandas
* pathlib
* matplotlib
* jwst (jwst-pipeline.readthedocs.io)


## JWST tools
* `read.py` - tools for organizing files downloaded from MAST. Useful functions:
  * `read.organize_mast_files(files)` - returns a pandas dataframe witha  row for each file and a column for each observation/visit identifier
  * `read.organize_files_by_header(files, ext)` - returns a pandas dataframe with a row for each file and a column for every fits keyword
* `dq_utils.py` - tools for examining the DQ header
* `plot_utils.py` - tools for plotting files and apertures. kinda weird and scattershot
* `utils.py` - miscellaneous, some has been folded into other modules

Author: Jonathan Aguilar
