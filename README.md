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
* read.py - tools for organizing files downloaded from MAST
* dq_utils.py - tools for examining the DQ header
* utils.py - miscellaneous, some has been folded into other modules
* plot_utils.py - tools for plotting files and apertures. kinda weird and scattershot
* boresight_offsets - tools for computing the correct boresight offsets

## Users who aren't Jonathan
Users who aren't me will probably be most interested in `read.py` and `dq_utils.py`.

### `read.py`
`read.py` has some handy functions for organizing JWST pipeline files. In particular, I like the function `read.organize_mast_files(list_of_files)`. This accepts a list of files and parses the filename to create a dataframe that associates the file with its observation number, visit group, activity number, etc (see https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html). You can also pass the optional argument `extra_keys = {"HEADER_KWD": EXT #}` to add fits header keywords to each row of the dataframe. There is also the function `read.organize_file_by_header(list_of_files)`, which will create one row for each file and one column for each keyword found in the headers. This allows you to quickly sort through all of the files by keyword. 

### `dq_tools.py`
The functions in this module are used for parsing and displaying the DQ image. First, make a dictionary of all the DQ flags and their pixels with `dq_tools.separate_dq_flags(img)`. Then, pass that dictionary to `dq_tools.plot_dq_flags(flag_dict)` to get a plot of the locations of the flagged pixels, one plot for each flag.

Author: Jonathan Aguilar
