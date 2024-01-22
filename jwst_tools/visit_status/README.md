# Check Visit Status

This program takes a list of JWST programs and checks their visit statuses. It uses curl to download program visit information from https://www.stsci.edu/jwst/science-execution/program-information.html.


It reads from a list of programs stored in the file `jwst_programs.txt`. This file has one program ID number per line.

Curl must use the `-o, --output_file "file.xml"` option to access the visit table. 

