"""
Tools for loading and manipulating the filter profiles
"""

from pathlib import Path
from astropy.io.votable import parse as voparse

table_path = Path(__file__).parent / "filter_profiles" 


def list_available_filters():
    """
    List the available filters

    Parameters
    ----------
    none

    Output
    ------
    prints a list of available filter names

    """
    filter_files = table_path.glob("*.xml")
    filters = sorted([f.stem.split('.')[-1] for f in filter_files])
    print("Available filters:")
    for f in filters:
        print(f"\t{f}")
    return


def load_profile(filter_name):
    """
    Get a filter profile by entering the filter name, e.g. "F1000W" or "F2300C"

    Parameters
    ----------
    filter_name : str
      the name of the filter

    Output
    ------
    profile : astropy.table.Table
      Table of the filte profile with two columns: angstrom, and photon transmission efficiency

    """
    filter_file = table_path / ("JWST.MIRI." + filter_name + ".xml")
    try:
        filter_file.exists()
    except AssertionError:
        print(f"Filter {filter_name} not found. Available filters are:")
        list_available_filters()
        return
    table = voparse(filter_file).get_first_table().to_table()
    return table

