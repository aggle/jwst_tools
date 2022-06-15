"""
Utilities for reading, writing, and parsing files and lists of files
"""

from io import StringIO
from pathlib import Path
import pandas as pd
import re

from astropy.io import fits

def parse_observation(obs_chunk):
    """
    Parse the information for a single visit into a dataframe

    Parameters
    ----------
    obs_chunk: a string containing the visit information

    Output
    ------
    dataframe and visit name

    """
    _, obs_name, visit_id, columns, *data, = obs_chunk.split("\n")
    obs_name = obs_name[2:]
    # the pull out the observation number and the mosaic number, which we toss
    visit_id, mosaic_id = visit_id.split(' ')[-1].split(':')
    columns = columns.split() + ['base']
    df = pd.read_csv(StringIO("\n".join(data)),
                     sep="\s+",
                     names=columns,
                     index_col=False)
    df['ObsName'] = obs_name
    return (int(visit_id), int(mosaic_id), df)


def parse_apt_pointing_file(path):
    """
    Take a pointing file and return a dataframe of the different observations

    Parameters
    ----------
    path: path to the pointing file

    Output
    ------
    dataframe with the pointing information.
        Index is: _vis_it #, _mos_aic # within the visit, _obs_ # within the mosaic

    """
    path = Path(path) # make sure it's a Path object
    with open(str(path)) as f:
        text = f.read()
    # split the text up using the line that separates observations
    obs_break = "========================================"
    observations = text.split(obs_break)[1:]
    visits = [parse_observation(chunk) for chunk in observations]
    df =  pd.concat({(v[0], v[1]): v[2] for v in visits})
    # give the index names
    # sequence - a series of related observations, like TA and dithers
    # obs - an observation within that sequence, can be multiple exposures
    df.index.names = ['vis', 'mos', 'obs']
    return df


def parse_jwst_exposure_filename(filename):
    """
    Take a filename from the JWST pipeline stage 0-2 (single-exposure files) and parse the information
    see specs here: https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html
    jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>(-<”seg”NNN>)_<detector>_<prodType>.fits
    where
        ppppp: program ID number
        ooo: observation number
        vvv: visit number
        gg: visit group
        s: parallel sequence ID (1=prime, 2-5=parallel)
        aa: activity number (base 36)
        eeeee: exposure number
        segNNN: the text “seg” followed by a three-digit segment number (optional)
        detector: detector name (e.g. ‘nrca1’, ‘nrcblong’, ‘mirimage’)
        prodType: product type identifier (e.g. ‘uncal’, ‘rate’, ‘cal’)

    Parameters
    ----------
    filename: str or path

    Output
    ------
    Define your output

    """
    filename = Path(filename)
    parts = filename.stem.split("_")# strip the suffix

    fname_dict = {}

    fname_dict['path'] = str(filename.resolve().parent)
    fname_dict['filename'] = filename.name

    fname_dict['prog_id'] = parts[0][2:7]
    fname_dict['obs_num'] = parts[0][7:10]
    fname_dict['vis_num'] = parts[0][10:]

    fname_dict['vis_grp'] = parts[1][:2]
    fname_dict['pll_seq'] = parts[1][2:3]
    fname_dict['act_num'] = parts[1][3:]

    fname_dict['exp_num'] = parts[2][:5]
    fname_dict['seg_num'] = 'none'
    if parts[2].find('-') >= 0:
        fname_dict['seg_num'] = parts[2].split('-')[1][3:]
    fname_dict['detector'] = parts[3]
    fname_dict['prod_type'] = parts[4]
    return fname_dict

def parse_jwst_filename_stage3(filename):
    """
    Parse a JWST pipeline stage3 filename
    specs here: https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html

    jw<ppppp>-<AC_ID>_[<”t”TargID | “s”SourceID>](-<”epoch”X>)_<instr>_<optElements>(-<subarray>)_<prodType>(-<ACT_ID>).fits
    where
        ppppp: Program ID number
        AC_ID: Association candidate ID
        TargID: 3-digit Target ID (either TargID or SourceID must be present)
        SourceID: 5-digit Source ID
        epochX: The text “epoch” followed by a single digit epoch number (optional)
        instr: Science instrument name (e.g. ‘nircam’, ‘miri’)
        optElements: A single or hyphen-separated list of optical elements (e.g. filter, grating)
        subarray: Subarray name (optional)
        prodType: Product type identifier (e.g. ‘i2d’, ‘s3d’, ‘x1d’)
        ACT_ID: 2-digit activity ID (optional)

    Parameters
    ----------
    filename : str or path
        stage 3 filename

    Output
    ------
    dict of parameters

    """
    fname_dict = {}
    filename = Path(filename)
    name = filename.name
    prog_id, stage3_info = name.split('-')
    stage3_info = stage3_info.split("_")

    fname_dict['prog_id'] = prog_id[2:]
    fname_dict['ac_id'] = stage3_info.pop(0)

    # target ID, source ID, and epoch ID
    targ_text = stage3_info.pop(0)
    targ_pattern = '[t,s]([\w]*)'
    epoch_pattern = 'epoch[0-9]'
    targ_match = re.search(targ_pattern, targ_text)
    targ_id = None if (targ_match is None) else targ_match.group()
    epoch_match = re.search(epoch_pattern, targ_text)
    epoch_id = None if (epoch_match is None) else epoch_match.group()
    fname_dict['targ_id'] = targ_id
    fname_dict['epoch_id'] = epoch_id

    # Instrument, optical element, and subarray
    fname_dict['instr'] = stage3_info.pop(0)
    opt_el, *subarray =  stage3_info.pop(0).split('-')
    fname_dict['opt'] = opt_el
    fname_dict['subarray'] = None if (subarray == []) else subarray

    # Product type and optional Activity ID
    prodType, *act_id  = stage3_info.pop(0).split('-')
    fname_dict['prodType'] = prodType
    fname_dict['act_id'] = None if (act_id == []) else act_id

    fname_dict['path'] = str(filename.parent.resolve())
    fname_dict['filename'] = filename.name
 
    return fname_dict


def parse_mast_filename(filename):
    """
    Take a filename from MAST/the JWST pipeline and parse the filename to
    get the observation information

    Parameters
    ----------
    filename: filename from MAST

    Output
    ------
    dict with the program, visit, observation, and other information

    """
    filename = Path(filename)
    if filename.name[7] == '-':  # stage 3 signifier
        fname_dict = parse_jwst_filename_stage3(filename)
    else:
        fname_dict = parse_jwst_exposure_filename(filename)

    return fname_dict

def organize_mast_files(files: list, extra_keys: dict[str, int] = {}):
    """
    Take a list of the files from MAST and then parse the filenames to sort them

    Parameters
    ----------
    files: list of files, possibly pathlib.Path objects
    extra_keys: a list of extra keywords you want to add to the dataframe. Format is key: header_int

    Output
    ------
    data organizer dataframe

    """
    # only do stage 2 files
    for i in range(len(files))[::-1]:
        if Path(files[i]).name[7] == '-': # file is a stage 3 file, drop it
            ff = files.pop(i)
    data_organizer = pd.concat([pd.Series(parse_mast_filename(Path(fname)))
                                for fname in files], axis=1).T
    data_organizer.reset_index(inplace=True)
    data_organizer.rename(columns={'index': 'id'}, inplace=True)
    # add any extra keyword values
    for key, hdr in extra_keys.items():
        data_organizer[key.lower()] = data_organizer.apply(lambda row: fits.getval(Path(row['path']) / row['filename'], key.upper(), hdr), axis=1)
        pass
    return data_organizer


def organize_files_by_header(files, ext=0):
    """
    Take a bunch of FITS files and combine their headers into a dataframe for sorting

    Parameters
    ----------
    files: list of strings or pathlib.Paths
      list of paths to fits files from jwst
    ext : int
      number of the extension (0=PRI, 1=SCI, 2=ERR, etc)

    Output
    ------
    hdr_df : pd.DataFrame
      dataframe of all the file headers

    """
    # make a list of headers and format them
    hdrs = []
    for f in files:
        hdr = fits.getheader(str(f), ext)
        hdr = pd.Series(hdr)
        # drop duplicated index entries, usually this is "''"" and "COMMENT"
        drop_index = hdr.index[hdr.index.duplicated()]
        # hdr = hdr[~hdr.index.duplicated()]
        hdr.drop(index=drop_index, inplace=True)
        # also drop all instances of "''" and 'COMMENT'
        for label in ['','COMMENT']:
            try:
                hdr.drop(labels=label)
            except KeyError:
                # probably this means there are no entries with this index label
                pass
        hdrs.append(hdr)
    hdr_df = pd.concat(hdrs, axis=1).T
    return hdr_df
