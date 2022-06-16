"""
Script to run the pipeline on a set of files
Usage:
conda activate jwst_inspector (or other env with the jwst pipeline)
- define dict of stage 1, 2, and 3 parameters
- get a list of files
- call run_pipeline() with:
  - the list of files
  - the stage of those files
  - the list of stages to run
  - a place to put the output
  - the parameter arguments for the stages, if not default
"""

from pathlib import Path
from datetime import datetime

# pandas dataframes are handy ways of tracking progress
import pandas as pd

# pipeline imports
import jwst
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
from jwst.associations.asn_from_list import asn_from_list

# this library is useful for parsing mast filenames
from . import read

PIPELINE_VERSION = jwst.__version__

# convenience dictionary for getting stages
# format: stage_id: stage class
stages = {'2a': Detector1Pipeline,
          '2b': Image2Pipeline,
          '3' : Coron3Pipeline}


def print_usage():
    """Print usage information"""
    help_string = """
    Script to run the pipeline on a set of files
    Usage:
    conda activate jwst_inspector (or other env with the jwst pipeline)
    - define dict of stage 1, 2, and 3 parameters
    - get a list of files
    - call run_pipeline() with:
      - the list of files
      - the stage of those files
      - the list of stages to run
      - a place to put the output
      - the parameter arguments for the stages, if not default
    - check the output folder for the log file and output
    """
    print(help_string)

"""
How does the file manager work?
Example:
| index | id | filename        | stage | prev_stage_file_id |
|-------+----------------------+-------+--------------------|
|     0 |  0 | name_uncal.fits |     1 |               None |

The index is irrelevant.
id : unique identifier for that file
filename : the file name
stage: which stage is this file? 1, 2a, 2b, or 3
prev_stage_file_id : which file or files are the progenitors for this one

When a new file is generated, it will be added to the end of the file manager dataframe
"""
def initialize_files(files: list, stage: str):
    """
    Put the files into a pandas dataframe that will be used to organize the way
    they are processed through the stages.
    """
    # df = pd.DataFrame([(i.name, i.parent) for i in files], columns=['filename', 'path'])
    df = read.organize_mast_files(files)
    df.rename(columns={'index': 'file_id'})
    df['stage'] = stage
    df['prev_stage_file_id'] = pd.NA
    return df


"""
Run stages
Need to be able to:
 - select which stages to run
 - pass custom parameters
 - override reference files
 - collect files together
"""

# trying to write a generic pipeline is too complicated. Write a stage-specific one.
def run_stage2a(row, params):
    """
    Run pipeline stage 2a (uncal -> rate/rateints) with the given parameters

    Parameters
    ----------
    row: row of the file manager dataframe
    params: stage parameters

    Output
    ------
    no output; write files to disk

    """
    filename = Path(row['path']) / row['filename']
    stage = stages['2a']
    res = stage.call(str(filename), **params)

    # find all the output files
    outname_stem = str(filename.stem).replace('uncal', 'rate')
    output_files = Path(params['output_dir']).glob(outname_stem+"*.fits")

    stage_manager = []
    for f in output_files:
        # these are the fields that need to be updated for the output
        output_info = {'path': Path(params['output_dir']).resolve(),
                       'filename': str(f),
                       'prod_type': f.stem.split('_')[-1],
                       'stage': '2a',
                       'prev_stage_file_id': row['id']}
        # copy over the row data and update the relevant fields
        stage_manager.append(pd.Series({k: row[k] for k in row.index}))
        for k, v in output_info.items():
            stage_manager[-1][k] = v

    return pd.concat(stage_manager, ignore_index=True)

def run_stage2b(row, params):
    """
    Run pipeline stage 2b (rate/rateints -> cal/calints) with the given parameters

    Parameters
    ----------
    row: row of the file manager dataframe
    params: stage parameters

    Output
    ------
    no output; write files to disk

    """
    filename = Path(row['path']) / row['filename']
    stage = stages['2b']
    res = stage.call(str(filename), **params)

    # for stage 2b, there should only be one output file
    outname = str(filename.name).replace('rate', 'cal')
    output_file = Path(params['output_dir']) / outname
    try:
        assert(output_file.exists)
    except AssertionError:
        output_file = None

    # these are the fields that need to be updated for the output
    output_info = {'path': Path(params['output_dir']).resolve(),
                   'filename': str(outname),
                   'prod_type': output_file.stem.split('_')[-1],
                   'stage': '2b',
                   'prev_stage_file_id': row['id']}
    # copy over the row data and update the relevant fields
    stage_manager = pd.Series({k: row[k] for k in row.index})
    for k, v in output_info.items():
        stage_manager[k] = v
    return stage_manager



"""
Pipeline wrapper
"""
default_params = {'2a': {'save_results': True},
                  '2b': {'save_results': True},
                  '3' : {'save_results': True}}
def run_pipeline(input_files: list, input_stage: str, stages_to_run: list,
                 output_folder: str = '.', params: dict = {}):
    """
    Organize and manage running the requested pipeline stages

    Parameters
    ----------
    input_files: list
      A list of files to process (can be strings or pathlib.Paths)
    input_stage: integer [(0, 1, 2)]
      The stage of the input files
    stages_to_run: list-like
      A list of which stages to run [(1, 2, 3)]
      If Stage 3 is chosen, params must contain an association file
    output_folder: string or pathlib.Path ["."]
      Folder in which to store the pipeline output
    params: dictionary [{}]
      Dictionary of stage parameters. Includes reference file overrides.
      Format is {'2a': {'step_name': {'arg': val pairs}} for e.g. stage 2a

    Output
    ------
    Pipeline output and log file are written to provided output folder.
    The log file is `output_folder / pipeline.log`.

    """
    # Set up logging
    logfile_path = Path(output_folder) / "pipeline.log"
    logfile = open(logfile_path, 'w')

    print(f"Running pipeline version {PIPELINE_VERSION}", file=logfile)

    # set default values without overriding, and output directory
    for k1 in params.keys():
        for k2, v2 in default_params[k1].items():
            params[k1].setdefault(k2, v2)
            params[k1]['output_dir'] = output_folder

    run_manager = initialize_files(input_files, input_stage)
    # loop through stages and run them
    for stage_id in sorted(stages_to_run):
        if stage_id == '2a':
            results = run_manager.apply(run_stage2a, params=params['2a'], axis=1)
            results['id'] = results.index + len(run_manager)
            run_manager = run_manager.append(results, ignore_index=True)
        elif stage_id == '2b':
            results = run_manager.apply(run_stage2b, params=params['2b'], axis=1)
            results['id'] = results.index + len(run_manager)
            run_manager = run_manager.append(results, ignore_index=True)
        elif stage_id == '3':
            pass
        else:
            print(f"No such stage_id {stage_id} (in {stages_to_run})")
    return run_manager
