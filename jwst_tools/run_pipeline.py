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
def initialize_files(files, stage):
    """
    Put the files into a pandas dataframe that will be used to organize the way
    they are processed through the stages.
    """
    # df = pd.DataFrame([(i.name, i.parent) for i in files], columns=['filename', 'path'])
    df = read.organize_mast_files(files)
    df.rename(columns={'index': 'file_id'})
    df['stage'] = stage
    df['prev_stage_file_id'] = None
    return df


"""
Run stages
Need to be able to:
 - select which stages to run
 - pass custom parameters
 - override reference files
 - collect files together
"""
def run_stage2(stage_id, filename, params):
    """
    Run a pipeline stage 2a or 2b on a given file with given parameters, and return a
    pandas series with the results

    Parameters
    ----------
    stage_id: str
      '2a', '2b', or '3', corresponding to Detector1, Image2, or Coron3
    filename: string or pathlib.Path
      file to process to the next stage
    params: stage parameters

    Output
    ------
    no output, just writing files to disk

    """
    stage = stages[str(stage_id)]
    # stage_manager = pd.Series(None,
    #                           index=['filename', 'pass', 'prod', 'prod_ints'],
    #                           dtype=str)
    # # initialize output columns
    # stage_manager['filename'] = filename
    # stage_manager['pass'] = False # set to True if stage passes
    # stage_manager['prod'] = None
    # stage_manager['prodints'] = None
    # process a file and return the output product(s)
    # f = Path(stage_manager['filename'])
    # the stage.call method creates and runs a new instance of the class
    res = stage.call(str(filename), **params)
    # find all the output files
    output_files = []
    for f in sorted(Path(params['output_dir']).glob("_".join(str(filename).split("_")[:-1])+"*.fits")):
        vals = pd.Series({'filename': str(f), 'stage': stage_id})
        output_files.append(vals)
        print(vals)
    stage_manager = pd.concat(output_files)

    return stage_manager, res


def run_stage1(file_manager, stage1_params):
    """
    Run Stage 1

    Parameters
    ----------
    define your parameters

    Output
    ------
    Define your output

    """
    for i, f in sorted(files['filename'].items()):
        f = Path(f)
        print(f"\n\nProcessing {int(i)+1} of {len(files)} ({str(f.name)})\n\n")
        # Below you can choose how you want to provide the parameters by setting the switch to True
        try:
            det1 = Detector1Pipeline.call(str(f), **stage1_params)
            files.loc[i, 'stage1_pass'] = True
        except:
            pass

def run_stage2(files, stage2_params):
    pass

def run_stage3(assn_file, stage3_params):
    pass

"""
Pipeline wrapper
"""
pipeline_params = {'stage1': {'save_results': True},
                   'stage2': {'save_results': True},
                   'stage3': {'save_results': True}}
def run_pipeline(input_files, input_stage, stages_to_run,
                 output_folder='.', params={}):
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
      Format is {'stageN': {'step_name': {'arg': val pairs}}

    Output
    ------
    Pipeline output and log file are written to provided output folder.
    The log file is `output_folder / pipeline.log`.

    """
    # Set up logging
    logfile_path = Path(output_folder) / "pipeline.log"
    logfile = open(logfile_path, 'w')

    print(f"Running pipeline version {PIPELINE_VERSION}", file=logfile)

    pipeline_params.update(params)
