"""
Script to run the pipeline on a set of files
Usage:
conda activate jwst_inspector (or other env with the jwst pipeline)
- define dict of stage 1, 2, and 3 parameters
- get a list of files
- call run_pipeline() with:
  - the list of files
  - the list of stages to run
  - a place to put the output
  - the parameter arguments for the stages, if not default
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

import json

import jwst
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
from jwst.associations.asn_from_list import asn_from_list

from . import read

PIPELINE_VERSION = jwst.__version__

def print_usage():
    """Print usage information"""
    pass


def initialize_files(files, stage):
    """
    Put the files into a pandas dataframe that will be used to organize the way
    they are processed through the stages.
    """
    # df = pd.DataFrame([(i.name, i.parent) for i in files], columns=['filename', 'path'])
    df = read.organize_mast_files(files)
    df['stage'] = stage
    return df

"""
Run stages
Need to be able to:
 - select which stages to run
 - pass custom parameters
 - override reference files
 - collect files together
"""


def run_stage1(files, stage1_params):
    pass

def run_stage2(files, stage2_params):
    pass

def run_stage3(assn_file, stage3_params):
    pass

"""
Pipeline wrapper
"""
pipeline_params = {'stage1': None,
                  'stage2': None,
                  'stage3': None}
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
