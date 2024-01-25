#!/usr/bin/env python
# 
# Script to retrieve OSS event messages for a given visit from the engineering DB
# By Marshall Perrin, based on engdb_visit_times.py by Jeff Valenti


import os
import re
import argparse
from csv import reader
from datetime import datetime, timedelta, timezone
from getpass import getpass
from requests import Session
from time import sleep

from astropy import time, units

def get_ictm_event_log(
        startdate : str,
        enddate : str,
        mast_api_token : str = '',
        verbose : bool = False,
) -> str:
    """
    Get the event log between a start time and an end time
    Parameters:
    startdate: str
      starting time in iso format
    enddate: str
      starting time in iso format
    mast_api_token : str ['']
      MAST token from https://auth.mast.stsci.edu/tokens
    verbose : bool [False]
      if True, print debugging messages
    """
    # parameters
    mnemonic = 'ICTM_EVENT_MSG'

    # constants
    base = 'https://mast.stsci.edu/jwst/api/v0.1/Download/file?uri=mast:jwstedb'
    mastfmt = '%Y%m%dT%H%M%S'
    millisec = timedelta(milliseconds=1)
    tz_utc = timezone(timedelta(hours=0))
    colhead = 'theTime'

    # set or interactively get mast token
    if not mast_api_token:
        mast_api_token = os.environ.get('MAST_API_TOKEN')
        if mast_api_token is None:
            raise ValueError("Must define MAST_API_TOKEN env variable or specify mast_api_token parameter")


    # establish MAST session
    session = Session()
    session.headers.update({'Authorization': f'token {mast_api_token}'})

    # fetch event messages from MAST engineering database (lags FOS EDB)
    start = datetime.fromisoformat(startdate)
    end = datetime.fromisoformat(enddate)
    # end = datetime.now(tz=tz_utc)
    startstr = start.strftime(mastfmt)
    endstr = end.strftime(mastfmt)
    filename = f'{mnemonic}-{startstr}-{endstr}.csv'
    url = f'{base}/{filename}'

    if verbose:
        print(f"Retrieving {url}")
    response = session.get(url)
    if response.status_code == 401:
        exit('HTTPError 401 - Check your MAST token and EDB authorization. May need to refresh your token if it expired.')
    response.raise_for_status()
    events = response.content.decode('utf-8')#.splitlines()

    return events


def extract_oss_event_msgs_for_visit(
        eventlog : str,
        selected_visit_id : str,
        ta_only : bool =False,
        verbose : bool =False
):
    # parse response (ignoring header line) and print new event messages
    vid = ''
    in_selected_visit = False
    in_ta = False

    if verbose:
        print(f"\tSearching for visit: {selected_visit_id}")
    for value in reader(eventlog, delimiter=',', quotechar='"'):
        if in_selected_visit and ((not ta_only) or in_ta) :
            print(value[0][0:22], "\t", value[2])

        if value[2][:6] == 'VISIT ':
            if value[2][-7:] == 'STARTED':
                vstart = 'T'.join(value[0].split())[:-3]
                vid = value[2].split()[1]

                if vid==selected_visit_id:
                    print(f"VISIT {selected_visit_id} START FOUND at {vstart}")
                    in_selected_visit = True
                    if ta_only:
                        print("Only displaying TARGET ACQUISITION RESULTS:")

            elif value[2][-5:] == 'ENDED' and in_selected_visit:
                assert vid == value[2].split()[1]
                assert selected_visit_id  == value[2].split()[1]

                vend = 'T'.join(value[0].split())[:-3]
                print(f"VISIT {selected_visit_id} END FOUND at {vend}")


                in_selected_visit = False
        elif value[2][:31] == f'Script terminated: {vid}':
            if value[2][-5:] == 'ERROR':
                script = value[2].split(':')[2]
                vend = 'T'.join(value[0].split())[:-3]
                dur = datetime.fromisoformat(vend) - datetime.fromisoformat(vstart)
                note = f'Halt in {script}'
                in_selected_visit = False
        elif in_selected_visit and value[2].startswith('*'): # this string is used to mark the start and end of TA sections
            in_ta = not in_ta


def extract_visit_events(
        logtxt : str,
        visit_id : str
) -> str:
    """
    This function takes a log entry that starts before and ends after some
    visit. It will crop out everything except the messages from the visit you're
    interested in, and return only those.

    Parameters
    ----------
    logtxt : str
      the event messages from the engineering db, encompassing the visit messages
    visit_id : str
      11-char visit id taken from the 'VISIT_ID' entry of the PRI fits header
      note: does NOT have a 'V' in front

    Output
    ------
    visit_text : str
      the event log containing only the messages between the start and end of the specified visit

    """
    # add the preceding 'V' to the visit_id
    if visit_id[0] != 'V' and len(visit_id) == 11:
        visit_id = 'V'+visit_id
    pattern=fr"""(?=VISIT\ {visit_id}\ STARTED) # find the start of this visit
                 (.*)                           # match everything including newlines (with re.DOTALL)
                 (?<=VISIT\ {visit_id}\ ENDED)  # find the end of this visit"""
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)
    match =  regex.search(logtxt)
    if match is not None:
        num_chars_to_BOL = 44 # number of characters to the beginning of the line
        visit_text = logtxt[match.start()-num_chars_to_BOL : match.end()][:]
    else:
        # failure mode: return no visit log
        print(f"No log found for Visit {visit_id}")
        visit_text = ''
    return visit_text



def extract_MIRTAMAIN_events(
        visit_events : str,
        obs_id : str
) -> str:
    """
    From a single visit log, extract the MIRTAMAIN events

    Parameters
    ----------
    visit events: str
      the events for just that visit
    obs_id: str
      the OBS_ID keyword from the header

    Output
    -----
    ta_text : str
      the events related to the TA script (MIRTAMAIN)
    """
    pattern = fr"(?=Script activated: {obs_id}:MIRTAMAIN)(.*)(?<=Script terminated: {obs_id}:MIRTAMAIN)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    match = regex.search(visit_events)
    if match is None:
        return ''
    ta_text = match.group()
    return ta_text


def split_ta_stages(ta_text):
    """
    From the TA log, split the text up into the two TA stages. If a stage is missing,
    handle it.

    Parameters
    ----------
    ta_text : str
      Event logs from the TA script

    Output
    ------
    stage_text : dict(str, str)
      a dict with the text for each TA stage, or None
    """
    # we only want successful TA attempts, so let's filter on events
    # where MIR_TACQ completed successfully
    start = 'MIR_TACQ completed'
    end = "SUCCESS"
    pattern = fr"(?={start})(.*?)(?<={end})"
    regex = re.compile(pattern, re.DOTALL)

    # some hacky looping to collect all the snippets
    match = regex.search(ta_text, 0)
    if match is None:
        return None
    matches = [match]
    while match is not None:
        # start the next search were the last one left off
        last_match = matches[-1]
        match = regex.search(ta_text, last_match.end())
        matches.append(match)
    matches.pop(-1)
    ta_stages = dict(enumerate([i.group() for i in matches]))
    return ta_stages


def extract_psf_centroids(ta_stages):
    """
    From the TA stage event entries, get the two centroids. If both centroids are not there,
    handle it.

    Parameters
    ----------
    ta_stages : dict({ str: str })
      Event logs from the TA script

    Output
    ------
    centroids : dict(str, tuple(float))
      a dict with keys (1, 2) for the first and second phase of TA
      each entry is a tuple of floats for the 1-indexed detector coords
      of the PSF centroid

    """
    # now get the centroids for each TA stage
    pattern = "detector coord \(colCentroid, rowCentroid\)       \= \([0-9]+\.[0-9]+, [0-9]+\.[0-9]+\)"
    centroid_order = {0: 'outer', 1: 'inner'}
    centroids = {}
    for stage, text in ta_stages.items():
        regex = re.compile(pattern)
        coords = regex.search(text)
        if coords is not None:
            coords = coords.group()
            centroids[centroid_order[stage]] = tuple(float(i) for i in coords.split(' = ')[1][1:-1].split(', '))
        else:
            centroids[centroid_order[stage]] = None
    return centroids


def ta_centroids_wrapper(
        pri_hdr : dict,
        mast_api_token : str,
        verbose : bool =False
):
    """
    Put it all together

    Parameters
    ----------
    pri_hdr : dict-like
      any dict-like object that has all the required keywords from the primary header:
      'VISIT_ID', 'OBS_ID', 'DATE-BEG', 'DATE-END'

    Output
    ------
    centroids: dict(str: tuple(float))
      psf centroids for each TA step
    """
    visit_id = pri_hdr['VISIT_ID']

    start_time = time.Time(pri_hdr['DATE-BEG']) - 2*units.hour
    end_time = time.Time(pri_hdr['DATE-END']) + 2*units.hour

    # get all messages within 2 hours of the start and end of the visit
    events = get_ictm_event_log(start_time.to_string(), end_time.to_string(),
                                mast_api_token=mast_api_token, verbose=verbose)

    # filter down to just events related to the visit
    visit_events = extract_visit_events(events, visit_id)
    if visit_events is None:
        return None

    # now filter down to just the TA events
    ta_text = extract_MIRTAMAIN_events(visit_events, pri_hdr['OBS_ID'])
    if ta_text is None:
        return None

    ta_stages = split_ta_stages(ta_text)
    if ta_stages is None:
        return None
    # and now, get the centroids
    centroids = extract_psf_centroids(ta_stages)
    return centroids
# for backwards compatibility
get_ta_centroids_from_engdb = ta_centroids_wrapper


# if __name__=="__main__":
#     main(visit_id='V02153002001',
#          mast_api_token='23dba78d48fc4c439e8abdb1f3859040',
#          start_date='2023-08-31T11:08:54.599',
#          ta_only=False,
#          verbose=True)



