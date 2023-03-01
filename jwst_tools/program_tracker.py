"""Generate an html page that contains all the visit information"""

import re
from pathlib import Path
from datetime import date

from requests.sessions import get_environ_proxies
import pandas as pd
from . import observing_windows as ow

template_path = Path(__file__).parent / "html_templates"

def parse_plan_windows(window, dates=[]):
    """Parse the 'planWindow' field of the observing window results"""
    # if there's more than one window, use recursion to parse them all
    template = pd.Series({'planWindow-begin_cal': pd.NA,
                          'planWindow-end_cal': pd.NA,
                          'planWindow-begin_dec': pd.NA,
                          'planWindow-end_dec': pd.NA})
    if isinstance(window, list):
        for w in window:
            parse_plan_windows(w, dates)
    # if it's NA, return NA
    elif pd.isna(window):
        dates.append(template)
    # once you get down to a string, parse the string
    else:
        # find the decimal year string because that's more predictable
        dec_dates_fmt = "\([0-9]{4}\.[0-9]{3}\ \-\ [0-9]{4}\.[0-9]{3}\)"
        dec_dates_span = re.search(dec_dates_fmt, window).span()
        # use the indices to separate the two date formats
        dec_dates = window[dec_dates_span[0]+1:dec_dates_span[1]-1].split(' - ')
        cal_dates = window[:dec_dates_span[0]-1].split(" - ")
        dates_tmp = template.copy()
        dates_tmp['planWindow-begin_cal'] = cal_dates[0]
        dates_tmp['planWindow-end_cal'] = cal_dates[1]
        dates_tmp['planWindow-begin_dec'] = dec_dates[0]
        dates_tmp['planWindow-end_dec'] = dec_dates[1]
        dates.append(dates_tmp)
    # if the original window was not a list, then don't return a list
    if not isinstance(window, list):
        dates = pd.DataFrame(dates[0]).T
    else:
        dates = pd.concat(list(dates), axis=1).T
    return dates


def add_plan_windows_to_program_df(program_df):
    """
    Given the results of the web crawl for program info, parse the program windows
    into their own columns for beginning and end dates
    """
    parsed = {i: parse_plan_windows(row['planWindow'], [])
              for i, row in programs_df.iterrows()}
    parsed = pd.concat(parsed.values(),
                       keys=parsed.keys(),
                       names=['row','window'],
                       axis=0)
    program_df.index.name = 'row'
    df = program_df.merge(parsed, how='inner', on='row').set_index(parsed.index)
    df.drop(columns="planWindow", inplace=True)
    return df


def filter_miri(visit_list):
    """Only keep MIRI observations"""
    pop_list = []
    for i, el in enumerate(visit_list):
        if "MIRI" not in el['configuration'].upper():
            pop_list.append(i)
    for i in pop_list[::-1]:
        visit_list.pop(i)    
    return visit_list


def prog2df(info, miri_only=True):
    """Turn the visit information into a dataframe"""
    visits = info['visit']
    # keep only MIRI observations?
    if miri_only == True:
        visits = filter_miri(visits)
    df = pd.concat([ow.pd.Series(i) for i in visits], axis=1).T
    df.rename(columns={"@observation": "observation", "@visit":"visit"}, inplace=True)
    # split the planWindow column into something sortable
    df = add_plan_windows_to_program_df(df)

    return df

def get_program_table(list_of_programs, verbose=True):
    """Given a list of program IDs, get their visit status information"""
    programs = {}
    for pid in list_of_programs:
        print(str(pid))
        info = ow.program_info(pid)
        programs[info['proposal_id']] = prog2df(info['visit'])
        if verbose: 
            print(str(pid) + " finished")
    programs = pd.concat(programs, names=['pid'])
    return programs

### HTML GENERATION ###

# top of the HTML file
def head_template():
    with open(template_path / "head_template.txt", 'r') as ff:
        template = ff.read()
    return template

# start the body, up to the table
def body_start_template():
    with open(template_path / "body_start_template.txt", 'r') as ff:
        template = ff.read()
    today = str(date.today())
    return template.replace("{insert_date_here}", today)

# bottom of the HTML file
def body_end_template():
    with open(template_path / "body_end_template.txt", 'r') as ff:
        template = ff.read()
    return template


# start the body, up to the table
def table_start_template(columns):
    with open(template_path / "table_start_template.txt", 'r') as ff:
        template = ff.read()
    column_text = '\n'.join(f'<th class="align-middle">{c}</th>' for c in columns)
    return template.replace("{replace_text_here_with_columns}", column_text)

# bottom of the HTML file
def table_end_template():
    with open(template_path / "table_end_template.txt", 'r') as ff:
        template = ff.read()
    return template

def generate_table_rows(database):
    def df2html_row(row):
        text = "\n".join(f'<td class="align-middle">{i}</td>' for i in row)
        text = "<tr>" + text + "</tr>"
        return text
    row_data = ""
    for i, row in programs.iterrows():
        row_data = row_data + "\n" + df2html_row(row)
    return row_data

def write_html(outfile, database):
    """write the visit info to html"""
    with open(outfile, 'w') as ff:
        ff.write(head_template())

        ff.write(body_start_template())

        ff.write(table_start_template(database.columns))

        row_data = generate_table_rows(database)
        ff.write(row_data)

        ff.write(table_end_template())

        ff.write(body_end_template())


if __name__ == "__main__":
    prog_ids = [
        1194,
        1241,
        1277,
        1282,
        1413,
        1618,
        1668,
        2243,
        2538,
        2153
    ]
    programs = get_program_table(prog_ids)
    write_html("~/test.html", programs)
