"""Generate an html page that contains all the visit information"""

import sys
import re
from pathlib import Path
from datetime import date

from requests.sessions import get_environ_proxies
import pandas as pd
import observing_windows as ow

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
    Applies the parse_plan_windows() function to the scraper results
    """
    # programs that have already executed do not have a planWindow column
    # if this is the case, add a dummy column
    if "planWindow" not in program_df.columns:
        program_df["planWindow"] = pd.NA

    parsed = {i: parse_plan_windows(row['planWindow'], [])
              for i, row in program_df.iterrows()}
    parsed = pd.concat(parsed.values(),
                       keys=parsed.keys(),
                       names=['obs_index','obs_window'],
                       axis=0)
    program_df.index.name = 'obs_index'
    df = program_df.merge(parsed, how='inner', on='obs_index').set_index(parsed.index)
    df.drop(columns="planWindow", inplace=True)
    return df


def filter_miri(visit_list):
    """Only keep MIRI observations"""
    pop_list = []
    for i, el in enumerate(visit_list):
        # if "MIRI" not in el['configuration'].upper():
        #     pop_list.append(i)
        if el['configuration'].lower() != 'miri coronagraphic imaging':
            pop_list.append(i)
    pop_list = sorted(pop_list)[::-1]
    for i in pop_list:
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
        programs[info['proposal_id']] = prog2df(info)
        if verbose: 
            print(str(pid) + " finished")
    # combine the programs and drop the dummy indices
    programs = pd.concat(programs, names=['pid']).reset_index()
    programs.drop(columns=['obs_index', 'obs_window'], inplace=True)
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
def body_end_template(table_keys=[]):
    with open(template_path / "body_end_template.txt", 'r') as ff:
        template = ff.read()

    # if there's more than one table to sort, pass the keys
    if table_keys == []:
        pass
    else:
        sort_lines = '\n'.join(f"$('#sortTable_{k}').DataTable();" for k in table_keys)
        template = template.replace("$('#sortTable').DataTable();", sort_lines)
    return template


# start the body, up to the table
def table_start_template(columns, sort_key=''):
    with open(template_path / "table_start_template.txt", 'r') as ff:
        template = ff.read()
    if sort_key != '':
        template = template.replace('id="sortTable"', f' id=sortTable_{sort_key}')
    column_props = 'class="align-middle"'
    column_text = '\n'.join(f'<th {column_props}>{c}</th>' for c in columns)
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
    for i, row in database.iterrows():
        row_data = row_data + "\n" + df2html_row(row)
    return row_data


def write_html(outfile, database):
    """write the visit info to html"""
    with open(outfile, 'w') as ff:
        ff.write(head_template())

        ff.write(body_start_template())

        tables_gb = database.groupby("status")
        table_keys = sorted(tables_gb.groups.keys())
        print("Generating the following tables:")
        print("\t", ", ".join(table_keys))
        # make a separate table for each kind of visit status
        for key, group in tables_gb:
            group_dropna = group.dropna(how='all', axis=1)
            ff.write(f"<br><br>Status: {key.title()}<br>\n")
            ff.write(table_start_template(group_dropna.columns, key))
            row_data = generate_table_rows(group_dropna)
            ff.write(row_data)
            ff.write(table_end_template())

        ff.write(body_end_template(table_keys))


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
    try:
        html_path = Path(sys.argv[1])
    except:
        html_path = Path("~/test.html")
    write_html(str(html_path), programs)
