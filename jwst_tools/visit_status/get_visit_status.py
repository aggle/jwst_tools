from pathlib import Path
import requests

programs_file = "jwst_programs.txt"

outdir = Path("visit_files")


def download_visit_files(programs_file=programs_file, outdir=outdir):
    """
    Download the visit files to the output directory

    Parameters
    ----------
    programs_file : str or Path
      a file storing the list of program IDs, one per line
    outdir : str or Path
      the directory in which to write the visit files

    Output
    ------
    writes visit files to the output directory

    """

    with open(programs_file, 'r') as f:
        program_ids = f.readlines()

    for pid in program_ids:
        url = f"https://www.stsci.edu/cgi-bin/get-visit-status?id={pid}&markupFormat=xml&observatory=JWST"
        outfile = outdir / f"visits-{pid}.xml"
        print(f"Getting visit status for PID {pid}")
        r = requests.get(url, allow_redirects=True)
        with open(str(outfile), "wb") as f:
            f.write(r.content)
        print(f"\tVisit status written to {str(outfile)}\n\n")


if __name__ == "__main__":
    download_visit_files(programs_file, outdir)
