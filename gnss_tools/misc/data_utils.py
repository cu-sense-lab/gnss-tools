"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from ftplib import FTP, FTP_TLS
from typing import Optional, Tuple
import requests
import os.path

import datetime
import logging
import zipfile
import tarfile
import gzip
import shutil
import subprocess

from gnss_tools.time.gpst import GPSTime


def format_filepath(filepath_expr: str, dt: Optional[datetime.datetime | datetime.date] = None, **kwargs) -> str:
    """
    ------------------------------------------------------------------------
    `filepath_expr` -- a formattable string that defines the full filepath.
        Intended to mirror filepath format on website / FTP site (should be
        mirrored by folder structure of downloaded files).  See below about
        placeholders.

    Placeholders:
        Use the Python format syntax for adding placeholders to the path or
        file expressions.  E.g. `/path/to/data/{type}/`.  These will be
        replaced via `.format` using the date/time arguments derived from
        `dt` and the keywords arguments from `kwargs`

    Reserved Placeholders:
        For date/time-related expressions in the path or filename
        expressions, use the placeholders used on IGS products page (e.g.
        wwww, yyyy, ddd, etc).  When passing in a datetime into
        `format_filepath(<filepath>, dt=datetime)`, this function computes
        all the necessary placeholders.

    Example:
        IGS MGEX path expression: /pub/gps/products/mgex/{wwww}/
        IGS MGEX sp3 file expression: com{wwww}{d}.sp3.Z
        New MGEX SP3 file expression: COD0MGXFIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz

    Note on `path_expression` leading '/':
        Strips leading forward slash in path expression to avoid issues
        with `join`.  Path expression is always relative to some host URL
        or local data directory.

    Note for SP3:
        For MGEX (basepath = '/pub/igs/products/mgex'), we can use the `com` files which include all available GNSS data
        When using basepath pub/products/ ...:
            for GPS use 'igs';
            for GLONASS use 'igl' (after GPS week 1300) or 'igx' (before GPS week 1300)
    """
    if dt is not None:
        if isinstance(dt, datetime.date):
            dt = datetime.datetime(dt.year, dt.month, dt.day)
        gpst = GPSTime.from_datetime(dt)
        week_num = gpst.week_num
        week_day = int(gpst.tow.to_float() / 86400)
        yyyy = "{0:04}".format(dt.year)  # year
        yy = "{0:02}".format(dt.year % 100)  # 2-digit year
        mm = "{0:02}".format(dt.month)  # month
        wwww = "{0:04}".format(week_num)  # gps week no.
        ddd = "{0:03}".format(dt.timetuple().tm_yday)  # day of year
        d = "{0:01}".format(int(week_day))  # day of week
        date_params = {
            "yyyy": yyyy,
            "yy": yy,
            "mm": mm,
            "wwww": wwww,
            "ddd": ddd,
            "d": d,
        }
        kwargs.update(date_params)
    return filepath_expr.format(**kwargs)


def fix_bad_zip_file(zip_filepath: str):
    """See here:  https://stackoverflow.com/questions/3083235/unzippng-file-results-in-badzipfile-file-is-not-a-zip-file"""
    with open(zip_filepath, "rb") as f:
        data = f.read()
        pos = data.find(b"\x50\x4b\x05\x06")  # End of central directory signature
        if pos > 0:
            logging.warning("Truncating file at location " + str(pos + 22) + ".")
            f.seek(pos + 22)  # size of 'ZIP end of central directory record'
            f.truncate()
            f.close()
        else:
            # raise error, file is truncated
            raise Exception("Bad zip file: file is truncated")


def decompress(filepath: str, output_filepath: str) -> bool:
    """
    ----------------------------------------------------------------------------
    Given the path to file `filepath`, determines whether one of the modules
    `zipfile`, `tarfile`, `gzip`, or (if available) `rarfile` is capable of
    reading and decompressing the file.  Then, decompresses file and writes to
    `output_filepath`.  Returns False if decompression fails.

    Notes:  `gzip` module will only be used if filepath ends in '.gz'
    """
    path = os.path.dirname(filepath)
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, "r") as f:
            res = f.extractall(path)
        return os.path.join(path, os.path.basename(output_filepath)) == output_filepath
    if tarfile.is_tarfile(filepath):
        with tarfile.TarFile(filepath, "r") as f:
            f.extractall(path)
        return os.path.join(path, os.path.basename(output_filepath)) == output_filepath
    if filepath[-3:] == ".gz":
        with gzip.open(filepath, "rb") as f_in:
            with open(output_filepath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    if filepath[-2:] in [".Z", ".z"]:
        # Unix compression; use uncompress
        res = subprocess.call("uncompress -f " + filepath, shell=True)
        if res == 0:
            return filepath[:-2] == output_filepath
    # check for bad zip file
    if filepath[-4:] in [".zip"]:
        res = subprocess.call("unzip " + filepath, shell=True)
        if res == 0:
            return filepath[:-4] == output_filepath
    return False


def ftp_download(ftp_host: str, url_filepath: str, output_filepath: str):
    """Given URL to data on FTP site and a local output location, uses Python
    FTP library's `retrbinary` function to download file.
    """
    path = os.path.dirname(url_filepath)
    filename = os.path.basename(url_filepath)
    ftp = FTP(ftp_host)
    ftp.login()
    ftp.cwd(path)
    with open(output_filepath, "wb") as f:
        ftp.retrbinary("RETR " + filename, f.write)
    ftp.quit()
    return True


def cddis_download(
    filepath: str,
    output_filepath: str,
    cddis_url: str = "gdc.cddis.eosdis.nasa.gov",
    cddis_username: str = "anonymous",
    cddis_password: str = "cuboulder",
) -> bool:
    # this is deprecated
    logging.warning("cddis_download is deprecated.  Use `ftp_download` or `http_download` instead.")
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    ftps = FTP_TLS(host=cddis_url)
    ftps.login(user=cddis_username, passwd=cddis_password)
    ftps.prot_p()
    ftps.cwd(directory)
    ftps.retrbinary("RETR " + filename, open(output_filepath, "wb").write)
    return True


def http_download(
        url_path: str,
        output_filepath: str,
        auth: Optional[Tuple[str, str]] = None
    ) -> bool:
    """Given URL to HTTP resource and a local output filepath, uses Python's
    `requests.get` to fetch HTTP resource and write to local disc.

    auth -- Tuple of 'username' and 'password'

    Returns True if successful, otherwise raises Exception
    """
    if auth is not None:
        r = requests.get(url_path, auth=auth)
    else:
        r = requests.get(url_path)
    if r.status_code != 200:
        raise Exception("Failure to download: {0}".format(url_path))
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_filepath, "wb") as f:
        f.write(r.content)
    return True

