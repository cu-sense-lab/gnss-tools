
import logging
from typing import Dict, List, Tuple, Optional
import requests
from gnss_lib.misc.data_utils import format_filepath
from datetime import datetime, timedelta
import os
import gnss_lib.rinex_io.rinex3 as rinex3
import hatanaka
from gnss_lib.misc.data_utils import http_download

DATA_DIR = os.environ.get("DATA_DIR")

CDDIS_ARCHIVE_URL = f"https://cddis.nasa.gov/archive/"
RINEX3_ARCHIVE_PATH_TEMPLATE = "gnss/data/daily/{yyyy}/{ddd}/{yy}d/"

EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD")
EARTHDATA_AUTH = (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)

if DATA_DIR is None:
    logging.warning("DATA_DIR environment variable not set.")
if EARTHDATA_USERNAME is None:
    logging.warning("EARTHDATA_USERNAME environment variable not set.")
if EARTHDATA_PASSWORD is None:
    logging.warning("EARTHDATA_PASSWORD environment variable not set.")


def get_cddis_rinex3_file_list(
    dt: datetime, auth: Optional[Tuple[str, str]] = EARTHDATA_AUTH
) -> Tuple[List[str], List[int]]:

    rinex3_archive_path = format_filepath(RINEX3_ARCHIVE_PATH_TEMPLATE, dt)
    url = CDDIS_ARCHIVE_URL + rinex3_archive_path
    url += "*?list"
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        raise Exception("Failure to retrieve list: {0}".format(url))
    items = r.content.decode("utf-8").splitlines()
    file_names, file_sizes = zip(*map(lambda x: x.split(), items))
    return list(file_names), list(file_sizes)


def find_and_download_igs_station_rinex_obs(
        day: datetime,
        station_names: List[str],
        verbose: bool = False,
        strict: bool = True
) -> Dict[str, rinex3.Dataset]:
    filenames, _ = get_cddis_rinex3_file_list(day, auth=EARTHDATA_AUTH)

    # download station data for relevant IGS stations
    relevant_rinex3_filenames = {}
    for station_name in station_names:
        for fname in filenames:
            if fname.startswith(station_name):
                relevant_rinex3_filenames[station_name] = fname

    overwrite = False
    rinex3_zipped_filepaths = {}
    for station_name, filename in relevant_rinex3_filenames.items():
        rinex3_archive_path = format_filepath(RINEX3_ARCHIVE_PATH_TEMPLATE, day)
        url = os.path.join(CDDIS_ARCHIVE_URL, rinex3_archive_path, filename)
        if verbose:
            print(f"Downloading {url}", end="... ")
        output_filepath = os.path.join(DATA_DIR, "cddis", rinex3_archive_path, filename)
        if os.path.exists(output_filepath) and not overwrite:
            if verbose:
                print("Already downloaded")
            rinex3_zipped_filepaths[station_name] = output_filepath
            continue
        success = http_download(url, output_filepath, auth=EARTHDATA_AUTH)
        if success:
            if verbose:
                print("Success")
            rinex3_zipped_filepaths[station_name] = output_filepath
        else:
            if verbose:
                print("Failed")

    # Directly read RINEX3 files with Hatanaka decompression
    rinex3_datasets = {}
    for station_name, filepath in rinex3_zipped_filepaths.items():
        print(f"Reading {filepath}")
        filepath = hatanaka.decompress_on_disk(filepath)
        dataset = rinex3.Dataset().load_files([filepath], strict)
        rinex3_datasets[station_name] = dataset

    return rinex3_datasets