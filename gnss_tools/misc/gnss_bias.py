"""
Author: Brian Breitsch
Date: 2025-01-02
"""

import os
from gnss_lib.time.gpst import GPSTime
from gnss_lib.misc.data_utils import cddis_download, decompress, format_filepath, http_download
from gnss_lib.rinex_io.sinex_bias import SINEX_Dataset, SINEX_BiasSolutionEntry
from datetime import datetime, timedelta
from typing import List, Dict

CDDIS_BIAS_FILEPATH_TEMPLATE_CAS = (
    "archive/gnss/products/bias/{yyyy}/CAS0OPSRAP_{yyyy}{ddd}0000_01D_01D_DCB.BIA.gz"
)
CDDIS_BIAS_FILEPATH_TEMPLATE_GFZ = (
    "archive/gnss/products/bias/{yyyy}/GFZ0OPSRAP_{yyyy}{ddd}0000_01D_01D_DCB.BIA.gz"
)

def download_and_decompress_gnss_bias_file(
    dt: datetime,
    data_dir: str,
    overwrite: bool = False,
    filepath_template: str = CDDIS_BIAS_FILEPATH_TEMPLATE_CAS
) -> bool:
    """
    Automatically downloads CDDIS DCB SINEX BIA file for datetime `dt`
    Returns the filepath to the downloaded and decompressed SINEX BIA file.
    """
    filepath = format_filepath(filepath_template, dt)
    decompressed_filepath = filepath[:-3]  # remove .gz extension

    output_dir = os.path.join(data_dir, "cddis/")
    output_filepath = os.path.join(output_dir, filepath)
    decompressed_filepath = os.path.join(output_dir, decompressed_filepath)
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))
    downloaded = decompressed = False
    if overwrite or not (
        os.path.exists(output_filepath) or os.path.exists(decompressed_filepath)
    ):
        # downloaded = cddis_download(filepath, output_filepath)
        CDDIS_URL = "https://cddis.nasa.gov/"
        url_path = os.path.join(CDDIS_URL, filepath)
        auth = (os.getenv("EARTHDATA_USERNAME"), os.getenv("EARTHDATA_PASSWORD"))
        downloaded = http_download(url_path, output_filepath, auth)
    if os.path.exists(output_filepath):
        decompressed = decompress(output_filepath, decompressed_filepath)
    
    if os.path.exists(decompressed_filepath):
        return decompressed_filepath
    elif os.path.exists(output_filepath):
        return output_filepath
    else:
        return False
    

def download_and_parse_gnss_bias_data(
    start_time_gpst: GPSTime,
    end_time_gpst: GPSTime,
    data_dir: str,
    verbose: bool = False,
    overwrite: bool = False,
) -> SINEX_Dataset:
    """
    `start_time_gpst` -- earliest time at which SP3 data is required
    `end_time_gpst` -- latest time at which SP3 data is required
    `data_dir` -- root directory from which SP3 data is stored
    `sat_ids` -- the satellite IDs corresponding to the satellites for which splines should
        be computed.  If `None` (default), computes for all satellites in the SP3 files
    """

    day_start = start_time_gpst.to_datetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end = end_time_gpst.to_datetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(hours=24)

    bias_filepaths = []
    day = day_start
    while day < day_end:

        if verbose:
            print(f"\rFetching SINEX BIAS data... {day.strftime('%Y%m%d')}", end="")

        try:
            bias_filepath = download_and_decompress_gnss_bias_file(day, data_dir, overwrite)
            bias_filepaths.append(bias_filepath)
        except Exception as e:
            print(
                f"Failed to download/decompress SINEX BIA filepath for day: {day.strftime('%Y%m%d')}"
            )
            raise e

        day += timedelta(days=1)

    if verbose:
        print("... done.")
        print("Parsing BIAS data...", end="")

    # print(bias_filepaths)
    sinex_dataset = SINEX_Dataset()
    for bias_filepath in bias_filepaths:
        with open(bias_filepath, "r") as f:
            sinex_dataset.parse(f)
            sinex_dataset._filepaths.append(bias_filepath)

    if verbose:
        print(" done.")

    return sinex_dataset





def sort_gnss_bias_solutions(
        bias_entries: List[SINEX_BiasSolutionEntry],
) -> Dict[datetime, Dict[str, Dict[str, Dict[str, Dict[str, List[SINEX_BiasSolutionEntry]]]]]]:
    """
    Returns dict of sorted bias solution entries.  Sorted first by bias start time, then station,
    then PRN, then obs1, then obs2.
    """
    sorted_entries = {}
    for entry in bias_entries:
        if entry.bias_start not in sorted_entries:
            sorted_entries[entry.bias_start] = {}
        if entry.station not in sorted_entries[entry.bias_start]:
            sorted_entries[entry.bias_start][entry.station] = {}
        if entry.prn not in sorted_entries[entry.bias_start][entry.station]:
            sorted_entries[entry.bias_start][entry.station][entry.prn] = {}
        prn_entries = sorted_entries[entry.bias_start][entry.station][entry.prn]
        if entry.obs1 not in prn_entries:
            prn_entries[entry.obs1] = {}
        if entry.obs2 not in prn_entries[entry.obs1]:
            prn_entries[entry.obs1][entry.obs2] = []
        prn_entries[entry.obs1][entry.obs2].append(entry)
    return sorted_entries

def extract_gnss_satellite_code_biases_from_sorted_entries(
        sorted_entries: Dict[datetime, Dict[str, Dict[str, Dict[str, Dict[str, List[SINEX_BiasSolutionEntry]]]]]],
        obs1: str,
        obs2: str,
        strict: bool = True
) -> Dict[datetime, Dict[str, float]]:
    """
    Further refine sorted bias solution entries to only contain satellite code biases for a particular
    observation pair.
    """
    extracted_entries = {}
    for dt, entries in sorted_entries.items():
        for prn, entries in entries[""].items():
            if obs1 not in entries:
                continue
            if obs2 not in entries[obs1]:
                continue
            if len(entries[obs1][obs2]) != 1:
                if strict:
                    raise Exception("Expected only one entry per bias start time")
                else:
                    pass
            entry = entries[obs1][obs2][0]
            if dt not in extracted_entries:
                extracted_entries[dt] = {}
            if prn not in extracted_entries[dt]:
                extracted_entries[dt][prn] = entry.estimated_value
    return extracted_entries