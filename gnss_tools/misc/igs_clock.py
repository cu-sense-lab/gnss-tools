import os
from typing import List, Optional
import h5py
import numpy as np
from datetime import datetime, timedelta

import scipy.interpolate

from ..time.gpst import GPSTime, convert_datetime_to_gps_seconds, convert_gps_seconds_to_datetime
from .data_utils import cddis_download, format_filepath, decompress
from .hdf5_utils import read_hdf5_into_dict, write_dict_to_hdf5
from ..rinex_io import rinex3


def parse_value(
        value_str: str, dtype=float, default_value=None
) -> Optional[float]:
    """
    Parse a value from a string.  If the value is not valid, return
    the default value.
    """
    try:
        if value_str.strip() == "":
            return default_value
        if dtype is int:
            return int(value_str.strip())
        elif dtype is float:
            return float(value_str.strip())
        else:
            raise ValueError("Unsupported data type")
    except ValueError:
        return default_value


def parse_RINEX3_clk_records(lines, designators: Optional[List[str]] = None):
    """
    See: ftp://igs.org/pub/data/format/rinex_clock300.txt
    """
    data = {}
    lines = iter(lines)
    try:
        line = next(lines)
        while True:
            clock_data_type = line[:2]
            designator = line[3:7].strip()
            if designators is not None and designator not in designators:
                line = next(lines)
                continue
            if designator not in data.keys():
                data[designator] = {"epochs": [], "values": []}
            epoch = line[8:36].strip()
            year = parse_value(epoch[0:4], int)
            month = parse_value(epoch[5:7], int)
            day = parse_value(epoch[8:10], int)
            hour = parse_value(epoch[11:13], int)
            minute = parse_value(epoch[14:16], int)
            seconds = parse_value(epoch[17:27])
            if np.isnan(seconds):
                microseconds = np.nan
            else:
                microseconds = int(1e6 * (seconds % 1))
                seconds = int(seconds)
            dt = datetime(year, month, day, hour, minute, seconds, microseconds)
            #             print(year, month, day, hour, minute, seconds)
            n_values = parse_value(line[36:40], int)
            line = line[40:].rjust(40)
            if not np.isnan(n_values) and n_values > 2:
                line += next(lines).rjust(80)
            values = []
            for i in range(n_values):
                values.append(parse_value(line[i * 20 : (i + 1) * 20]))
            data[designator]["epochs"].append(dt)
            data[designator]["values"].append(values)
            line = next(lines)
    except StopIteration:
        pass
    return data


def parse_RINEX3_clk_file(filepath: os.PathLike, designators: Optional[List[str]] = None):
    """
    ------------------------------------------------------------
    Given the filepath to a RINEX clock file, parses and
    returns header and clock data.

    Input
    -----
    `filepath` -- filepath to RINEX clock file
    `designators` -- list of designators for the objects whose
        clock data should be parsed and stored.  Can be IGS
        4-character station designator or the 3-character
        satellite ID.  If designators is `'all'` (default),
        parses all information in file.

    Output
    ------
    `header, clk_records` where `header` is a dictionary
    containing the parsed header information and `clk_records`
    is a dictionary containing the observation data in the
    format:

        {
            'time': ndarray,
            'satellites': {
                <sat_id>: {
                    'index': ndarray,
                    <obs_id>: ndarray
                }
            }
        }

    Note: `time` in `observations` is in GPST seconds
    """
    with open(filepath, "r") as f:
        header = rinex3.parse_header(header_lines)
        lines = list(f.readlines())
    if len(lines) == 0:
        raise Exception(
            "Error when parsing RINEX 3 file.  The file appears to be empty."
        )
    for i, line in enumerate(lines):
        if line.find("END OF HEADER") >= 0:
            break
    header_lines = lines[: i + 1]
    clk_lines = lines[i + 1 :]
    clk_records = parse_RINEX3_clk_records(clk_lines, designators)
    return header, clk_records


def parse_igs_clk_data(filepaths: List[os.PathLike], designators: Optional[List[str]] = None, save_dir: Optional[os.PathLike]=None, overwrite: bool=False):
    """
    `filepaths` -- list of IGS clock data filepaths to parse
    `designators` -- optional list of sat IDs for which to parse / store clock info
    `save_dir` -- if a valid directory, checks for `os.path.basename(filepath) + '.h5'` in `save_dir`.
        If it finds it, loads parsed clock info from saved H5 file instead of re-parsing text clock file.  This
        can save a lot of time for high-rate clock files.  If it is not there, then saves the parsed clock info as
        an h5 file.
    `overwrite` -- if True, overwrites saved parsed clock files

    Returns:
    `all_epochs, data`

        `all_epochs` -- GPST epochs of clock records
        `clock_data_dict` -- dict with receiver / vehicle IDs as keys and ndarray of records
    """
    # First, parse everything
    headers, clk_data_list = [], []
    clk_ids = set()

    def preprocess_header_and_clk_data(header, clk_data):
        for clk_id in clk_data.keys():
            clk_data[clk_id]["epochs"] = np.array(
                [GPSTime.from_datetime(e).to_float() for e in clk_data[clk_id]["epochs"]]
            )
            clk_data[clk_id]["values"] = np.array(
                [v[0] for v in clk_data[clk_id]["values"]]
            )  # just get clock error
        return header, clk_data

    for filepath in filepaths:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = os.path.join(save_dir, os.path.basename(filepath) + ".h5")
            if os.path.exists(save_filepath) and not overwrite:
                with h5py.File(save_filepath, "r") as f:
                    header = read_hdf5_into_dict(f["header"])
                    clk_data = read_hdf5_into_dict(f["clk_data"])
            else:
                header, clk_data = parse_RINEX3_clk_file(filepath, designators)
                header, clk_data = preprocess_header_and_clk_data(header, clk_data)
                # cannot deal with `comments` in header -- just remove them
                header.pop("comments")
                with h5py.File(save_filepath, "w") as f:
                    write_dict_to_hdf5({"header": header, "clk_data": clk_data}, f)
        else:
            print(filepath)
            header, clk_data = parse_RINEX3_clk_file(filepath, designators)
            header, clk_data = preprocess_header_and_clk_data(header, clk_data)

        headers.append(header)
        clk_data_list.append(clk_data)
        clk_ids |= set(clk_data.keys())

    clk_ids = sorted(list(clk_ids))
    #     for clk_id in clk_ids:
    #         for clk_data in clk_data_list:
    #             if clk_id not in clk_data:
    #                 continue
    #             clk_data[clk_id]['epochs'] = np.array([dt2gpst(e) for e in clk_data[clk_id]['epochs']])
    #             clk_data[clk_id]['values'] = np.array([v[0] for v in clk_data[clk_id]['values']])

    all_epochs = set()
    for clk_id in clk_ids:
        for clk_data in clk_data_list:
            if clk_id not in clk_data:
                continue
            all_epochs = all_epochs | set(clk_data[clk_id]["epochs"])
    all_epochs = np.sort(np.array(list(all_epochs)))

    data = {}
    for clk_id in clk_ids:
        data[clk_id] = np.zeros(len(all_epochs))
        for clk_data in clk_data_list:
            if clk_id not in clk_data:
                continue
            indices = np.searchsorted(all_epochs, clk_data[clk_id]["epochs"])
            data[clk_id][indices] = clk_data[clk_id]["values"]
    return all_epochs, data


def download_and_decompress_igs_clk_file(day: datetime, user_data_dir: os.PathLike, overwrite: bool=False) -> os.PathLike:
    """
    Automatically downloads the IGS CLK products for datetime `dt`
    ----------------------------------------------------------------------------
    Input:
    `day` -- datetime for which to download IGS clock data
    `user_data_dir` -- root directory to store IGS clock data
    Output:
        filepaths corresponding to IGS CLK files
    """
    path_expression = "/gnss/products/{wwww}/"
    file_expression = "cod{wwww}{d}.clk_05s.Z"
    #     file_expression = 'jpl{wwww}{d}.clk.Z'
    ftp_host = "cddis.gsfc.nasa.gov"

    url_filepath = format_filepath(os.path.join(path_expression, file_expression), day)
    download_filepath = (
        user_data_dir + url_filepath[1:]
    )  # cannot join with `/` at beginning
    os.makedirs(os.path.dirname(download_filepath), exist_ok=True)
    decompressed_filepath = download_filepath[:-2]
    if overwrite or not os.path.exists(decompressed_filepath):
        if overwrite or not os.path.exists(download_filepath):
            try:
                download_success = cddis_download(url_filepath, download_filepath)
            except Exception as e:
                raise Exception(
                    "Error in downloading IGS CLK file: {0}".format(download_filepath)
                ) from e
    if os.path.exists(download_filepath) and not os.path.exists(decompressed_filepath):
        try:
            decompress_success = decompress(download_filepath, decompressed_filepath)
            print(decompress_success, "AAAA")
        except Exception as e:
            raise Exception(
                "Error in decompressing IGS CLK file: {0}".format(download_filepath)
            ) from e
    if not os.path.exists(decompressed_filepath):
        raise Exception(
            "Error in download / decompress of file: {0}".format(decompressed_filepath)
        )
    return decompressed_filepath


def download_and_parse_igs_clk_data(
    start_time_gpst: int,
    end_time_gpst: int,
    data_dir: os.PathLike,
    designators: Optional[List[str]] = None,
    verbose=False,
    save_dir=None,
    overwrite=False,
):
    """
    `start_time_gpst` -- earliest time at which clock data is required
    `end_time_gpst` -- latest time at which clock data is required
    `data_dir` -- root directory from which clock data is stored
    `designators` -- the clock IDs corresponding to the receivers/satellites for which clock data should be parsed
    `verbose` -- whether to print progress
    `save_dir` -- directory to cache parsed IGS clock data
    `overwrite` -- whether to overwrite the cached IGS clock data in `save_dir`
    """
    day_start = convert_gps_seconds_to_datetime(start_time_gpst).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end = convert_gps_seconds_to_datetime(end_time_gpst).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(hours=24)

    clk_filepaths = []
    day = day_start - timedelta(days=1)

    while day < day_end + timedelta(days=1):
        if verbose:
            print(
                "\rFetching IGS Clock data... {0}".format(day.strftime("%Y%m%d")),
                end="",
            )

        try:
            clk_filepath = download_and_decompress_igs_clk_file(day, data_dir)
            clk_filepaths.append(clk_filepath)
        except Exception as e:
            print(
                "Failed to download/decompress IGS clock filepath for day: {0}".format(
                    day.strftime("%Y%m%d")
                )
            )
            raise e
        day += timedelta(days=1)

    if verbose:
        print("... done.")
        print("Parsing clock data...", end="")

    clk_data = parse_igs_clk_data(clk_filepaths, designators, save_dir, overwrite)

    if verbose:
        print("done")

    return clk_data
