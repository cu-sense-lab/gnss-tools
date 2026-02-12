"""
sp3.py

Utilities for SP3 data file download and parsing

@author Brian Breitsch
@email brian.breitsch@colorado.edu
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from gnss_tools.time.gtime import GTIME_DTYPE
import numpy as np
import scipy.interpolate
from .parse_sp3 import Dataset, SP3Arrays
from ..misc.data_utils import cddis_download, format_filepath, decompress, http_download
from ..time.gpst import GPSTime
from .array_lagrange_interpolation import compute_array_lagrange_interpolation


def download_and_decompress_sp3_file(
    dt: datetime, data_dir: str, overwrite: bool = False
) -> Optional[str]:
    """
    Automatically downloads MGEX SP3 file for datetime `dt`
    Returns the filepath to the downloaded and decompressed SP3 file.
    """
    gpst = GPSTime.from_datetime(dt)
    if gpst.week_num >= 1962:
        # if greater than or equal to GPS week 1962, then use MGEX file format
        filepath_template = (
            "gps/products/{wwww}/COD0MGXFIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz"
        )
    else:
        # else use old file format
        # filepath_template = 'gps/products/{wwww}/com{wwww}{d}.sp3.Z'
        # or try reprocessed data
        filepath_template = (
            "gps/products/{wwww}/repro3/COD0R03FIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz"
        )

    # # The addition of the reprocessed data in `repro3` means that we can use the new format for all dates
    # # Actually nevermind, the new format is not available for all dates using COD0R03FIN
    # filepath_template = 'gps/products/{wwww}/repro3/COD0R03FIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz'

    decompressed_filepath_template = filepath_template[
        :-3
    ]  # just filepath template without '.gz' extension
    filepath = format_filepath(filepath_template, dt)
    decompressed_filepath = format_filepath(decompressed_filepath_template, dt)

    output_dir = os.path.join(data_dir, "cddis/")
    output_filepath = os.path.join(output_dir, filepath)
    decompressed_filepath = os.path.join(output_dir, decompressed_filepath)
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))
    # logging.info(output_filepath, filepath)
    downloaded = decompressed = False
    if overwrite or not (
        os.path.exists(output_filepath) or os.path.exists(decompressed_filepath)
    ):
        # downloaded = cddis_download(filepath, output_filepath)
        url_path = "https://cddis.nasa.gov/archive/" + filepath
        downloaded = http_download(url_path, output_filepath)
    if os.path.exists(output_filepath):
        decompressed = decompress(output_filepath, decompressed_filepath)

    if os.path.exists(decompressed_filepath):
        return decompressed_filepath
    elif os.path.exists(output_filepath):
        return output_filepath
    else:
        return None


def download_and_parse_sp3_data(
    start_time_gpst: GPSTime | float,
    end_time_gpst: GPSTime | float,
    data_dir: str,
    verbose: bool = False,
    overwrite: bool = False,
    remove_duplicates: bool = False,
) -> Optional[SP3Arrays]:
    """
    `start_time_gpst` -- earliest time at which SP3 data is required
    `end_time_gpst` -- latest time at which SP3 data is required
    `data_dir` -- root directory from which SP3 data is stored
    `sat_ids` -- the satellite IDs corresponding to the satellites for which splines should
        be computed.  If `None` (default), computes for all satellites in the SP3 files
    """
    if isinstance(start_time_gpst, GPSTime):
        pass
    elif isinstance(start_time_gpst, float):
        start_time_gpst = GPSTime.from_float_seconds(start_time_gpst)
    else:
        raise Exception("`start_time_gpst` must be GPSTime or float.")
    if isinstance(end_time_gpst, GPSTime):
        pass
    elif isinstance(end_time_gpst, float):
        end_time_gpst = GPSTime.from_float_seconds(end_time_gpst)
    else:
        raise Exception("`end_time_gpst` must be GPSTime or float.")

    day_start = start_time_gpst.to_datetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end = end_time_gpst.to_datetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(hours=24)

    sp3_filepaths = []
    day = day_start
    while day < day_end:

        if verbose:
            logging.info("\rFetching SP3 data... {0}".format(day.strftime("%Y%m%d")))

        try:
            sp3_filepath = download_and_decompress_sp3_file(day, data_dir, overwrite)
            sp3_filepaths.append(sp3_filepath)
        except Exception as e:
            logging.info(
                "Failed to download/decompress SP3 filepath for day: {0}".format(
                    day.strftime("%Y%m%d")
                )
            )
            raise e

        day += timedelta(days=1)

    if verbose:
        logging.info("... done.")
        logging.info("Parsing SP3 data...")

    sp3_dataset = Dataset()
    sp3_dataset.load_files(sp3_filepaths, strict=False)

    if verbose:
        logging.info(" done.")

    return sp3_dataset.get_sp3_arrays(merge_duplicates=True)


def compute_splines_from_sp3_dict(epochs: np.ndarray, records: Dict[str, np.ndarray], order: int=5) -> Dict[str, List[scipy.interpolate.UnivariateSpline]]:
    """
    For each SP3 record, computes a spline
    """
    splines = {}
    for sat_id, rec in records.items():
        # Compute univariate spline for each record
        splines[sat_id] = [
            scipy.interpolate.UnivariateSpline(epochs, rec[:, i], k=order)
            for i in range(rec.shape[1])
        ]
    return splines


def compute_values_from_sp3_splines(
    times: np.ndarray,
    sp3_splines: Dict[str, List[scipy.interpolate.UnivariateSpline]],
) -> Dict[str, np.ndarray]:
    """
    Given the splines computed for a set of SP3 records, evaluates each spline at `times`
    """
    interpolated = {}
    for sat_id, splines in sp3_splines.items():
        interpolated[sat_id] = np.stack([spline(times) for spline in splines]).T
    return interpolated


def compute_derivatives_from_sp3_splines(
    times: np.ndarray, sp3_splines: Dict[str, List[scipy.interpolate.UnivariateSpline]]
) -> Dict[str, np.ndarray]:
    """
    Given the splines computed for a set of SP3 records, obtains the derivative spline and then evaluates it at `times`
    """
    interpolated = {}
    for sat_id, splines in sp3_splines.items():
        interpolated[sat_id] = np.stack(
            [spline.derivative()(times) for spline in splines]
        ).T
    return interpolated


def compute_satellite_ecf_positions_from_cddis_sp3(
    times: np.ndarray,  # GTIME_DTYPE or float/int array  # TODO: this should actually just take GTIME_DTYPE array
    data_dir: str,
    order: int = 5,
    sat_ids: Optional[List[str]] = None,
    use_splines: bool = True,  # otherwise use lagrange
    overwrite: bool = False,
) -> Dict[str, np.ndarray]:
    """
    `times` -- times (GPS seconds) for which to compute satellite positions
    `data_dir` -- the root directory for storing SP3 data
    """
    start_time_gpst, end_time_gpst = times[[0, -1]]
    if times.dtype == GTIME_DTYPE:
        start_time = GPSTime(*start_time_gpst)
        end_time = GPSTime(*end_time_gpst)
    elif times.dtype in [np.float64, np.int32, np.int64]:
        start_time = GPSTime.from_float_seconds(start_time_gpst)
        end_time = GPSTime.from_float_seconds(end_time_gpst)
    else:
        raise Exception("`times` must be of type GTIME_DTYPE or float/int array.")
    sp3_arrays = download_and_parse_sp3_data(start_time, end_time, data_dir, overwrite=overwrite)
    if sp3_arrays is None:
        raise ValueError("No SP3 data available")
    
    records = sp3_arrays.position
    if records is None:
        raise ValueError("No SP3 position data available")
    
    if sat_ids is None:
        sat_ids = records.keys()
    else:
        records = {k: records[k] for k in sat_ids if k in records.keys()}

    # `epochs` contains float GPST values
    initial_time = sp3_arrays.epochs[0]
    epoch_deltas = sp3_arrays.epochs - initial_time
    if times.dtype == GTIME_DTYPE:
        time_deltas = times["whole_seconds"] - initial_time
        time_deltas += times["frac_seconds"]
    else:
        time_deltas = times - initial_time
    
    if use_splines:
        sp3_splines = compute_splines_from_sp3_dict(epoch_deltas, records, order)
        sp3_positions = compute_values_from_sp3_splines(time_deltas, sp3_splines)
    else:
        sp3_positions = {
            sat_id: compute_array_lagrange_interpolation(
                time_deltas, epoch_deltas, records[sat_id], order
            ) for sat_id in records.keys()
        }

    interpolated = {}
    for sat_id in sat_ids:
        if sat_id not in sp3_positions:
            continue
        ecf_pos = sp3_positions[sat_id]
        if np.all(np.isnan(ecf_pos)):
            continue
        interpolated[sat_id] = ecf_pos

    return interpolated
