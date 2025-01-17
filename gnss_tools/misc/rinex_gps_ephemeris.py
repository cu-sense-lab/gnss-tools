"""
Miscellaneous utilities for working with RINEX GPS Navigation files

Author: Brian Breitsch
Date: 2025-01-02
"""

from ..misc.gps_ephemeris import (
    compute_gps_orbital_params_from_ephemeris,
    compute_ecf_position_from_orbital_parameters,
    compute_ecf_velocity_from_orbital_parameters,
    compute_gps_transmitter_clock_bias_with_relativity_correction,
)
from ..misc.data_utils import format_filepath, cddis_download, decompress
from ..rinex.rinex_nav import parse_RINEX_LNAV_file, RINEX_LNAVEphemeris
from ..time.gpst import GPSTime

import os.path
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np


def remove_bad_ephemerides(
    ephemerides: List[RINEX_LNAVEphemeris], tol: float = 1
) -> List[RINEX_LNAVEphemeris]:
    """If any ephemerides occur 16 seconds before another ephemeris epoch, removes other ephemeris epoch"""
    ephemerides = sorted(ephemerides, key=lambda eph: eph.epoch)
    bad_indices = []
    for i in range(len(ephemerides) - 1):
        eph = ephemerides[i]
        next_eph = ephemerides[i + 1]
        if abs((next_eph.epoch - eph.epoch).total_seconds()) < tol:
            bad_indices.append(i + 1)
    return [eph for i, eph in enumerate(ephemerides) if i not in bad_indices]


def get_best_ephemeris_indices(
    ephemerides: List[RINEX_LNAVEphemeris], times: np.ndarray
) -> List[int]:
    """
     ----------------------------------------------------------------------------
    Given a list of ephemerides parsed from `parse_RINEX_LNAV_file`, finds and returns
    the indices of ephemerideswhose epoch is closest to `times`.
    """
    eph_index = np.argmin(
        np.vstack(
            [
                np.abs(GPSTime.from_datetime(eph.epoch).to_float() - times)
                for eph in ephemerides
            ]
        ),
        axis=0,
    )
    return eph_index


def get_best_ephemerides(ephemerides: List[RINEX_LNAVEphemeris], target_time: datetime):
    """
     ----------------------------------------------------------------------------
    Given a list of ephemerides parsed from `parse_RINEX_LNAV_file`, finds and
    returns the ephemeris whose epoch is closest to `target_times`.  Note,
    `target_time` should have the same type as `eph.epoch`, in this case,
    `datetime`.
    """

    def time_delta(eph: RINEX_LNAVEphemeris):
        dt_delta = eph.epoch - target_time
        return np.abs(dt_delta.total_seconds())

    eph_index = np.argmin([np.abs(time_delta(eph)) for eph in ephemerides])
    return ephemerides[eph_index]


def download_and_decompress_gps_ephemerides(
    start_time: GPSTime, end_time: GPSTime, data_dir: str, overwrite: bool = False
) -> Dict[str, List[RINEX_LNAVEphemeris]]:
    """
    Automatically downloads all (GPS) ephemerides from IGS within interval
    `min(times), max(times)`
    ----------------------------------------------------------------------------
    Input:
    `times` -- list of times (GPST seconds) for the duration of which to obtain
        ephemerides
    Output:
        dictionary containing the ephemerides for each satellite, with
        corresponding integer keys `<prn>` and values as lists of `RINEX_LNAVEphemeris` objects
    """
    path_expression = "/gnss/data/daily/{yyyy}/{ddd}/{yy}n/"
    # extension before 2020-335 is .Z; after is .gz; on is a mix
    if end_time.to_datetime() < datetime(2020, 12, 1):
        file_expression = "brdc{ddd}0.{yy}n.Z"
    else:
        file_expression = "brdc{ddd}0.{yy}n.gz"

    # TODO check if microseconds makes a difference here
    start_day = start_time.to_datetime().replace(hour=0, minute=0, second=0)
    end_day = end_time.to_datetime().replace(hour=0, minute=0, second=0) + timedelta(
        days=1
    )

    all_ephemerides: Dict[int, List[RINEX_LNAVEphemeris]] = {}
    day = start_day
    while day <= end_day + timedelta(days=1):
        try:
            url_filepath = format_filepath(
                os.path.join(path_expression, file_expression), day
            )
            local_filepath = (
                data_dir + url_filepath[1:]
            )  # cannot join with `/` at beginning
            decompressed_local_filepath = local_filepath[:-2]
            if not os.path.exists(decompressed_local_filepath) or overwrite:
                if not os.path.exists(local_filepath) or overwrite:
                    os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
                    cddis_download(url_filepath, local_filepath)
                decompress(local_filepath, decompressed_local_filepath)
            _, ephemerides = parse_RINEX_LNAV_file(decompressed_local_filepath)

            for key in ephemerides.keys():
                if key not in all_ephemerides.keys():
                    all_ephemerides[key] = []
                all_ephemerides[key] += ephemerides[key]
        except Exception as e:
            print("Error in download / decompress / parse of file:")
            print(f"    Remote: {url_filepath}")
            print(f"    Local: {local_filepath}")
            raise e
        day += timedelta(days=1)
    return all_ephemerides


def compute_gps_satellite_pvt_from_ephemeris(
    times: np.ndarray,
    data_dir: str,
    sat_ids="all",
    rollover: int = 0,
    compute_velocity: bool = False,
    compute_clock_bias: bool = False,
    overwrite: bool = False,
    print_progress: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Automatically compute GPS satellite position (and optionally, velocity and
    clock bias) for `times` (in GPS seconds).
    ----------------------------------------------------------------------------
    Input:
    `times` -- array of times (GPST seconds) for which to compute satellite positions
    `sat_ids` -- list of GPS satellite IDs for which to compute positions.  Sat.
        IDs must be of format `'G<PRN>'` where <PRN> is the 2-digit GPS PRN
        number.  If `sat_ids` is 'all', then computes positions for all GPS
        satellites
    Output:
        dictionary in the form:
        {
            'position': {
                '<sat_id>': <(N, 3) array>,
                ...
            },
            'velocity': {  # optional entry
                '<sat_id>': <(N, 3) array>,
                ...
            },
            'clock_bias': {  # optional entry
                '<sat_id>': <(N,) array>
            }
        }
    """
    if sat_ids == "all":
        sat_ids = ["G{0:02}".format(prn) for prn in range(1, 33)]
    result: Dict[str, Dict[str, Any]] = {"position": {}}
    if compute_velocity:
        result["velocity"] = {}
    if compute_clock_bias:
        result["clock_bias"] = {}

    all_ephemerides = download_and_decompress_gps_ephemerides(
        times, data_dir, overwrite
    )

    N = len(times)
    for i, sat_id in enumerate(sat_ids):
        if print_progress:
            print(
                "\r {0: >4} / {1: >4} {2}".format(i + 1, len(sat_ids), sat_id), end=""
            )
        if sat_id[0] != "G":
            continue  # must be GPS satellite
        prn = int(sat_id[1:])
        if prn not in all_ephemerides:
            continue
        ephemerides = all_ephemerides[prn]
        best_eph_indices = get_best_ephemeris_indices(ephemerides, times)
        # ephemerides = [ephemerides[i] for i in eph_ind]
        # params = compute_gps_orbital_params_from_ephemeris(ephemerides[i], times[i])

        result["position"][sat_id] = np.zeros((N, 3))
        if compute_velocity:
            result["velocity"][sat_id] = np.zeros((N, 3))
        if compute_clock_bias:
            result["clock_bias"][sat_id] = np.zeros((N,))

        for j in range(N):
            eph = ephemerides[best_eph_indices[j]]
            a, e, i, r, Omega, Phi, n, E = compute_gps_orbital_params_from_ephemeris(
                eph, times[j], rollover
            )
            result["position"][sat_id][j, :] = (
                compute_ecf_position_from_orbital_parameters(i, r, Omega, Phi)
            )
            if compute_velocity:
                result["velocity"][sat_id][j, :] = (
                    compute_ecf_velocity_from_orbital_parameters(a, e, i, Omega, n, E)
                )
            if compute_clock_bias:
                result["clock_bias"][sat_id] = (
                    compute_gps_transmitter_clock_bias_with_relativity_correction(
                        times[j], eph.toc, eph.af0, eph.af1, eph.af2, eph.a, eph.e, E
                    )
                )

    return result
