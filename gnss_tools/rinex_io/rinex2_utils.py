"""
Author Brian Breitsch
Date: 2025-01-02
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# RINEX 2.10 - 2.11
CONSTELLATION_IDS = {
    "G": "GPS",
    "R": "GLONASS",
    "E": "Galileo",
    "S": "SBAS",
}
BAND_FREQUENCIES = {
    "L1": 1575.42,
    "L2": 1227.60,
    "L5": 1176.45,
    "G1": lambda k: 1602 + k * (9 / 16),
    "G2": lambda k: 1246 + k * 716,
    "E1": 1575.42,
    "E5a": 1176.45,
    "E5b": 1207.14,
    "E5": 1191.795,
    "E6": 1278.75,
}
# RINEX version 2 does not distinguish tracking modes (added in RINEX V3)
# As such, for GPS, simple signal names L1, L2, L5 are used
OBSERVATION_DATATYPES = {
    "GPS": {
        "C1": {"band": "L1", "name": "pseudorange"},
        "P1": {"band": "L1", "name": "pseudorange"},
        "L1": {"band": "L1", "name": "carrier"},
        "D1": {"band": "L1", "name": "doppler"},
        "S1": {"band": "L1", "name": "snr"},
        "C2": {"band": "L2", "name": "pseudorange"},
        "P2": {"band": "L2", "name": "pseudorange"},
        "L2": {"band": "L2", "name": "carrier"},
        "D2": {"band": "L2", "name": "doppler"},
        "S2": {"band": "L2", "name": "snr"},
        "C5": {"band": "L5", "name": "pseudorange"},
        "P5": {"band": "L5", "name": "pseudorange"},
        "L5": {"band": "L5", "name": "carrier"},
        "D5": {"band": "L5", "name": "doppler"},
        "S5": {"band": "L5", "name": "snr"},
    },
    "GLONASS": {
        "C1": {"band": "G1", "name": "pseudorange"},
        "P1": {"band": "G1", "name": "pseudorange"},
        "L1": {"band": "G1", "name": "carrier"},
        "D1": {"band": "G1", "name": "doppler"},
        "S1": {"band": "G1", "name": "snr"},
        "C2": {"band": "G2", "name": "pseudorange"},
        "P2": {"band": "G2", "name": "pseudorange"},
        "L2": {"band": "G2", "name": "carrier"},
        "D2": {"band": "G2", "name": "doppler"},
        "S2": {"band": "G2", "name": "snr"},
    },
    "Galileo": {
        "C1": {"band": "E1", "name": "pseudorange"},
        "L1": {"band": "E1", "name": "carrier"},
        "D1": {"band": "E1", "name": "doppler"},
        "S1": {"band": "E1", "name": "snr"},
        "C5": {"band": "E5a", "name": "pseudorange"},
        "L5": {"band": "E5a", "name": "carrier"},
        "D5": {"band": "E5a", "name": "doppler"},
        "S5": {"band": "E5a", "name": "snr"},
        "C7": {"band": "E5b", "name": "pseudorange"},
        "L7": {"band": "E5b", "name": "carrier"},
        "D7": {"band": "E5b", "name": "doppler"},
        "S7": {"band": "E5b", "name": "snr"},
        "C8": {"band": "E5ab", "name": "pseudorange"},
        "L8": {"band": "E5ab", "name": "carrier"},
        "D8": {"band": "E5ab", "name": "doppler"},
        "S8": {"band": "E5ab", "name": "snr"},
        "C6": {"band": "E6", "name": "pseudorange"},
        "L6": {"band": "E6", "name": "carrier"},
        "D6": {"band": "E6", "name": "doppler"},
        "S6": {"band": "E6", "name": "snr"},
    },
    "SBAS": {
        "C1": {"band": "L1", "name": "pseudorange"},
        "L1": {"band": "L1", "name": "carrier"},
        "D1": {"band": "L1", "name": "doppler"},
        "S1": {"band": "L1", "name": "snr"},
        "C5": {"band": "L5", "name": "pseudorange"},
        "L5": {"band": "L5", "name": "carrier"},
        "D5": {"band": "L5", "name": "doppler"},
        "S5": {"band": "L5", "name": "snr"},
    },
}


def parse_RINEX2_header(lines: List[str]) -> Dict[str, Any]:
    """
    ------------------------------------------------------------
    Given list of lines corresponding to the header of a RINEX 3
    file, parses the header of the file and returns a dictionary
    containing the header information.

    Input
    -----
    `lines` -- lines corresponding to RINEX header

    Output
    ------
    dictionary containing RINEX header information
    """
    header = {}
    header["comments"] = []
    lines = iter(lines)
    try:
        while True:
            line = next(lines)
            header_label = line[60:].strip()
            if header_label == "RINEX VERSION / TYPE":
                header["version"] = line[:20].strip()
                header["file_type"] = line[20:40].strip()
                header["system_type"] = line[40:60].strip()
            elif header_label == "PGM / RUN BY / DATE":
                header["program"] = line[:20].strip()
                header["run_by"] = line[20:40].strip()
                header["date"] = line[40:60].strip()
            elif header_label == "MARKER NAME":
                header["marker_name"] = line[:60].strip()
            elif header_label == "MARKER NUMBER":
                header["marker_number"] = line[0:60].strip()
            elif header_label == "MARKER TYPE":
                header["marker_type"] = line[0:20].strip()
            elif header_label == "OBSERVER / AGENCY":
                header["observer"] = line[:20].strip()
                header["agency"] = line[20:60].strip()
            elif header_label == "REC # / TYPE / VERS":
                header["receiver_number"] = line[:20].strip()
                header["receiver_type"] = line[20:40].strip()
                header["receiver_version"] = line[40:60].strip()
            elif header_label == "ANT # / TYPE":
                header["antenna_number"] = line[:20].strip()
                header["antenna_type"] = line[20:60].strip()
            elif header_label == "APPROX POSITION XYZ":
                header["approximate_position_xyz"] = line[:60].strip()
            elif header_label == "ANTENNA: DELTA H/E/N":
                header["delta_hen"] = line[:60].strip()
            elif header_label == "APPROX POSITION XYZ":
                header["approximate_position_xyz"] = line[:60].strip()
            elif header_label == "WAVELENGTH FACT L1/2":
                header["wavelength_fact_l12"] = line[:60].strip()
            elif header_label == "APPROX POSITION XYZ":
                header["approximate_position_xyz"] = line[:60].strip()
            elif header_label == "TIME OF FIRST OBS":
                header["time_of_first_obs"] = line[:60].strip()
            elif header_label == "# / TYPES OF OBSERV":
                n_obs_str = line[:10].strip()
                if n_obs_str:
                    header["n_obs"] = int(n_obs_str)
                    header["obs_types"] = []
                header["obs_types"] += line[10:60].split()
            elif header_label == "COMMENT":
                header["comments"].append(line[:60])
    except StopIteration:
        pass
    return header


def parse_RINEX2_obs_data(
    lines: List[str],
    observations: List[str],
    century: int = 2000,
    exception_behavior: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[np.datetime64], List[int]]:
    """
    ------------------------------------------------------------
    Given `lines` corresponding to the RINEX observation file
    data (non-header) lines, and a list of the types of
    observations recorded at each epoch, produces a dictionary
    containing the observation time and values for each
    satellite.

    Input
    -----
    lines -- data lines from RINEX observation file
    observations -- list of the observations reported at
        each epoch
    exception_behavior = None -- (optional) use "skip" to try and skip
        exception-inducing lines

    Output
    ------
    data: dict -- dictionary of format:
        {<sat_id>: {"index": [<int...>], <obs_id>: [<values...>]}}
    time: List[np.datetime64] -- list of times corresponding to epochs
    """
    data: Dict[str, Any] = (
        {}
    )  # <sat_id>: {"index": [<int...>], <obs_id>: [<values...>]}
    lines = iter(lines)
    epoch_index = 0
    time = []
    epoch_flags = []
    comments = []
    while True:
        try:
            # at each epoch, the two-digit year, month, day, hour, minute, and seconds
            # of the measurement epoch are specified, along with the number and ids of
            # the satellites whose measurements are given
            line = next(lines)
            yy = int(line[:4].strip())
            year = century + yy
            month = int(line[4:7])
            day = int(line[7:10])
            hour = int(line[10:13])
            minute = int(line[13:16])
            seconds = float(line[16:25])
            microseconds = int(1e6 * (seconds % 1))
            seconds = int(seconds)
            dt = np.datetime64(
                datetime(year, month, day, hour, minute, seconds, microseconds)
            )
            time.append(dt)
            epoch_flag = int(line[25:28])
            epoch_flags.append(epoch_flag)
            num_sats = int(line[29:32])
            # there is space for (80 - 32) / 3 = 16 satellite ids
            # if there are more than 16, then they continue on the next line
            # a general approach is to consume lines until we have determined all sat IDs
            # TODO: techinically spec says if there are more than 12, continue on next line
            # should change this to 12
            sat_ids = []
            # line = line[32:].strip()
            line = line[32:68].strip()
            while len(sat_ids) < num_sats:
                sat_ids.append(line[:3].replace(" ", "0"))
                line = line[3:]
                if line == "" and len(sat_ids) < num_sats:
                    line = next(lines)
                    assert line[:32].strip() == ""
                    line = line.strip()
                    assert (
                        len(line) % 3 == 0
                    )  # sanity check -- each sat ID takes 3 chars
            for sat_id in sat_ids:
                # create new entry if `sat_id` is new
                if sat_id not in data.keys():
                    data[sat_id] = {"index": []}
                # append time/index first, then append obs values
                data[sat_id]["index"].append(epoch_index)
                # each line of observation values contains up to 5 entries
                # each entry is of width 16, starting at index 0
                num_lines_per_sat = len(observations) // 5 + int(
                    len(observations) % 5 > 0
                )
                line = ""
                for i in range(num_lines_per_sat):
                    line += next(lines).replace("\n", "").ljust(80)
                for i, obs_id in enumerate(observations):
                    val_str = line[16 * i : 16 * (i + 1)].strip()
                    if val_str == "":
                        val_str = "nan"
                    val_str = val_str.split()
                    if len(val_str) == 1:
                        val_str = val_str[0]
                    elif len(val_str) == 2:
                        val_str, sig_flag = val_str
                    else:
                        val_str = "nan"
                        # assert(False)  # error
                    try:
                        val = float(val_str)
                    except Exception:
                        val = np.nan
                    if obs_id not in data[sat_id].keys():
                        data[sat_id][obs_id] = []
                    data[sat_id][obs_id].append(val)
            epoch_index += 1
        except StopIteration:
            break
        except Exception as e:
            # TODO:
            # need to be able to handle mid-file comments
            if line.rstrip().endswith("COMMENT"):
                comments.append((epoch_index, line))
            elif exception_behavior == "skip":
                continue
            else:
                print(line)
                raise e
    return data, time, epoch_flags


def transform_values_from_RINEX2_obs(
    rinex_data: Dict[str, Any], frequency_numbers: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    ------------------------------------------------------------
    Transforms output from `parse_RINEX3_obs_data` to more
    useful format.

    Input:
    -------
    `rinex_data` -- Python dictionary with format:
        {
            <sat_id>: {
                    "index": [<int>,...],
                    <obs_id>: [<values...>]
                }
        }

    Output:
    -------
    `data` -- dictionary in format:
        {<sat_id>: {
                "index": ndarray,
                <band_id>: {
                    <obs_name>: ndarray
                }
            }
        }
    """
    data = {}
    for sat_id, rnx_sat in rinex_data.items():
        if sat_id not in data.keys():
            data[sat_id] = {}
        constellation = CONSTELLATION_IDS[sat_id[0]]
        obs_datatypes = OBSERVATION_DATATYPES[constellation]
        for obs_id, mapping in obs_datatypes.items():
            if obs_id in rnx_sat.keys():
                band_id = mapping["band"]
                obs_name = mapping["name"]
                if band_id not in data[sat_id].keys():
                    frequency = BAND_FREQUENCIES[band_id]
                    if constellation == "GLONASS" and callable(frequency):
                        if (
                            frequency_numbers is not None
                            and sat_id in frequency_numbers.keys()
                        ):
                            frequency = frequency(frequency_numbers[sat_id])
                        else:
                            frequency = np.nan
                    data[sat_id][band_id] = {"frequency": frequency * 1e6}
                values = np.array(rnx_sat[obs_id])
                if not np.all(np.isnan(values)):
                    data[sat_id][band_id][obs_name] = values
        if "index" in rnx_sat.keys():
            data[sat_id]["index"] = np.array(rnx_sat["index"], dtype=int)
    return data


def parse_RINEX2_obs_file(
    filepath: str, century: int = 2000, exception_behavior: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ------------------------------------------------------------
    Given the filepath to a RINEX observation file, parses and
    returns header and observation data.

    Input
    -----
    filepath: str -- filepath to RINEX observation file
    exception_behavior = None -- passed to `parse_RINEX2_obs_data`;
        use "skip" to try and skip error-inducing lines

    Output
    ------
    header, observations: Tuple[dict, dict] -- where `header` is a dictionary
        containing the parsed header information and `observations`
        is a dictionary containing the observation data in the
        format:

        {
            "time": ndarray,
            "satellites": {
                <sat_id>: {
                    "index": ndarray,
                    <obs_id>: ndarray
                }
            }
        }

    Note: `time` in `observations` is in GPST seconds
    **Warning**: this function cannot currently handle splicing
        / comments in the middle of a RINEX file.
    """
    with open(filepath, "r") as f:
        lines = list(f.readlines())
    for i, line in enumerate(lines):
        if line.find("END OF HEADER") >= 0:
            break
    header_lines = lines[: i + 1]
    obs_lines = lines[i + 1 :]
    header = parse_RINEX2_header(header_lines)
    if "obs_types" not in header.keys():
        raise Exception(
            "RINEX header must contain `# / TYPES OF OBS.` and `header` dict from `parse_parse_RINEX2_header` must contain corresponding list `obs_types`"
        )
    obs_data, time, epoch_flags = parse_RINEX2_obs_data(
        obs_lines, header["obs_types"], century, exception_behavior
    )
    obs_data = transform_values_from_RINEX2_obs(obs_data)
    gps_epoch = np.datetime64(datetime(1980, 1, 6))
    time = (np.array(time) - gps_epoch).astype(float) / 1e6  # dt64 is in microseconds
    observations = {"time": time, "satellites": obs_data, "flags": epoch_flags}
    return header, observations
