"""
Author Brian Breitsch
Date: 2025-01-02
"""

import re
from datetime import datetime
from dataclasses import dataclass
from .rinex2 import parse_RINEX2_header
from ..time.gpst import GPS_EPOCH
from typing import Any, Dict, List, Tuple


@dataclass
class Header:

    # fields marked as Optional are listed as optional in the RINEX specification
    rinex_version: str
    file_type: str
    system_code: str
    program_name: str
    run_by: str
    ionospheric_corr: List[str]
    leap_seconds: int
    
    comment_lines: List[str]

@dataclass
class RINEX_LNAVEphemeris:
    """
    Parameter definitions:
        epoch - Python `datetime` object that is the epoch corresponding to the
            ephemeris -- this is also the "Time of Clock" (`toc`)
        af0, af1, af2 - satellite clock error correction parameters
        toe - time of ephemeris
        a - semi-major axis (m); usually given as SQRT
        e - eccentricity
        i0 - inclination at reference time (rad)
        Omega0 - right ascension at week (rad)
        OmegaDot - rate of right ascension (rad/s)
        omega - argument of perigee
        M0 - mean anomaly of reference time (rad)
        week - GPS week number
        deln - mean motion difference (rad/s)
        iDot - rate of inclination angle (rad/s)
        Cus - argument of latitude (amplitude of cosine, radians)
        Crs - orbit radius (amplitude of sine, meters)
        Cis - inclination (amplitude of sine, meters)
        Cuc - argument of latitude (amplitude of cosine, radians)
        Crc - orbit radius (amplitude of cosine, meters)
        Cic - inclination (amplitude of cosine, meters)

        # Other flags from ephemeris
        l2_codes -
        l2_data_flag -
        sv_accuracy_flag -
        sv_health_flag -
        tgd -
        iodc -
        transmit_time -
        fit_interval -
    """
    epoch: datetime
    af0: float
    af1: float
    af2: float
    week_num: int
    toe: int
    a: float
    e: float
    i0: float
    OmegaDot: float
    Omega0: float
    omega: float
    M0: float
    deln: float
    iDot: float
    Cus: float
    Crs: float
    Cis: float
    Cuc: float
    Crc: float
    Cic: float
    l2_codes: int
    l2_data_flag: int
    sv_accuracy_flag: int
    sv_health_flag: int
    tgd: float
    iodc: int
    transmit_time: float
    fit_interval: float

    def __post_init__(self):
        # epoch in RINEX nav record corresponds to TOC
        # But it should be interpreted as GPS time -- i.e. no leap seconds
        self.epoch_gpst_seconds = (
            self.epoch - GPS_EPOCH.replace(tzinfo=None)
        ).total_seconds()
        self.toc = self.epoch_gpst_seconds - self.week_num * 604800


def parse_RINEX_LNAV_data(
    lines: List[str], century: int = 2000
) -> Dict[int, List[RINEX_LNAVEphemeris]]:
    """
    Given filepath to RINEX Navigation file, parses navigation into ephemeris.
    Returns dictionary {prn: [{<eph>}]} of ephemeris dictionaris

    Output
    ------
    Dictionary of format:
        {<prn>: [RINEX_LNAVEphemeris]}
    """
    epoch_pattern = (
        "(\s?\d+)\s(\s?\d+)\s(\s?\d+)\s(\s?\d+)\s(\s?\d+)\s(\s?\d+)\s(\s?\d+\.\d)"
    )
    number_pattern = "\n?\s*([+-]?\d+\.\d{12}D[+-]?\d{2})"
    pattern = epoch_pattern + 29 * number_pattern
    data: Dict[int, List[RINEX_LNAVEphemeris]] = {}
    matches = re.findall(pattern, "\n".join(lines))
    for m in matches:
        prn, yy, month, day, hour, minute = (int(i) for i in m[:6])
        (
            second,
            af0,
            af1,
            af2,
            iode,
            Crs,
            deln,
            M0,
            Cuc,
            e,
            Cus,
            sqrt_a,
            toe,
            Cic,
            Omega0,
            Cis,
            i0,
            Crc,
            omega,
            OmegaDot,
            iDot,
            l2_codes,
            week_num,
            l2_data_flag,
            sv_accuracy_flag,
            sv_health_flag,
            tgd,
            iodc,
            transmit_time,
            fit_interval,
        ) = (float(s.replace("D", "E")) for s in m[6:36])

        year = century + yy
        epoch = datetime(
            year, month, day, hour, minute, int(second), int(1e6 * (second % 1))
        )

        eph = RINEX_LNAVEphemeris(
            epoch,
            af0,
            af1,
            af2,
            week_num,
            toe,
            sqrt_a**2,
            e,
            i0,
            OmegaDot,
            Omega0,
            omega,
            M0,
            deln,
            iDot,
            Cus,
            Crs,
            Cis,
            Cuc,
            Crc,
            Cic,
            l2_codes,
            l2_data_flag,
            sv_accuracy_flag,
            sv_health_flag,
            tgd,
            iodc,
            transmit_time,
            fit_interval,
        )
        if prn not in data.keys():
            data[prn] = []
        data[prn].append(eph)
    return data


def parse_RINEX_LNAV_file(filepath: str) -> Tuple[Dict[str, Any], Dict[int, List[RINEX_LNAVEphemeris]]]:
    """Given the filepath to a RINEX navigation message file, parses and returns header
    and navigation ephemeris data.

    Input
    -----
    `filepath` -- filepath to RINEX navigation file

    Output
    ------
    `header, nav_data` where `header` is a dictionary containing the parsed header information
        and `nav_data` is a dictionary containing the navigation data in the format:
        {<prn>: [<dict>, ... ]})}

    where each dictionary corresponds to a different ephemeris set.  See documentation in
    `parse_RINEX_LNAV_data` for information on the contents of each namespace.

    Note: `epoch` is a `datetime` object
    """
    with open(filepath, "r") as f:
        lines = list(f.readlines())
    for i, line in enumerate(lines):
        if line.find("END OF HEADER") >= 0:
            break
    header_lines = lines[: i + 1]
    nav_lines = lines[i + 1 :]
    header = parse_RINEX2_header(header_lines)
    nav_data = parse_RINEX_LNAV_data(nav_lines)
    return header, nav_data
