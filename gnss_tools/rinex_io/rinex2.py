"""
Author Brian Breitsch
Date: 2025-01-02
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import io
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# logging.warning("This module is currently broken and not usable")

__RINEX_VERSION_2_11_FORMAT_DEFINITIONS__ = """
APPENDIX: RINEX VERSION 2.11 FORMAT DEFINITIONS AND EXAMPLES
 +----------------------------------------------------------------------------+
 |                                   TABLE A1                                 |
 |          GNSS OBSERVATION DATA FILE - HEADER SECTION DESCRIPTION           |
 +--------------------+------------------------------------------+------------+
 |    HEADER LABEL    |               DESCRIPTION                |   FORMAT   |
 |  (Columns 61-80)   |                                          |            |
 +--------------------+------------------------------------------+------------+
 |RINEX VERSION / TYPE| - Format version (2.11)                  | F9.2,11X,  |
 |                    | - File type ('O' for Observation Data)   |   A1,19X,  |
 |                    | - Satellite System: blank or 'G': GPS    |   A1,19X   |
 |                    |                     'R': GLONASS         |            |
 |                    |                     'S': Geostationary   |            |
 |                    |                          signal payload  |            |
 |                    |                     'E': Galileo         |            |
 |                    |                     'M': Mixed           |            |
 +--------------------+------------------------------------------+------------+
 |PGM / RUN BY / DATE | - Name of program creating current file  |     A20,   |
 |                    | - Name of agency  creating current file  |     A20,   |
 |                    | - Date of file creation                  |     A20    |
 +--------------------+------------------------------------------+------------+
*|COMMENT             | Comment line(s)                          |     A60    |*
 +--------------------+------------------------------------------+------------+
 |MARKER NAME         | Name of antenna marker                   |     A60    |
 +--------------------+------------------------------------------+------------+
*|MARKER NUMBER       | Number of antenna marker                 |     A20    |*
 +--------------------+------------------------------------------+------------+
 |OBSERVER / AGENCY   | Name of observer / agency                |   A20,A40  |
 +--------------------+------------------------------------------+------------+
 |REC # / TYPE / VERS | Receiver number, type, and version       |    3A20    |
 |                    | (Version: e.g. Internal Software Version)|            |
 +--------------------+------------------------------------------+------------+
 |ANT # / TYPE        | Antenna number and type                  |    2A20    |
 +--------------------+------------------------------------------+------------+
 |APPROX POSITION XYZ | Approximate marker position (WGS84)      |   3F14.4   |
 +--------------------+------------------------------------------+------------+
 |ANTENNA: DELTA H/E/N| - Antenna height: Height of bottom       |   3F14.4   |
 |                    |   surface of antenna above marker        |            |
 |                    | - Eccentricities of antenna center       |            |
 |                    |   relative to marker to the east         |            |
 |                    |   and north (all units in meters)        |            |
 +--------------------+------------------------------------------+------------+
*|WAVELENGTH FACT L1/2| - Default wavelength factors for         |            |*
 |                    |   L1 and L2 (GPS only)                   |    2I6,    |
 |                    |   1:  Full cycle ambiguities             |            |
 |                    |   2:  Half cycle ambiguities (squaring)  |            |
 |                    |   0 (in L2): Single frequency instrument |            |
 |                    | - zero or blank                          |     I6     |
 |                    |                                          |            |
 |                    | The wavelength factor record is optional |            |
 |                    | for GPS and obsolete for other systems.  |            |
 |                    | Wavelength factors default to 1.         |            |
 |                    | If the record exists it must precede any |            |
 |                    | satellite-specific records (see below).  |            |
 +--------------------+------------------------------------------+------------+
*|WAVELENGTH FACT L1/2| - Wavelength factors for L1 and L2 (GPS) |    2I6,    |*
 |                    |   1:  Full cycle ambiguities             |            |
 |                    |   2:  Half cycle ambiguities (squaring)  |            |
 |                    |   0 (in L2): Single frequency instrument |            |
 |                    | - Number of satellites to follow in list |     I6,    |
 |                    |   for which these factors are valid.     |            |
 |                    | - List of PRNs (satellite numbers with   | 7(3X,A1,I2)|
 |                    |   system identifier)                     |            |
 |                    |                                          |            |
 |                    | These optional satellite specific lines  |            |
 |                    | may follow, if they identify a state     |            |
 |                    | different from the default values.       |            |
 |                    |                                          |            |
 |                    | Repeat record if necessary.              |            |
 +--------------------+------------------------------------------+------------+
 |# / TYPES OF OBSERV | - Number of different observation types  |     I6,    |
 |                    |   stored in the file                     |            |
 |                    | - Observation types                      |            |
 |                    |   - Observation code                     | 9(4X,A1,   |
 |                    |   - Frequency code                       |         A1)|
 |                    |   If more than 9 observation types:      |            |
 |                    |     Use continuation line(s) (including  |6X,9(4X,2A1)|
 |                    |     the header label in cols. 61-80!)    |            |
 |                    |                                          |            |
 |                    | The following observation types are      |            |
 |                    | defined in RINEX Version 2.11:           |            |
 |                    |                                          |            |
 |                    | Observation code (use uppercase only):   |            |
 |                    |   C: Pseudorange  GPS: C/A, L2C          |            |
 |                    |                   Glonass: C/A           |            |
 |                    |                   Galileo: All           |            |
 |                    |   P: Pseudorange  GPS and Glonass: P code|            |
 |                    |   L: Carrier phase                       |            |
 |                    |   D: Doppler frequency                   |            |
 |                    |   S: Raw signal strengths or SNR values  |            |
 |                    |      as given by the receiver for the    |            |
 |                    |      respective phase observations       |            |
 |                    |                                          |            |
 |                    | Frequency code                           |            |
 |                    |      GPS    Glonass   Galileo    SBAS    |            |
 |                    |   1:  L1       G1     E2-L1-E1    L1     |            |
 |                    |   2:  L2       G2        --       --     |            |
 |                    |   5:  L5       --        E5a      L5     |            |
 |                    |   6:  --       --        E6       --     |            |
 |                    |   7:  --       --        E5b      --     |            |
 |                    |   8:  --       --       E5a+b     --     |            |
 |                    |                                          |            |
 |                    | Observations collected under Antispoofing|            |
 |                    | are converted to "L2" or "P2" and flagged|            |
 |                    | with bit 2 of loss of lock indicator     |            |
 |                    | (see Table A2).                          |            |
 |                    |                                          |            |
 |                    | Units : Phase       : full cycles        |            |
 |                    |         Pseudorange : meters             |            |
 |                    |         Doppler     : Hz                 |            |
 |                    |         SNR etc     : receiver-dependent |            |
 |                    |                                          |            |
 |                    | The sequence of the types in this record |            |
 |                    | has to correspond to the sequence of the |            |
 |                    | observations in the observation records  |            |
 +--------------------+------------------------------------------+------------+
*|INTERVAL            | Observation interval in seconds          |   F10.3    |*
 +--------------------+------------------------------------------+------------+
 |TIME OF FIRST OBS   | - Time of first observation record       | 5I6,F13.7, |
 |                    |   (4-digit-year, month,day,hour,min,sec) |            |
 |                    | - Time system: GPS (=GPS time system)    |   5X,A3    |
 |                    |                GLO (=UTC time system)    |            |
 |                    |                GAL (=Galileo System Time)|            |
 |                    |   Compulsory in mixed GPS/GLONASS files  |            |
 |                    |   Defaults: GPS for pure GPS files       |            |
 |                    |             GLO for pure GLONASS files   |            |
 |                    |             GAL for pure Galileo files   |            |
 +--------------------+------------------------------------------+------------+
*|TIME OF LAST OBS    | - Time of last  observation record       | 5I6,F13.7, |*
 |                    |   (4-digit-year, month,day,hour,min,sec) |            |
 |                    | - Time system: Same value as in          |   5X,A3    |
 |                    |                TIME OF FIRST OBS record  |            |
 +--------------------+------------------------------------------+------------+
*|RCV CLOCK OFFS APPL | Epoch, code, and phase are corrected by  |     I6     |*
 |                    | applying the realtime-derived receiver   |            |
 |                    | clock offset: 1=yes, 0=no; default: 0=no |            |
 |                    | Record required if clock offsets are     |            |
 |                    | reported in the EPOCH/SAT records        |            |
 +--------------------+------------------------------------------+------------+
*|LEAP SECONDS        | Number of leap seconds since 6-Jan-1980  |     I6     |*
 |                    | Recommended for mixed files              |            |
 +--------------------+------------------------------------------+------------+
*|# OF SATELLITES     | Number of satellites, for which          |     I6     |*
 |                    | observations are stored in the file      |            |
 +--------------------+------------------------------------------+------------+
*|PRN / # OF OBS      | PRN (sat.number), number of observations |3X,A1,I2,9I6|*
 |                    | for each observation type indicated      |            |
 |                    | in the "# / TYPES OF OBSERV" - record.   |            |
 |                    |                                          |            |
 |                    |   If more than 9 observation types:      |            |
 |                    |   Use continuation line(s) including     |   6X,9I6   |
 |                    |   the header label in cols. 61-80!       |            |
 |                    |                                          |            |
 |                    | This record is (these records are)       |            |
 |                    | repeated for each satellite present in   |            |
 |                    | the data file                            |            |
 +--------------------+------------------------------------------+------------+
 |END OF HEADER       | Last record in the header section.       |    60X     |
 +--------------------+------------------------------------------+------------+

  +----------------------------------------------------------------------------+
 |                                   TABLE A2                                 |
 |            GNSS OBSERVATION DATA FILE - DATA RECORD DESCRIPTION            |
 +-------------+-------------------------------------------------+------------+
 | OBS. RECORD | DESCRIPTION                                     |   FORMAT   |
 +-------------+-------------------------------------------------+------------+
 | EPOCH/SAT   | - Epoch :                                       |            |
 |     or      |   - year (2 digits, padded with 0 if necessary) |  1X,I2.2,  |
 | EVENT FLAG  |   - month,day,hour,min,                         |  4(1X,I2), |
 |             |   - sec                                         |   F11.7,   |
 |             |                                                 |            |
 |             | - Epoch flag 0: OK                              |   2X,I1,   |
 |             |              1: power failure between           |            |
 |             |                 previous and current epoch      |            |
 |             |             >1: Event flag                      |            |
 |             | - Number of satellites in current epoch         |     I3,    |
 |             | - List of PRNs (sat.numbers with system         | 12(A1,I2), |
 |             |   identifier, see 5.1) in current epoch         |            |
 |             | - receiver clock offset (seconds, optional)     |   F12.9    |
 |             |                                                 |            |
 |             |   If more than 12 satellites: Use continuation  |    32X,    |
 |             |   line(s)                                       | 12(A1,I2)  |
 |             |                                                 |            |
 |             | If epoch flag  2-5:                             |            |
 |             |                                                 |            |
 |             |   - Event flag:                                 |  [2X,I1,]  |
 |             |     2: start moving antenna                     |            |
 |             |     3: new site occupation (end of kinem. data) |            |
 |             |        (at least MARKER NAME record follows)    |            |
 |             |     4: header information follows               |            |
 |             |     5: external event (epoch is significant,    |            |
 |             |        same time frame as observation time tags)|            |
 |             |                                                 |            |
 |             |   - "Number of satellites" contains number of   |    [I3]    |
 |             |     special records to follow.                  |            |
 |             |     Maximum number of records: 999              |            |
 |             |                                                 |            |
 |             |   - For events without significant epoch the    |            |
 |             |     epoch fields can be left blank              |            |
 |             |                                                 |            |
 |             | If epoch flag = 6:                              |            |
 |             |     6: cycle slip records follow to optionally  |            |
 |             |        report detected and repaired cycle slips |            |
 |             |        (same format as OBSERVATIONS records;    |            |
 |             |         slip instead of observation; LLI and    |            |
 |             |         signal strength blank or zero)          |            |
 +-------------+-------------------------------------------------+------------+
 |OBSERVATIONS | - Observation      | rep. within record for     |  m(F14.3,  |
 |             | - LLI              | each obs.type (same seq    |     I1,    |
 |             | - Signal strength  | as given in header)        |     I1)    |
 |             |                                                 |            |
 |             | If more than 5 observation types (=80 char):    |            |
 |             | continue observations in next record.           |            |
 |             |                                                 |            |
 |             | This record is (these records are) repeated for |            |
 |             | each satellite given in EPOCH/SAT - record.     |            |
 |             |                                                 |            |
 |             | Observations:                                   |            |
 |             |   Phase  : Units in whole cycles of carrier     |            |
 |             |   Code   : Units in meters                      |            |
 |             | Missing observations are written as 0.0         |            |
 |             | or blanks.                                      |            |
 |             |                                                 |            |
 |             | Phase values overflowing the fixed format F14.3 |            |
 |             | have to be clipped into the valid interval (e.g.|            |
 |             | add or subtract 10**9), set LLI indicator.      |            |
 |             |                                                 |            |
 |             | Loss of lock indicator (LLI). Range: 0-7        |            |
 |             |  0 or blank: OK or not known                    |            |
 |             |  Bit 0 set : Lost lock between previous and     |            |
 |             |              current observation: cycle slip    |            |
 |             |              possible                           |            |
 |             |  Bit 1 set : Opposite wavelength factor to the  |            |
 |             |              one defined for the satellite by a |            |
 |             |              previous WAVELENGTH FACT L1/2 line |            |
 |             |              or opposite to the default.        |            |
 |             |              Valid for the current epoch only.  |            |
 |             |  Bit 2 set : Observation under Antispoofing     |            |
 |             |              (may suffer from increased noise)  |            |
 |             |                                                 |            |
 |             |  Bits 0 and 1 for phase only.                   |            |
 |             |                                                 |            |
 |             | Signal strength projected into interval 1-9:    |            |
 |             |  1: minimum possible signal strength            |            |
 |             |  5: threshold for good S/N ratio                |            |
 |             |  9: maximum possible signal strength            |            |
 |             |  0 or blank: not known, don't care              |            |
 +-------------+-------------------------------------------------+------------+
 """


class SystemCode(Enum):
    GPS = "G"
    GLONASS = "R"
    GALILEO = "E"
    BEIDOU = "C"
    QZSS = "J"
    IRNSS = "I"
    SBAS = "S"
    NAVIC = "N"
    GEOSTATIONARY = "G"
    MIXED = "M"


@dataclass
class Header:

    # fields marked as Optional are listed as optional in the RINEX specification
    rinex_version: str
    file_type: str
    system_code: str
    program_name: str
    run_by: str
    date: datetime
    marker_name: str
    marker_number: str
    marker_type: str
    observer: str
    agency: str
    receiver_number: str
    receiver_type: str
    receiver_version: str
    antenna_number: str
    antenna_type: str
    approximate_position: Tuple[float, float, float]
    antenna_offset: Tuple[float, float, float]
    default_wavelength_factors: Tuple[int, int]
    wavelength_factors: Dict[str, Tuple[int, int]]
    observation_types: List[str]
    interval: Optional[float]
    time_system: Optional[str]
    time_of_first_obs: datetime
    time_of_last_obs: Optional[datetime]
    is_receiver_clock_offset_applied: Optional[bool]
    num_leap_seconds: Optional[int]
    number_of_satellites: Optional[int]
    number_of_obs: Optional[Dict[str, List[int]]]  # satellite_id -> [number_of_obs]

    comment_lines: List[str]


def parse_rinex_version_type(line: str) -> Tuple[str, str, str]:
    rinex_version = line[:9].strip()
    file_type = line[20:21].strip()
    system_code = line[40:41].strip()
    return rinex_version, file_type, system_code


def format_rinex_version_type(
    rinex_version: str, file_type: str, system_code: str
) -> str:
    return f"{rinex_version:<9}{'': <10}{file_type: <20}{system_code: <20}"


def parse_program_run_by_date(line: str) -> Tuple[str, str, datetime | str]:
    program_name = line[:20].strip()
    run_by = line[20:40].strip()
    date_str = line[40:60].strip()
    try:
        date = datetime.strptime(date_str, "%Y %m %d %H %M %S")
    except ValueError:
        date = date_str
    return program_name, run_by, date


def format_program_run_by_date(
    program_name: str, run_by: str, date: datetime | str
) -> str:
    if isinstance(date, datetime):
        date_str = date.strftime("%Y %m %d %H %M %S")
    else:
        date_str = date
    return f"{program_name:<20}{run_by:<20}{date_str: <20}"


def parse_marker_name(line: str) -> str:
    return line[:60].strip()


def format_marker_name(marker_name: str) -> str:
    return f"{marker_name: <60}"


def parse_marker_number(line: str) -> str:
    return line[:20].strip()


def format_marker_number(marker_number: str) -> str:
    return f"{marker_number: <20}"


def parse_observer_agency(line: str) -> Tuple[str, str]:
    observer = line[:20].strip()
    agency = line[20:60].strip()
    return observer, agency


def format_observer_agency(observer: str, agency: str) -> str:
    return f"{observer: <20}{agency: <40}"


def parse_receiver_type_version(line: str) -> Tuple[str, str, str]:
    receiver_number = line[:20].strip()
    receiver_type = line[20:40].strip()
    receiver_version = line[40:60].strip()
    return receiver_number, receiver_type, receiver_version


def format_receiver_type_version(
    receiver_number: str, receiver_type: str, receiver_version: str
) -> str:
    return f"{receiver_number: <20}{receiver_type: <20}{receiver_version: <20}"


def parse_antenna_type(line: str) -> Tuple[str, str]:
    antenna_number = line[:20].strip()
    antenna_type = line[20:40].strip()
    return antenna_number, antenna_type


def format_antenna_type(antenna_number: str, antenna_type: str) -> str:
    return f"{antenna_number: <20}{antenna_type: <40}"


def parse_approx_position_xyz(line: str) -> Tuple[float, float, float]:
    x = float(line[:14])
    y = float(line[14:28])
    z = float(line[28:42])
    return (x, y, z)


def format_approx_position_xyz(x: float, y: float, z: float) -> str:
    return f"{x:14.4f}{y:14.4f}{z:14.4f}"


def parse_antenna_offset(line: str) -> Tuple[float, float, float]:
    x = float(line[:14])
    y = float(line[14:28])
    z = float(line[28:42])
    return (x, y, z)


def format_antenna_offset(x: float, y: float, z: float) -> str:
    return f"{x:14.4f}{y:14.4f}{z:14.4f}"


def parse_default_wavelength_factors(line: str) -> Tuple[int, int]:
    l1_factor = int(line[:6])
    l2_factor = int(line[6:12])
    num_sats_str = line[12:18].strip()
    if num_sats_str:
        num_sats = int(num_sats_str)
        if num_sats > 0:
            raise ValueError(
                "WAVELENGTH FACT L1/2 line with satellite count is not supported"
            )
    return l1_factor, l2_factor


def format_default_wavelength_factors(l1_factor: int, l2_factor: int) -> str:
    return f"{l1_factor:6d}{l2_factor:6d}{'': <18}"


def parse_obs_types(lines: List[str]) -> List[str]:
    # these entries use continuation lines, so we need to keep track of how many obs codes we have left
    num_obs_types_str = lines[0][:6].strip()
    if not num_obs_types_str:
        raise ValueError("Invalid # / TYPES OF OBSERV line: missing number of obs types")
    num_obs_types = int(num_obs_types_str)
    obs_codes = []

    for line in lines:
        obs_codes_strs = line[6:60].strip().split()
        for obs_code in obs_codes_strs:
            obs_codes.append(obs_code)
            num_obs_types -= 1
    
    return obs_codes


def format_obs_types(obs_types: List[str]) -> List[str]:
    lines = []
    num_obs_types = len(obs_types)
    first_obs_types_str = f"{num_obs_types:6d}"
    current_line = first_obs_types_str
    for obs_type in obs_types:
        if len(current_line) + 4 > 60:
            lines.append(f"{current_line: <60}")
            current_line = "      "
        current_line += f"{obs_type:4}"
    if current_line.strip():
        lines.append(f"{current_line: <60}")
    return lines


def parse_interval(line: str) -> float:
    return float(line[:10])


def format_interval(interval: float) -> str:
    return f"{interval:10.3f}"


def parse_time_of_first_obs(line: str) -> Tuple[datetime | str, str]:
    time_of_first_obs_str = line[:43].strip()
    try:
        time_of_first_obs = datetime.strptime(
            time_of_first_obs_str, "%Y-%m-%d %H:%M:%S.%f"
        )
    except ValueError:
        time_of_first_obs = time_of_first_obs_str
    time_system = line[48:51].strip()
    return time_of_first_obs, time_system


def format_time_of_first_obs(
    time_of_first_obs: datetime | str, time_system: str
) -> str:
    if isinstance(time_of_first_obs, datetime):
        time_of_first_obs_str = time_of_first_obs.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        time_of_first_obs_str = time_of_first_obs
    return f"{time_of_first_obs_str}{'': <5}{time_system: <3}"


def parse_time_of_last_obs(line: str) -> Tuple[datetime | str, str]:
    time_of_last_obs_str = line[:43].strip()
    try:
        time_of_last_obs = datetime.strptime(
            time_of_last_obs_str, "%Y-%m-%d %H:%M:%S.%f"
        )
    except ValueError:
        time_of_last_obs = time_of_last_obs_str
    time_system = line[48:51].strip()
    return time_of_last_obs, time_system


def format_time_of_last_obs(time_of_last_obs: datetime | str, time_system: str) -> str:
    if isinstance(time_of_last_obs, datetime):
        time_of_last_obs_str = time_of_last_obs.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        time_of_last_obs_str = time_of_last_obs
    return f"{time_of_last_obs_str}{'': <5}{time_system: <3}"


def parse_receiver_clock_offset_applied(line: str) -> bool:
    is_receiver_clock_offset_applied = line[:20].strip() == "1"
    return is_receiver_clock_offset_applied


def format_receiver_clock_offset_applied(is_receiver_clock_offset_applied: bool) -> str:
    return f"{1 if is_receiver_clock_offset_applied else 0: <20}"


# TODO this is wrong / unfinished
def parse_leap_seconds(line: str) -> int:
    leap_seconds_str = line[:6].strip()
    if leap_seconds_str:
        current_leap_seconds = int(leap_seconds_str)
    else:
        current_leap_seconds = 0
    return current_leap_seconds


def format_leap_seconds(current_leap_seconds: int) -> str:
    return f"{current_leap_seconds: >6}"

def parse_number_of_satellites(line: str) -> int:
    return int(line[:6].strip())


def format_number_of_satellites(number_of_satellites: int) -> str:
    return f"{number_of_satellites: >6}"


def parse_number_of_obs(
    lines: List[str], obs_types: List[str]
) -> Dict[str, List[int]]:
    entries: Dict[str, List[int]] = {}
    raise NotImplementedError()
    # for system_code, obs_codes in sys_obs_types_entries.items():
    #     entries[system_code] = {obs_code: 0 for obs_code in obs_codes}
    return entries


def format_number_of_obs(
    entries: Dict[str, List[int]]
) -> List[str]:
    lines = []
    # for system_code, obs_entries in entries.items():
    raise NotImplementedError()



# def parse_float_or_nan(float_str: str) -> float:
#     try:
#         return float(float_str)
#     except ValueError:
#         return float("nan")


# def parse_glonass_slot_frequencies(lines: Iterable[str]) -> List[Tuple[str, int]]:
#     entries: List[Tuple[str, int]] = []
#     lines = iter(lines)
#     line = next(lines, None)
#     if line is None:
#         return entries
#     num_satellites = int(line[:4])
#     if num_satellites < 1:
#         return entries
#     while len(entries) < num_satellites and line is not None:
#         i = 4
#         while i + 7 <= 60:
#             sat_id = line[i : i + 3].strip().replace(" ", "0")
#             if not sat_id:
#                 raise ValueError(
#                     f"Satellite ID not present for `GLONASS SLOT / FRQ #` entry; {line[i:i + 3]}"
#                 )
#             freq_num = int(line[i + 4 : i + 6])
#             entries.append((sat_id, freq_num))
#             if len(entries) >= num_satellites:
#                 break
#             i += 7
#         if len(entries) >= num_satellites:
#             break
#         line = next(lines, None)
#     return entries


# def format_glonass_slot_frequencies(entries: List[Tuple[int, int]]) -> List[str]:
#     lines = []
#     num_satellites = len(entries)
#     line = f"{num_satellites:4}"
#     entry_index = 0
#     while entry_index < num_satellites:
#         prn, freq_num = entries[entry_index]
#         line += f"R{prn:0<2} {freq_num:2} "
#         if len(line) + 7 > 60:
#             lines.append(line)
#             line = " " * 4
#         entry_index += 1
#     return lines


# def parse_glonass_phase_bias_corrections(line: str) -> Dict[str, float]:
#     entries: Dict[str, float] = {}
#     if line[:60].strip():
#         return entries
#     for i in range(4):
#         obs_code = line[i * 13 + 1 : i * 13 + 3].strip()
#         phase_bias_str = line[i * 13 + 4 : i * 13 + 13].strip()
#         phase_bias = parse_float_or_nan(phase_bias_str)
#         entries[obs_code] = phase_bias
#     return entries


# def format_glonass_phase_bias_corrections(entries: Dict[str, float]) -> str:
#     return "".join(
#         f" {obs_code: <3} {phase_bias:8.3f}" for obs_code, phase_bias in entries.items()
#     )



LABEL_END_OF_HEADER = "END OF HEADER"
LABEL_COMMENT = "COMMENT"
LABEL_RINEX_VERSION_TYPE = "RINEX VERSION / TYPE"
LABEL_PGM_RUN_BY_DATE = "PGM / RUN BY / DATE"
LABEL_MARKER_NAME = "MARKER NAME"
LABEL_MARKER_NUMBER = "MARKER NUMBER"
LABEL_MARKER_TYPE = "MARKER TYPE"
LABEL_OBSERVER_AGENCY = "OBSERVER / AGENCY"
LABEL_REC_TYPE_VERS = "REC # / TYPE / VERS"
LABEL_ANT_TYPE = "ANT # / TYPE"
LABEL_APPROX_POSITION_XYZ = "APPROX POSITION XYZ"
LABEL_ANTENNA_DELTA_HEN = "ANTENNA: DELTA H/E/N"
LABEL_WAVELENGTH_FACTORS = "WAVELENGTH FACT L1/2"
# LABEL_ANTENNA_DELTA_XYZ = "ANTENNA: DELTA X/Y/Z"
# LABEL_ANTENNA_PHASECENTER = "ANTENNA: PHASECENTER"
# LABEL_ANTENNA_BORESIGHT = "ANTENNA: BORESIGHT"
# LABEL_ANTENNA_ZERODIR_AZI = "ANTENNA: ZERODIR A/Z"
# LABEL_ANTENNA_ZERODIR_XYZ = "ANTENNA: ZERODIR X/Y/Z"
# LABEL_CENTER_OF_MASS_XYZ = "CENTER OF MASS: XYZ"
# LABEL_SYS_NUM_OBS = "SYS / # / OBS TYPES"
# LABEL_SIGNAL_STRENGTH_UNIT = "SIGNAL STRENGTH UNIT"
LABEL_NUM_TYPES_OF_OBS = "# / TYPES OF OBSERV"
LABEL_INTERVAL = "INTERVAL"
LABEL_TIME_OF_FIRST_OBS = "TIME OF FIRST OBS"
LABEL_TIME_OF_LAST_OBS = "TIME OF LAST OBS"
LABEL_RCV_CLOCK_OFFS_APPL = "RCV CLOCK OFFS APPL"
# LABEL_SYS_DCBS_APPLIED = "SYS / DCBS APPLIED"
# LABEL_SYS_PCVS_APPLIED = "SYS / PCVS APPLIED"
# LABEL_SYS_SCALE_FACTOR = "SYS / SCALE FACTOR"
# LABEL_SYS_PHASE_SHIFT = "SYS / PHASE SHIFT"
LABEL_GLONASS_SLOT_FRQ = "GLONASS SLOT / FRQ #"
LABEL_GLONASS_COD_PHS_BIS = "GLONASS COD/PHS/BIS"
LABEL_LEAP_SECONDS = "LEAP SECONDS"
LABEL_NUM_SATELLITES = "# OF SATELLITES"
LABEL_PRN_NUM_OBS = "PRN / # OF OBS"


def parse_header(input: io.TextIOWrapper, strict: bool = True) -> Header:
    rinex_version: str | None = None
    file_type: str | None = None
    system_code: str | None = None
    program_name: str | None = None
    run_by: str | None = None
    date: datetime | str | None = None
    marker_name: str | None = None
    marker_number: str | None = None
    marker_type: str | None = None
    observer: str | None = None
    agency: str | None = None
    receiver_number: str | None = None
    receiver_type: str | None = None
    receiver_version: str | None = None
    antenna_number: str | None = None
    antenna_type: str | None = None
    approximate_position: Tuple[float, float, float] | None = None
    antenna_offset: Tuple[float, float, float] | None = None
    # antenna_offset_frame: AntennaOffsetFrame | None = None
    # antenna_phase_center_offsets: Optional[List[AntennaPhaseCenterOffsetEntry]] = None
    # antenna_boresight: Optional[Tuple[float, float, float]] = None
    # antenna_zerodir_azi: Optional[float] = None
    # antenna_zerodir_xyz: Optional[Tuple[float, float, float]] = None
    # vehicle_center_of_mass_xyz: Optional[Tuple[float, float, float]] = None
    # system_obs_types: Dict[str, List[str]] | None = None
    default_wavelength_factors: Optional[Tuple[int, int]] = None
    wavelength_factors: Optional[Dict[str, Tuple[int, int]]] = None
    obs_types: List[str] = []
    # signal_strength_unit: Optional[str] = None
    interval: Optional[float] = None
    time_system: str | None = None
    time_of_first_obs: Optional[datetime | str] = None
    time_of_last_obs: Optional[datetime | str] = None
    is_receiver_clock_offset_applied: Optional[bool] = None
    # applied_dcbs: Optional[Dict[str, float]] = None  # system_code -> dcb
    # applied_pcvs: Optional[Dict[str, float]] = None  # system_code -> pcv
    # applied_scale_factors: Optional[Dict[str, Dict[str, float]]] = (
    #     None  # system_code -> obs_code -> scale_factor
    # )
    # applied_phase_shifts: (
    #     Dict[str, Dict[str, Tuple[float, Optional[List[str]]]]] | None
    # ) = None  # system_code -> obs_code -> (phase_shift, [satellite_ids])
    # glonass_slot_frequencies: List[Tuple[str, int]] | None = None
    # glonass_phase_bias_corrections: Dict[str, float] | None = None
    current_leap_seconds: Optional[int] = None
    number_of_satellites: Optional[int] = None
    number_of_obs: Optional[Dict[str, List[int]]] = (
        None  # satellite_id -> [number_of_obs]
    )

    comment_lines: List[str] = []

    if strict:
        line = input.readline()
        line_label = line[60:].strip()
        if not line_label.startswith("RINEX VERSION / TYPE"):
            raise ValueError(
                f"Invalid RINEX file; expected `RINEX VERSION / TYPE`, got {line_label}"
            )
        rinex_version, file_type, system_code = parse_rinex_version_type(line)

    # Single-line header entries are parsed immediately
    # Multi-line entries are agregated in a list and parsed once the end of the header is reached
    # antenna_phase_center_lines: List[str] = []
    num_types_obs_lines: List[str] = []
    # sys_dcbs_applied_lines: List[str] = []
    # sys_pcvs_applied_lines: List[str] = []
    sys_phase_shift_lines: List[str] = []
    glonass_slot_freq_lines: List[str] = []
    prn_num_obs_lines: List[str] = []

    for line in input:
        line_label = line[60:].strip()
        if line_label == LABEL_END_OF_HEADER:
            break
        elif line_label == LABEL_COMMENT:
            comment_lines.append(line[:60].strip())
        elif line_label == LABEL_RINEX_VERSION_TYPE:
            rinex_version, file_type, system_code = parse_rinex_version_type(line)
        elif line_label == LABEL_PGM_RUN_BY_DATE:
            program_name, run_by, date = parse_program_run_by_date(line)
        elif line_label == LABEL_MARKER_NAME:
            marker_name = parse_marker_name(line)
        elif line_label == LABEL_MARKER_NUMBER:
            marker_number = parse_marker_number(line)
        elif line_label == LABEL_OBSERVER_AGENCY:
            observer, agency = parse_observer_agency(line)
        elif line_label == LABEL_REC_TYPE_VERS:
            receiver_number, receiver_type, receiver_version = (
                parse_receiver_type_version(line)
            )
        elif line_label == LABEL_ANT_TYPE:
            antenna_number, antenna_type = parse_antenna_type(line)
        elif line_label == LABEL_APPROX_POSITION_XYZ:
            approximate_position = parse_approx_position_xyz(line)
        elif line_label == LABEL_ANTENNA_DELTA_HEN:
            antenna_offset = parse_antenna_offset(line)
        elif line_label == LABEL_WAVELENGTH_FACTORS:
            default_wavelength_factors = parse_default_wavelength_factors(line)
        elif line_label == LABEL_NUM_TYPES_OF_OBS:
            num_types_obs_lines.append(line)
        elif line_label == LABEL_INTERVAL:
            interval = parse_interval(line)
        elif line_label == LABEL_TIME_OF_FIRST_OBS:
            time_of_first_obs, time_system = parse_time_of_first_obs(line)
        elif line_label == LABEL_TIME_OF_LAST_OBS:
            time_of_last_obs, time_system = parse_time_of_last_obs(line)
        elif line_label == LABEL_RCV_CLOCK_OFFS_APPL:
            is_receiver_clock_offset_applied = parse_receiver_clock_offset_applied(line)
        elif line_label == LABEL_GLONASS_SLOT_FRQ:
            glonass_slot_freq_lines.append(line)
        elif line_label == LABEL_LEAP_SECONDS:
            leap_second_metadata = parse_leap_seconds(line)
        elif line_label == LABEL_NUM_SATELLITES:
            number_of_satellites = parse_number_of_satellites(line)
        elif line_label == LABEL_PRN_NUM_OBS:
            prn_num_obs_lines.append(line)
        else:
            if strict:
                raise ValueError(f"Unknown header line label: {line_label}")
            else:
                logging.warning(f"Unknown header line label: {line_label}")

    # if antenna_phase_center_lines:
    #     antenna_phase_center_offsets = parse_antenna_phase_center_offsets(
    #         antenna_phase_center_lines
    #     )
    if num_types_obs_lines:
        obs_types = parse_obs_types(num_types_obs_lines)
    # if sys_dcbs_applied_lines:
    #     applied_dcbs = parse_applied_dcbs(sys_dcbs_applied_lines)
    # if sys_pcvs_applied_lines:
    #     applied_pcvs = parse_applied_pcvs(sys_pcvs_applied_lines)
    # if sys_phase_shift_lines:
    #     applied_phase_shifts = parse_applied_phase_shifts(sys_phase_shift_lines)
    # if glonass_slot_freq_lines:
    #     glonass_slot_frequencies = parse_glonass_slot_frequencies(
    #         glonass_slot_freq_lines
    #     )
    if prn_num_obs_lines:
        number_of_obs = parse_number_of_obs(prn_num_obs_lines, obs_types)

    return Header(
        rinex_version=rinex_version,  # type: ignore
        file_type=file_type,  # type: ignore
        system_code=system_code,  # type: ignore
        program_name=program_name,  # type: ignore
        run_by=run_by,  # type: ignore
        date=date,  # type: ignore
        marker_name=marker_name,  # type: ignore
        marker_number=marker_number,  # type: ignore
        marker_type=marker_type,  # type: ignore
        observer=observer,  # type: ignore
        agency=agency,  # type: ignore
        receiver_number=receiver_number,  # type: ignore
        receiver_type=receiver_type,  # type: ignore
        receiver_version=receiver_version,  # type: ignore
        antenna_number=antenna_number,  # type: ignore
        antenna_type=antenna_type,  # type: ignore
        approximate_position=approximate_position,  # type: ignore
        antenna_offset=antenna_offset,  # type: ignore
        default_wavelength_factors=default_wavelength_factors,  # type: ignore
        wavelength_factors=wavelength_factors,  # type: ignore
        observation_types=obs_types,
        interval=interval,
        time_system=time_system,
        time_of_first_obs=time_of_first_obs,  # type: ignore
        time_of_last_obs=time_of_last_obs,  # type: ignore
        is_receiver_clock_offset_applied=is_receiver_clock_offset_applied,
        num_leap_seconds=leap_second_metadata,  # type: ignore
        number_of_satellites=number_of_satellites,
        number_of_obs=number_of_obs,
        comment_lines=comment_lines,
    )


def format_header(header: Header) -> List[str]:
    lines = []
    lines.append(
        f"{format_rinex_version_type(header.rinex_version, header.file_type, header.system_code): <60}{LABEL_RINEX_VERSION_TYPE}"
    )
    lines.append(
        f"{format_program_run_by_date(header.program_name, header.run_by, header.date): <60}{LABEL_PGM_RUN_BY_DATE}"
    )
    lines.append(f"{format_marker_name(header.marker_name): <60}{LABEL_MARKER_NAME}")
    lines.append(
        f"{format_marker_number(header.marker_number): <60}{LABEL_MARKER_NUMBER}"
    )
    lines.append(
        f"{format_observer_agency(header.observer, header.agency): <60}{LABEL_OBSERVER_AGENCY}"
    )
    lines.append(
        f"{format_receiver_type_version(header.receiver_number, header.receiver_type, header.receiver_version): <60}{LABEL_REC_TYPE_VERS}"
    )
    lines.append(
        f"{format_antenna_type(header.antenna_number, header.antenna_type): <60}{LABEL_ANT_TYPE}"
    )
    lines.append(
        f"{format_approx_position_xyz(*header.approximate_position): <60}{LABEL_APPROX_POSITION_XYZ}"
    )
    lines.append(
        f"{format_antenna_offset(*header.antenna_offset): <60}{LABEL_ANTENNA_DELTA_HEN}"
    )
    # TODO: wavelength fact; num types observ
    if header.interval:
        lines.append(f"{format_interval(header.interval): <60}{LABEL_INTERVAL}")
    assert header.time_system is not None
    lines.append(
        f"{format_time_of_first_obs(header.time_of_first_obs, header.time_system): <60}{LABEL_TIME_OF_FIRST_OBS}"
    )
    if header.time_of_last_obs:
        lines.append(
            f"{format_time_of_last_obs(header.time_of_last_obs, header.time_system): <60}{LABEL_TIME_OF_LAST_OBS}"
        )
    if header.is_receiver_clock_offset_applied is not None:
        lines.append(
            f"{format_receiver_clock_offset_applied(header.is_receiver_clock_offset_applied): <60}{LABEL_RCV_CLOCK_OFFS_APPL}"
        )
    if header.num_leap_seconds is not None:
        lines.append(
            f"{format_leap_seconds(header.num_leap_seconds): <60}{LABEL_LEAP_SECONDS}"
        )
    if header.number_of_satellites:
        lines.append(
            f"{format_number_of_satellites(header.number_of_satellites): <60}{LABEL_NUM_SATELLITES}"
        )
    if header.number_of_obs:
        lines.extend(
            format_number_of_obs(header.number_of_obs)
        )
    lines.append(f"{'': >60}{LABEL_END_OF_HEADER}")
    return lines


# we need a data structure to hold epoch records, since we might want to log them to memory before writing them to file

# there are two modes we consider parsing in
# one where we do not preallocate any space -- then we just append epoch records to a list and sort it out later
# the other where we know the PRN / num obs and preallocate arrays for each satellite

# def parse_RINEX_int(val_str: str) -> Optionalint:
#     try:
#         return int(val_str)
#     except Exception:
#         return None


@dataclass
class EpochRecord:
    epoch: datetime
    epoch_flag: int
    transmitters: Dict[str, List[float]]  # transmitter_id -> (obs_values, ...)


def parse_epoch_header(
    line: str, strict: bool = True
) -> Optional[Tuple[datetime, int, int]]:
    try:
        year = int(line[2:6])
        month = int(line[7:9])
        day = int(line[10:12])
        hour = int(line[13:15])
        minute = int(line[16:18])
        seconds = float(line[19:29])
        epoch = datetime(
            year, month, day, hour, minute, int(seconds), int(1e6 * (seconds % 1))
        )
        epoch_flag_str = line[30:32].strip()
        if epoch_flag_str:
            epoch_flag = int(epoch_flag_str)
        else:
            epoch_flag = 0
        num_sats = int(line[32:35])
        return epoch, epoch_flag, num_sats
    except Exception as e:
        if strict:
            raise ValueError(f"Error parsing epoch header: {e}")
        return None


def format_epoch_header(epoch: datetime, epoch_flag: int, num_sats: int) -> str:
    return f">{epoch:%Y %m %d %H %M %S}{epoch_flag: <2}{num_sats: <3}"


# def format_epoch_header(epoch: datetime, epoch_flag: int, num_sats: int) -> str:
#     time_str = epoch.strftime("%Y %m %d %H %M %S")


@dataclass
class RinexObsTransmitterEntry:
    index: List[int]  # list of indices into the `epochs` list for each epoch where this transmitter is observed
    obs: Dict[str, List[float]]  # dict of obs values for each epoch
    lli_flags: Optional[Dict[str, List[int]]] = None  # dict of LLI flags for each epoch, only for phase observations
    ssi_flags: Optional[Dict[str, List[int]]] = None  # dict of SSI flags for each epoch, only for phase observations

@dataclass
class RinexObs:
    epochs: List[datetime]
    epoch_flags: List[int]
    transmitters: Dict[str, RinexObsTransmitterEntry]

def parse_observations(
    input: io.TextIOWrapper,
    obs_types: List[str],
    parse_ssi: bool = True,
    parse_lli: bool = True,
    strict: bool = True,
    use_strict_epoch_line_format: bool = True,
    default_century: int = 2000,
    verbose_warnings: bool = True,
) -> RinexObs:

    obs_data = RinexObs([], [], {})

    epoch_index = 0
    line_number = 0
    comment_lines: List[str] = []

    def get_next_obs_line(input: io.TextIOWrapper) -> str:
        nonlocal line_number
        while line := input.readline():
            line_number += 1
            if line[60:].strip() == LABEL_COMMENT:
                comment_lines.append(line[:60].strip())
            else:
                return line
        return ""

    while line := get_next_obs_line(input):
        # We always are looking for an epoch start at this level
        try:
            year = int(line[:4].strip())
            if year < 100:
                year = default_century + year
            month = int(line[4:7])
            day = int(line[7:10])
            hour = int(line[10:13])
            minute = int(line[13:16])
            seconds = float(line[16:25])
            microseconds = int(1e6 * (seconds % 1))
            seconds = int(seconds)
            epoch = datetime(
                year, month, day, hour, minute, seconds, microseconds
            )
            epoch_flag = int(line[25:28])
            current_epoch_num_sats = int(line[29:32])
            # There is space for (80 - 32) / 3 = 16 satellite ids
            # If there are more than 16, then they continue on the next line
            # A general approach is to consume lines until we have determined all sat IDs
            # TODO: techinically spec says if there are more than 12, continue on next line
            # should change this to 12
            current_epoch_sat_ids = []
            # line = line[32:].strip()
            line = line[32:].strip()
            while len(current_epoch_sat_ids) < current_epoch_num_sats:
                current_epoch_sat_ids.append(line[:3].replace(" ", "0"))
                line = line[3:]
                if line == "" and len(current_epoch_sat_ids) < current_epoch_num_sats:
                    line = input.readline()
                    if strict:
                        assert line[:32].strip() == ""
                    line = line.strip()
                    if strict:
                        assert (
                            len(line) % 3 == 0
                        )  # sanity check -- each sat ID takes 3 chars
            
            obs_data.epochs.append(epoch)
            obs_data.epoch_flags.append(epoch_flag)

            # The way we parse in RINEX 2, we initially don't know which observations correspond to which sat IDs
            # If a value chunk is blank, we don't parse the observation
            # If later on we find that the value chunk for that satellite is not blank, we backfill with nans
            for sat_id in current_epoch_sat_ids:
                # Create new entry if `sat_id` is new
                if sat_id not in obs_data.transmitters:
                    obs_data.transmitters[sat_id] = RinexObsTransmitterEntry([], {})
                sat_data = obs_data.transmitters[sat_id]
                prev_sat_data_length = len(sat_data.index)
                sat_data.index.append(epoch_index)

                # Technically, each line of observation values contains up to 5 entries
                # Each entry is of width 16, starting at index 0
                # Some bad receivers might write out more entries per line
                # We offer two approachs:
                #  1) strict mode, where we expect at most 5 entries per line
                #     and raise an error if there are more
                #  2) non-strict mode, where we consume 16-char chunks until we
                #     have consumed all obs values for the satellite, even if
                #     that means consuming more than 5 entries per line
                chunk_index = 0
                line = get_next_obs_line(input)
                for i, obs_code in enumerate(obs_types):
                    # Handle new lines
                    if chunk_index > 4:
                        if (
                            use_strict_epoch_line_format or
                            chunk_index * 16 >= len(line)
                        ):
                            line = get_next_obs_line(input)
                            chunk_index = 0
                    obs_val = None
                    ssi_flag = None
                    lli_flag = None

                    i0 = chunk_index * 16
                    obs_val_str = line[i0:i0 + 14].strip()
                    if obs_val_str != "":
                        obs_val = float(obs_val_str)
                    if i0 + 14 < len(line) and line[i0 + 14].strip() != "":
                        lli_flag = int(line[i0 + 14])
                    if i0 + 15 < len(line) and line[i0 + 15].strip() != "":
                        ssi_flag = int(line[i0 + 15])

                    if obs_code not in sat_data.obs:
                        if obs_val is not None:
                            sat_data.obs[obs_code] = [float("nan")] * prev_sat_data_length
                        else:
                            pass
                    if obs_val is not None:
                        sat_data.obs[obs_code].append(obs_val)
                    elif obs_code in sat_data.obs:
                        sat_data.obs[obs_code].append(float("nan"))
                    
                    if parse_ssi:
                        if sat_data.ssi_flags is None:
                            sat_data.ssi_flags = {}
                        if obs_code not in sat_data.ssi_flags:
                            if ssi_flag is not None:
                                sat_data.ssi_flags[obs_code] = [-1] * prev_sat_data_length + [ssi_flag]
                        else:
                            if ssi_flag is not None:
                                sat_data.ssi_flags[obs_code].append(ssi_flag)
                            else:
                                sat_data.ssi_flags[obs_code].append(-1)
                    if parse_lli:
                        if sat_data.lli_flags is None:
                            sat_data.lli_flags = {}
                        if obs_code not in sat_data.lli_flags:
                            if lli_flag is not None:
                                sat_data.lli_flags[obs_code] = [-1] * prev_sat_data_length + [lli_flag]
                        else:
                            if lli_flag is not None:
                                sat_data.lli_flags[obs_code].append(lli_flag)
                            else:
                                sat_data.lli_flags[obs_code].append(-1)
                    
                    chunk_index += 1
        except Exception as e:
            if strict:
                print(line)
                print(line_number)
                # print(sat_data.lli_flags)
                raise ValueError(f"Error parsing epoch: {e}")
            else:
                if verbose_warnings:
                    logging.warning(f"Error parsing epoch header: {e};\n{line}")
        # Update epoch index (should be index corresponding to entries in next iteration)
        #  NOTE: we do this in this particular way in case errors occur during parsing
        epoch_index = len(obs_data.epochs)
    
    return obs_data
                

class Dataset:

    def __init__(self, include_ssi: bool = True, include_lli: bool = True) -> None:
        self.header: Optional[Header] = None
        self.observations: Optional[RinexObs] = None

        # These should not be overwritten once set, since they determine the content of the observations
        self._include_ssi = include_ssi
        self._include_lli = include_lli

    def load(self, io: io.TextIOWrapper, strict: bool = True, verbose: bool = True) -> None:
        # todo: add options for epochs, etc.
        self.header = parse_header(io, strict)
        self.observations = parse_observations(
            io,
            self.header.observation_types,
            self._include_ssi,
            self._include_lli,
            strict,
            verbose_warnings=verbose
        )

    def save(self, io: io.TextIOWrapper) -> None:
        raise NotImplementedError()
        if self.header is None or self.observations is None:
            raise ValueError("Cannot save dataset without header and observations")
        lines = format_header(self.header)
        lines.extend(
            format_observations(self.observations, self.header.system_obs_types)
        )

    def get_obs_arrays(
        self, strip_all_nan: bool = True
    ) -> Tuple[NDArray[np.int64], Dict[str, Dict[str, NDArray[np.float64]]]]:
        """
        Get observations as numpy arrays.

        Returns:
            obs_gpst_epochs: numpy array of observation epochs (GPS seconds)
            obs_arrays: dict of satellite ID to dict of observation code to numpy array of observation values
        """

        obs_epochs: List[int] = []
        obs_arrays: Dict[str, Dict[str, np.ndarray]] = (
            {}
        )  # sat_id -> obs_code -> [obs_values]
        GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

        assert self.header is not None and self.observations is not None, "Header and observations must be loaded before calling `get_obs_arrays`"
        obs_types = self.header.observation_types

        obs_gpst_epochs = np.array([(epoch - GPS_EPOCH).total_seconds() for epoch in self.observations.epochs])
        obs_arrays = {}
        for sat_id, sat_data in self.observations.transmitters.items():
            obs_arrays[sat_id] = {}
            obs_arrays[sat_id]["index"] = np.array(sat_data.index)
            sat_obs_length = len(sat_data.index)
            for obs_code in sat_data.obs:
                obs_vals = sat_data.obs[obs_code]
                assert len(obs_vals) == sat_obs_length, f"Observation values for satellite {sat_id} have inconsistent lengths"
                obs_arrays[sat_id][obs_code] = np.array(obs_vals)
            if sat_data.lli_flags is not None:
                for obs_code, lli_flags in sat_data.lli_flags.items():
                    lli_obs_code = obs_code + "_LLI"
                    assert len(lli_flags) == sat_obs_length, f"LLI flags for satellite {sat_id} and observation code {obs_code} have inconsistent lengths"
                    obs_arrays[sat_id][lli_obs_code] = np.array(lli_flags)
            if sat_data.ssi_flags is not None:
                for obs_code, ssi_flags in sat_data.ssi_flags.items():
                    ssi_obs_code = obs_code + "_SSI"
                    assert len(ssi_flags) == sat_obs_length, f"SSI flags for satellite {sat_id} and observation code {obs_code} have inconsistent lengths"
                    obs_arrays[sat_id][ssi_obs_code] = np.array(ssi_flags)

        return obs_gpst_epochs, obs_arrays

        # create dict of sat ID
        # each has obs to value lists for each obs code
        # if all obs are nan, and prune, ignore that obs code for that satellite
        # keep track of epoch index; also have index list for each satellite
        # at end, convert to numpy arrays
