from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import io
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

VALID_GNSS_TIME_SYSTEM_IDS = ["GPS", "GLO", "GAL", "QZS", "BDT", "IRN"]


class AntennaOffsetFrame(Enum):
    XYZ = "XYZ"
    ENU = "ENU"
    UNKNOWN = "Unknown"


@dataclass
class AntennaPhaseCenterOffsetEntry:
    system_code: str
    obs_code: str
    offset: Tuple[float, float, float]
    offset_frame: AntennaOffsetFrame


@dataclass
class LeapSecondMetadata:
    current_leap_seconds: int
    next_leap_seconds: Optional[int]
    past_leap_seconds: Optional[int]
    effective_week_number: int
    effective_day_number: int
    time_system: str


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
    approximate_position: List[float]
    antenna_offset: List[float]
    antenna_offset_frame: AntennaOffsetFrame
    antenna_phase_center_offsets: Optional[List[AntennaPhaseCenterOffsetEntry]]
    antenna_boresight: Optional[Tuple[float, float, float]]
    antenna_zerodir_azi: Optional[float]
    antenna_zerodir_xyz: Optional[Tuple[float, float, float]]
    vehicle_center_of_mass_xyz: Optional[Tuple[float, float, float]]
    system_obs_types: Dict[str, List[str]]  # system_code -> [obs_codes]
    signal_strength_unit: Optional[str]
    interval: Optional[float]
    time_system: str
    time_of_first_obs: Optional[datetime]
    time_of_last_obs: Optional[datetime]
    is_receiver_clock_offset_applied: Optional[bool]
    applied_dcbs: Optional[Dict[str, float]]  # system_code -> dcb
    applied_pcvs: Optional[Dict[str, float]]  # system_code -> pcv
    applied_scale_factors: Optional[
        Dict[str, Dict[str, float]]
    ]  # system_code -> obs_code -> scale_factor
    applied_phase_shifts: Dict[
        str, Dict[str, Tuple[float, Optional[List[str]]]]
    ]  # system_code -> obs_code -> (phase_shift, [satellite_ids])
    glonass_slot_frequencies: List[Tuple[int, int]]
    glonass_phase_bias_corrections: Dict[str, float]
    leap_second_metadata: Optional[LeapSecondMetadata]
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


def parse_marker_type(line: str) -> str:
    return line[:20].strip()


def format_marker_type(marker_type: str) -> str:
    return f"{marker_type: <20}"


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


def parse_antenna_phase_center_offsets(
    lines: List[str],
) -> List[AntennaPhaseCenterOffsetEntry]:
    entries: List[AntennaPhaseCenterOffsetEntry] = []
    for line in lines:
        system_code = line[0:1].strip()
        obs_code = line[2:5].strip()
        # the next part of the spec seems to be poorly defined? so this is a bit of a hack...
        offset_frame = AntennaOffsetFrame.XYZ
        x, y, z = map(float, line[5:60].split()[:3])
        entries.append(
            AntennaPhaseCenterOffsetEntry(
                system_code, obs_code, (x, y, z), offset_frame
            )
        )
    return entries


def format_antenna_phase_center_offsets(
    entries: List[AntennaPhaseCenterOffsetEntry],
) -> List[str]:
    lines = []
    for entry in entries:
        line = f"{entry.system_code: <2}{entry.obs_code: <3}"
        line += " ".join(f"{value:14.4f}" for value in entry.offset)
        lines.append(line)
    return lines


def parse_antenna_boresight(line: str) -> List[float]:
    x = float(line[:14])
    y = float(line[14:28])
    z = float(line[28:42])
    return [x, y, z]


def format_antenna_boresight(x: float, y: float, z: float) -> str:
    return f"{x:14.4f}{y:14.4f}{z:14.4f}"


def parse_antenna_zerodir_azi(line: str) -> float:
    return float(line[:14])


def format_antenna_zerodir_azi(azi: float) -> str:
    return f"{azi:14.4f}"


def parse_antenna_zerodir_xyz(line: str) -> Tuple[float, float, float]:
    x = float(line[:14])
    y = float(line[14:28])
    z = float(line[28:42])
    return (x, y, z)


def format_antenna_zerodir_xyz(x: float, y: float, z: float) -> str:
    return f"{x:14.4f}{y:14.4f}{z:14.4f}"


def parse_center_of_mass_xyz(line: str) -> Tuple[float, float, float]:
    x = float(line[:14])
    y = float(line[14:28])
    z = float(line[28:42])
    return (x, y, z)


def format_center_of_mass_xyz(x: float, y: float, z: float) -> str:
    return f"{x:14.4f}{y:14.4f}{z:14.4f}"


def parse_system_obs_types(lines: List[str]) -> Dict[str, List[str]]:
    entries: Dict[str, List[str]] = {}
    # these entries use continuation lines, so we need to keep track of whether we're currently parsing obs codes from a previous line
    current_system_code: str | None = None
    current_obs_codes: List[str] | None = None
    current_num_obs_remaining: int | None = None

    for line in lines:
        if current_num_obs_remaining is None:
            current_system_code = line[0:1].strip()
            current_num_obs_remaining = int(line[3:6])
            current_obs_codes = []
            obs_codes_strs = line[6:60].strip().split()
            for obs_code in obs_codes_strs:
                current_obs_codes.append(obs_code)
                current_num_obs_remaining -= 1
                if current_num_obs_remaining == 0:
                    break
        else:
            assert current_system_code is not None and current_obs_codes is not None
            obs_codes_strs = line[6:60].strip().split()
            for obs_code in obs_codes_strs:
                current_obs_codes.append(obs_code)
                current_num_obs_remaining -= 1
                if current_num_obs_remaining == 0:
                    break
        if current_num_obs_remaining is not None and current_num_obs_remaining == 0:
            entries[current_system_code] = current_obs_codes
            current_system_code = None
            current_obs_codes = None
            current_num_obs_remaining = None

    return entries


def format_system_obs_types(entries: Dict[str, List[str]]) -> List[str]:
    lines = []
    for system_code, obs_codes in entries.items():
        line = f"{system_code: <3}{len(obs_codes): <3}"
        # we can fit 13 obs codes per line
        N = 13
        num_lines = max(1, (len(obs_codes) - 1) // N)
        i = 0
        while i < num_lines:
            line += " ".join(obs_codes[i * 13 : (i + 1) * 13])
            lines.append(line)
            i += 1
            if i >= num_lines:
                break
            line = "      "
    return lines


def parse_signal_strength_unit(line: str) -> str:
    return line[:20].strip()


def format_signal_strength_unit(signal_strength_unit: str) -> str:
    return f"{signal_strength_unit: <20}"


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
        print("XXXX")
        print(line)
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


def parse_applied_dcbs(lines: List[str]) -> Dict[str, float]:
    raise NotImplementedError()
    entries: Dict[str, float] = {}
    for line in lines:
        system_code = line[0:1].strip()
        program_name = line[2:19].strip()
        source_of_corrections = line[20:60].strip()
        # entries[system_code] = float(program_name)  # TODO so these are just program names, not float values?
    return entries


def format_applied_dcbs(entries: Dict[str, float]) -> List[str]:
    raise NotImplementedError()


def parse_applied_pcvs(lines: List[str]) -> Dict[str, str]:
    entries: Dict[str, float] = {}
    for line in lines:
        system_code = line[0:1].strip()
        program_name = line[2:19].strip()
        source_of_corrections = line[20:60].strip()
        # NOTE: these entries only describe the program used to make the corrections, not the corrections themselves
        entries[system_code] = (program_name, source_of_corrections)
    return entries


def format_applied_pcvs(entries: Dict[str, str]) -> List[str]:
    lines = []
    for system_code, (program_name, source_of_corrections) in entries.items():
        lines.append(
            f"{system_code: <1}{program_name: <17}{source_of_corrections: <40}"
        )
    return lines


def parse_sys_scale_factors(lines: List[str]) -> Dict[str, Dict[str, float]]:
    entries: Dict[str, List[Tuple[float, List[str] | None]]] = (
        {}
    )  # system_id -> [(scale factor, list of obs IDs)]; if list is None, applies to all obs IDs not listed
    currently_parsing_obs_codes = False
    current_system_code: str | None = None
    current_obs_codes: List[str] | None = None
    current_num_obs_remaining: int | None = None

    for line in lines:
        if not currently_parsing_obs_codes:
            current_system_code = line[0:1].strip()
            if current_system_code == "":
                raise ValueError("Invalid entry for system code in SYS / SCALE FACTOR")
            if current_system_code not in entries:
                entries[current_system_code] = []
            scale_factor = int(line[2:6])
            num_obs_remaining_str = line[8:10].strip()
            if num_obs_remaining_str:
                current_num_obs_remaining = int(num_obs_remaining_str)
            else:
                # no obs codes listed, so this applies to all obs codes
                entries[current_system_code].append((scale_factor, None))
                current_num_obs_remaining = None
            current_obs_codes = []
            obs_codes_strs = line[10:60].strip().split()
            for obs_code in obs_codes_strs:
                current_obs_codes.append(obs_code)
                current_num_obs_remaining -= 1
                if current_num_obs_remaining == 0:
                    break
        else:
            obs_codes_strs = line[10:60].strip().split()
            for obs_code in obs_codes_strs:
                assert (
                    current_system_code is not None
                    and current_obs_codes is not None
                    and current_num_obs_remaining is not None
                )
                current_obs_codes.append(obs_code)
                current_num_obs_remaining -= 1
                if current_num_obs_remaining == 0:
                    break
        if current_num_obs_remaining is not None:
            if current_num_obs_remaining == 0:
                entries[current_system_code].append((scale_factor, current_obs_codes))
                currently_parsing_obs_codes = False
                current_system_code = None
                current_obs_codes = None
                current_num_obs_remaining = None
            else:
                currently_parsing_obs_codes = True
    return entries


def format_sys_scale_factors(entries: Dict[str, Dict[str, float]]) -> List[str]:
    # before we write the entries out, we need to consolidate observations that share the same scale factor
    consolidated_entries: Dict[str, Dict[float, List[str]]] = {}
    for system_code, sys_entries in entries.items():
        for obs_codes, scale_factor in sys_entries.items():
            if system_code not in consolidated_entries:
                consolidated_entries[system_code] = {}
            if scale_factor not in consolidated_entries[system_code]:
                consolidated_entries[system_code][scale_factor] = []
            consolidated_entries[system_code][scale_factor] += obs_codes
    lines = []
    # for system_code, obs_entries in consolidated_entries.items():
    #     line = f"{system_code: <1}{scale_factor: <4}"
    #     for scale_factor, obs_codes in obs_entries:
    #         line = f"{system_code: <1}{scale_factor: <4}"
    #         if obs_codes is not None:
    #             line += " ".join(obs_codes)
    #         lines.append(line)
    raise NotImplementedError()
    return lines


def parse_float_or_nan(float_str: str) -> float:
    try:
        return float(float_str)
    except ValueError:
        return float("nan")


def parse_applied_phase_shifts(
    lines: Iterable[str],
) -> Dict[str, Dict[str, Tuple[float, Optional[List[str]]]]]:
    entries: Dict[str, Dict[str, Tuple[float, Optional[List[str]]]]] = {}
    lines = iter(lines)
    while line := next(lines, None):
        system_code = line[0:1].strip()
        obs_code = line[2:5].strip()
        phase_shift = parse_float_or_nan(line[6:14])
        num_satellites_str = line[16:18].strip()
        if num_satellites_str:
            num_satellites = int(num_satellites_str)
            satellite_ids = []
        else:
            num_satellites = 0
            satellite_ids = None
            line = None
        while line is not None:
            satellite_ids_str = line[18:60].strip()
            if not satellite_ids_str:
                raise ValueError(
                    f"Invalid entry for SYS / PHASE SHIFT; num_satellites = {num_satellites} > 0 but no more satellite IDs listed"
                )
            satellite_ids += list(satellite_ids_str.split())
            if len(satellite_ids) >= num_satellites:
                break
            line = next(lines, None)
        if system_code not in entries:
            entries[system_code] = {}
        entries[system_code][obs_code] = (phase_shift, satellite_ids)
    return entries


def format_applied_phase_shifts(
    entries: Dict[str, Dict[str, Tuple[float, Optional[List[str]]]]]
) -> List[str]:
    # raise NotImplementedError()
    lines = []
    for system_code, obs_entries in entries.items():
        for obs_code, (phase_shift, satellite_ids) in obs_entries.items():
            line = f"{system_code: <1}{obs_code: <3}{phase_shift:8.1f}"
            if satellite_ids is not None:
                satellite_ids_str = " ".join(satellite_ids)
                line += f"{len(satellite_ids_str):2}{satellite_ids_str}"
            lines.append(line)
    return lines


def parse_glonass_slot_frequencies(lines: Iterable[str]) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    lines = iter(lines)
    line = next(lines, None)
    if line is None:
        return entries
    num_satellites = int(line[:4])
    if num_satellites < 1:
        return entries
    while len(entries) < num_satellites and line is not None:
        i = 4
        while i + 7 <= 60:
            sat_id = line[i : i + 3].strip().replace(" ", "0")
            if not sat_id:
                raise ValueError(
                    f"Satellite ID not present for `GLONASS SLOT / FRQ #` entry; {line[i:i + 3]}"
                )
            freq_num = int(line[i + 4 : i + 6])
            entries.append((sat_id, freq_num))
            if len(entries) >= num_satellites:
                break
            i += 7
        if len(entries) >= num_satellites:
            break
        line = next(lines, None)
    return entries


def format_glonass_slot_frequencies(entries: List[Tuple[int, int]]) -> List[str]:
    lines = []
    num_satellites = len(entries)
    line = f"{num_satellites:4}"
    entry_index = 0
    while entry_index < num_satellites:
        prn, freq_num = entries[entry_index]
        line += f"R{prn:0<2} {freq_num:2} "
        if len(line) + 7 > 60:
            lines.append(line)
            line = " " * 4
        entry_index += 1
    return lines


def parse_glonass_phase_bias_corrections(line: str) -> Dict[str, float]:
    entries: Dict[str, float] = {}
    if line[:60].strip():
        return entries
    for i in range(4):
        obs_code = line[i * 13 + 1 : i * 13 + 3].strip()
        phase_bias_str = line[i * 13 + 4 : i * 13 + 13].strip()
        phase_bias = parse_float_or_nan(phase_bias_str)
        entries[obs_code] = phase_bias
    return entries


def format_glonass_phase_bias_corrections(entries: Dict[str, float]) -> str:
    return "".join(
        f" {obs_code: <3} {phase_bias:8.3f}" for obs_code, phase_bias in entries.items()
    )


# TODO this is wrong / unfinished
def parse_leap_seconds_metadata(line: str) -> LeapSecondMetadata:
    current_leap_seconds = int(line[:6])
    next_or_past_leap_seconds_str = line[6:12].strip()
    next_or_past_leap_seconds = (
        int(next_or_past_leap_seconds_str) if next_or_past_leap_seconds_str else None
    )
    # past_leap_seconds_str = line[12:18].strip()
    # past_leap_seconds = int(past_leap_seconds_str) if past_leap_seconds_str else None
    # effective_week_number = int(line[18:24])
    # effective_day_number = int(line[24:30])
    # time_system = line[30:32].strip()
    return LeapSecondMetadata(current_leap_seconds, None, None, None, None, None)


def format_leap_seconds_metadata(metadata: LeapSecondMetadata) -> str:
    future_or_past_leap_seconds = 0
    if metadata.next_leap_seconds is not None:
        future_or_past_leap_seconds = metadata.next_leap_seconds
    elif metadata.past_leap_seconds is not None:
        future_or_past_leap_seconds = metadata.past_leap_seconds
    return f"{metadata.current_leap_seconds: >6}{future_or_past_leap_seconds: >6}{'': <6}{metadata.effective_week_number: >6}{metadata.effective_day_number: >6}{metadata.time_system: <3}"


def parse_number_of_satellites(line: str) -> int:
    return int(line[:6].strip())


def format_number_of_satellites(number_of_satellites: int) -> str:
    return f"{number_of_satellites: >6}"


def parse_number_of_obs(
    line: str, sys_obs_types_entries: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    entries: Dict[str, Dict[str, int]] = {}
    # for system_code, obs_codes in sys_obs_types_entries.items():
    #     entries[system_code] = {obs_code: 0 for obs_code in obs_codes}
    return entries


def format_number_of_obs(
    entries: Dict[str, Dict[str, int]], sys_obs_types: Dict[str, List[str]]
) -> List[str]:
    lines = []
    # for system_code, obs_entries in entries.items():
    raise NotImplementedError()


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
LABEL_ANTENNA_DELTA_XYZ = "ANTENNA: DELTA X/Y/Z"
LABEL_ANTENNA_PHASECENTER = "ANTENNA: PHASECENTER"
LABEL_ANTENNA_BORESIGHT = "ANTENNA: BORESIGHT"
LABEL_ANTENNA_ZERODIR_AZI = "ANTENNA: ZERODIR A/Z"
LABEL_ANTENNA_ZERODIR_XYZ = "ANTENNA: ZERODIR X/Y/Z"
LABEL_CENTER_OF_MASS_XYZ = "CENTER OF MASS: XYZ"
LABEL_SYS_NUM_OBS = "SYS / # / OBS TYPES"
LABEL_SIGNAL_STRENGTH_UNIT = "SIGNAL STRENGTH UNIT"
LABEL_INTERVAL = "INTERVAL"
LABEL_TIME_OF_FIRST_OBS = "TIME OF FIRST OBS"
LABEL_TIME_OF_LAST_OBS = "TIME OF LAST OBS"
LABEL_RCV_CLOCK_OFFS_APPL = "RCV CLOCK OFFS APPL"
LABEL_SYS_DCBS_APPLIED = "SYS / DCBS APPLIED"
LABEL_SYS_PCVS_APPLIED = "SYS / PCVS APPLIED"
LABEL_SYS_SCALE_FACTOR = "SYS / SCALE FACTOR"
LABEL_SYS_PHASE_SHIFT = "SYS / PHASE SHIFT"
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
    date: datetime | None = None
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
    approximate_position: List[float] | None = None
    antenna_offset: List[float] | None = None
    antenna_offset_frame: AntennaOffsetFrame | None = None
    antenna_phase_center_offsets: Optional[List[AntennaPhaseCenterOffsetEntry]] = None
    antenna_boresight: Optional[List[float]] = None
    antenna_zerodir_azi: Optional[float] = None
    antenna_zerodir_xyz: Optional[List[float]] = None
    vehicle_center_of_mass_xyz: Optional[List[float]] = None
    system_obs_types: Dict[str, List[str]] | None = None
    signal_strength_unit: Optional[str] = None
    interval: Optional[float] = None
    time_system: str | None = None
    time_of_first_obs: Optional[datetime] = None
    time_of_last_obs: Optional[datetime] = None
    is_receiver_clock_offset_applied: Optional[bool] = None
    applied_dcbs: Optional[Dict[str, float]] = None  # system_code -> dcb
    applied_pcvs: Optional[Dict[str, float]] = None  # system_code -> pcv
    applied_scale_factors: Optional[Dict[str, Dict[str, float]]] = (
        None  # system_code -> obs_code -> scale_factor
    )
    applied_phase_shifts: (
        Dict[str, Dict[str, Tuple[float, Optional[List[str]]]]] | None
    ) = None  # system_code -> obs_code -> (phase_shift, [satellite_ids])
    glonass_slot_frequencies: List[Tuple[int, int]] | None = None
    glonass_phase_bias_corrections: Dict[str, float] | None = None
    leap_second_metadata: Optional[LeapSecondMetadata] = None
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
    antenna_phase_center_lines: List[str] = []
    sys_num_obs_lines: List[str] = []
    sys_dcbs_applied_lines: List[str] = []
    sys_pcvs_applied_lines: List[str] = []
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
        elif line_label == LABEL_MARKER_TYPE:
            marker_type = parse_marker_type(line)
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
            antenna_offset_frame = AntennaOffsetFrame.ENU
        elif line_label == LABEL_ANTENNA_DELTA_XYZ:
            antenna_offset = parse_antenna_offset(line)
            antenna_offset_frame = AntennaOffsetFrame.XYZ
        elif line_label == LABEL_ANTENNA_PHASECENTER:
            antenna_phase_center_lines.append(line)
        elif line_label == LABEL_ANTENNA_BORESIGHT:
            antenna_boresight = parse_antenna_boresight(line)
        elif line_label == LABEL_ANTENNA_ZERODIR_AZI:
            antenna_zerodir_azi = parse_antenna_zerodir_azi(line)
        elif line_label == LABEL_ANTENNA_ZERODIR_XYZ:
            antenna_zerodir_xyz = parse_antenna_zerodir_xyz(line)
        elif line_label == LABEL_CENTER_OF_MASS_XYZ:
            vehicle_center_of_mass_xyz = parse_center_of_mass_xyz(line)
        elif line_label == LABEL_SYS_NUM_OBS:
            sys_num_obs_lines.append(line)
        elif line_label == LABEL_SIGNAL_STRENGTH_UNIT:
            signal_strength_unit = parse_signal_strength_unit(line)
        elif line_label == LABEL_INTERVAL:
            interval = parse_interval(line)
        elif line_label == LABEL_TIME_OF_FIRST_OBS:
            time_of_first_obs, time_system = parse_time_of_first_obs(line)
        elif line_label == LABEL_TIME_OF_LAST_OBS:
            time_of_last_obs, _ = parse_time_of_last_obs(line)
        elif line_label == LABEL_RCV_CLOCK_OFFS_APPL:
            is_receiver_clock_offset_applied = parse_receiver_clock_offset_applied(line)
        elif line_label == LABEL_SYS_DCBS_APPLIED:
            sys_dcbs_applied_lines.append(line)
        elif line_label == LABEL_SYS_PCVS_APPLIED:
            sys_pcvs_applied_lines.append(line)
        elif line_label == LABEL_SYS_SCALE_FACTOR:
            sys_phase_shift_lines.append(line)
        elif line_label == LABEL_SYS_PHASE_SHIFT:
            sys_phase_shift_lines.append(line)
        elif line_label == LABEL_GLONASS_SLOT_FRQ:
            glonass_slot_freq_lines.append(line)
        elif line_label == LABEL_GLONASS_COD_PHS_BIS:
            glonass_phase_bias_corrections = parse_glonass_phase_bias_corrections(line)
        elif line_label == LABEL_LEAP_SECONDS:
            leap_second_metadata = parse_leap_seconds_metadata(line)
        elif line_label == LABEL_NUM_SATELLITES:
            number_of_satellites = parse_number_of_satellites(line)
        elif line_label == LABEL_PRN_NUM_OBS:
            prn_num_obs_lines.append(line)
        else:
            if strict:
                raise ValueError(f"Unknown header line label: {line_label}")
            else:
                # logging.warning(f"Unknown header line label: {line_label}")
                pass

    if antenna_phase_center_lines:
        antenna_phase_center_offsets = parse_antenna_phase_center_offsets(
            antenna_phase_center_lines
        )
    if sys_num_obs_lines:
        system_obs_types = parse_system_obs_types(sys_num_obs_lines)
    if sys_dcbs_applied_lines:
        applied_dcbs = parse_applied_dcbs(sys_dcbs_applied_lines)
    if sys_pcvs_applied_lines:
        applied_pcvs = parse_applied_pcvs(sys_pcvs_applied_lines)
    if sys_phase_shift_lines:
        applied_phase_shifts = parse_applied_phase_shifts(sys_phase_shift_lines)
    if glonass_slot_freq_lines:
        glonass_slot_frequencies = parse_glonass_slot_frequencies(
            glonass_slot_freq_lines
        )
    if prn_num_obs_lines:
        number_of_obs = parse_number_of_obs(prn_num_obs_lines, system_obs_types)

    return Header(
        rinex_version=rinex_version,
        file_type=file_type,
        system_code=system_code,
        program_name=program_name,
        run_by=run_by,
        date=date,
        comment_lines=comment_lines,
        marker_name=marker_name,
        marker_number=marker_number,
        marker_type=marker_type,
        observer=observer,
        agency=agency,
        receiver_number=receiver_number,
        receiver_type=receiver_type,
        receiver_version=receiver_version,
        antenna_number=antenna_number,
        antenna_type=antenna_type,
        approximate_position=approximate_position,
        antenna_offset=antenna_offset,
        antenna_offset_frame=antenna_offset_frame,
        antenna_phase_center_offsets=antenna_phase_center_offsets,
        antenna_boresight=antenna_boresight,
        antenna_zerodir_azi=antenna_zerodir_azi,
        antenna_zerodir_xyz=antenna_zerodir_xyz,
        vehicle_center_of_mass_xyz=vehicle_center_of_mass_xyz,
        system_obs_types=system_obs_types,
        signal_strength_unit=signal_strength_unit,
        interval=interval,
        time_system=time_system,
        time_of_first_obs=time_of_first_obs,
        time_of_last_obs=time_of_last_obs,
        is_receiver_clock_offset_applied=is_receiver_clock_offset_applied,
        applied_dcbs=applied_dcbs,
        applied_pcvs=applied_pcvs,
        applied_scale_factors=applied_scale_factors,
        applied_phase_shifts=applied_phase_shifts,
        glonass_slot_frequencies=glonass_slot_frequencies,
        glonass_phase_bias_corrections=glonass_phase_bias_corrections,
        leap_second_metadata=leap_second_metadata,
        number_of_satellites=number_of_satellites,
        number_of_obs=number_of_obs,
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
    lines.append(f"{format_marker_type(header.marker_type): <60}{LABEL_MARKER_TYPE}")
    lines.append(
        f"{format_observer_agency(header.observer, header.agency): <60}{LABEL_OBSERVER_AGENCY}"
    )
    lines.append(
        f"{format_receiver_type_version(header.receiver_number, header.receiver_type, header.receiver_version): <60}{LABEL_REC_TYPE_VERS}"
    )
    lines.append(
        f"{format_antenna_type(header.antenna_number, header.antenna_type): <60}{LABEL_ANT_TYPE}"
    )
    if header.approximate_position:
        lines.append(
            f"{format_approx_position_xyz(*header.approximate_position): <60}{LABEL_APPROX_POSITION_XYZ}"
        )
    if header.antenna_offset_frame == AntennaOffsetFrame.ENU:
        lines.append(
            f"{format_antenna_offset(*header.antenna_offset): <60}{LABEL_ANTENNA_DELTA_HEN}"
        )
    elif header.antenna_offset_frame == AntennaOffsetFrame.XYZ:
        lines.append(
            f"{format_antenna_offset(*header.antenna_offset): <60}{LABEL_ANTENNA_DELTA_XYZ}"
        )
    if header.antenna_phase_center_offsets:
        lines.extend(
            format_antenna_phase_center_offsets(header.antenna_phase_center_offsets)
        )
    if header.antenna_boresight:
        lines.append(
            f"{format_antenna_boresight(*header.antenna_boresight): <60}{LABEL_ANTENNA_BORESIGHT}"
        )
    if header.antenna_zerodir_azi:
        lines.append(
            f"{format_antenna_zerodir_azi(header.antenna_zerodir_azi): <60}{LABEL_ANTENNA_ZERODIR_AZI}"
        )
    if header.antenna_zerodir_xyz:
        lines.append(
            f"{format_antenna_zerodir_xyz(*header.antenna_zerodir_xyz): <60}{LABEL_ANTENNA_ZERODIR_XYZ}"
        )
    if header.vehicle_center_of_mass_xyz:
        lines.append(
            f"{format_center_of_mass_xyz(*header.vehicle_center_of_mass_xyz): <60}{LABEL_CENTER_OF_MASS_XYZ}"
        )
    lines.extend(format_system_obs_types(header.system_obs_types))
    if header.signal_strength_unit:
        lines.append(
            f"{format_signal_strength_unit(header.signal_strength_unit): <60}{LABEL_SIGNAL_STRENGTH_UNIT}"
        )
    if header.interval:
        lines.append(f"{format_interval(header.interval): <60}{LABEL_INTERVAL}")
    lines.append(
        f"{format_time_of_first_obs(header.time_of_first_obs, header.time_system): <60}{LABEL_TIME_OF_FIRST_OBS}"
    )
    if header.time_of_last_obs:
        lines.append(
            f"{format_time_of_last_obs(*header.time_of_last_obs): <60}{LABEL_TIME_OF_LAST_OBS}"
        )
    if header.is_receiver_clock_offset_applied is not None:
        lines.append(
            f"{format_receiver_clock_offset_applied(header.is_receiver_clock_offset_applied): <60}{LABEL_RCV_CLOCK_OFFS_APPL}"
        )
    if header.applied_dcbs:
        lines.extend(format_applied_dcbs(header.applied_dcbs))
    if header.applied_pcvs:
        lines.extend(format_applied_pcvs(header.applied_pcvs))
    if header.applied_scale_factors:
        lines.extend(format_sys_scale_factors(header.applied_scale_factors))
    if header.applied_phase_shifts:
        lines.extend(format_applied_phase_shifts(header.applied_phase_shifts))
    if header.glonass_slot_frequencies:
        lines.extend(format_glonass_slot_frequencies(header.glonass_slot_frequencies))
    if header.glonass_phase_bias_corrections:
        lines.append(
            f"{format_glonass_phase_bias_corrections(header.glonass_phase_bias_corrections): <60}{LABEL_GLONASS_COD_PHS_BIS}"
        )
    if header.leap_second_metadata:
        lines.append(
            f"{format_leap_seconds_metadata(header.leap_second_metadata): <60}{LABEL_LEAP_SECONDS}"
        )
    if header.number_of_satellites:
        lines.append(
            f"{format_number_of_satellites(header.number_of_satellites): <60}{LABEL_NUM_SATELLITES}"
        )
    if header.number_of_obs:
        lines.extend(
            format_number_of_obs(header.number_of_obs, header.system_obs_types)
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
    transmitters: Dict[str, List[float | Tuple[float, int, int]]]  # transmitter_id -> (obs_values, ...)


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



def parse_transmitter_observations(
    line: str,
    system_obs_types: Dict[str, List[str]],
    parse_ssi: bool,
    parse_lli: bool,
    strict: bool = True,
) -> Tuple[str, List[float | Tuple[float, int, int]]]:
    # if `parse_ssi` or `parse_lli` is True, carrier observations ("L" obs. code) are
    # expected to have tuples (Tuple[float, int, int]) for their corresponding
    # observation value, denoting the carrier observation value and corresponding
    # SSI and LLI flags

    # added for robustness; some really dumb writers use space instead of zero in sat ids, e.g. 'G 1'
    sat_id = line[0:3].replace(" ", "0")
    system_letter = sat_id[0]
    obs_codes = system_obs_types[system_letter]

    i0 = 3
    line_length = len(line)
    obs_vals = []

    # We need to append values for every obs code, even if they are missing
    # If an obs code is missing, we append a NaN value
    # A missing obs code is usually because the satellite does not transmit that particular signal (e.g. GPS and L5)
    for i, obs_code in enumerate(obs_codes):
        if i0 >= line_length:
            # # We don't do continuation lines (for now), so raise error if we run out of characters
            # if strict:
            #     raise ValueError(f"Error parsing observation value for {sat_id} at index {i0}: not enough characters in line")
            # logging.warning(f"When parsing observation value for {sat_id} at index {i0}: not enough characters in line")
            
            # Actually nevermind -- lines don't have to fill whitespace to complete the line
            # So just append NaN and continue
            if obs_code[0] == "L":
                obs_val = (float("nan"), DEFAULT_SSI_VALUE, DEFAULT_LLI_VALUE)
            else:
                obs_val = float("nan")
            obs_vals.append(obs_val)
            continue

        i1 = min(i0 + 14, line_length)
        obs_val_str = line[i0:i1].strip()
        obs_val = float("nan")
        if obs_val_str:
            try:
                obs_val = float(obs_val_str)
            except ValueError:
                if strict:
                    raise ValueError(f"Error parsing observation value for {sat_id} at index {i0}: {obs_val_str}")
                logging.warning(f"When parsing observation value for {sat_id} at index {i0}: {obs_val_str}")
                # assert(False)
            if obs_code[0] == "L":
                ssi = DEFAULT_SSI_VALUE
                lli = DEFAULT_LLI_VALUE
                if parse_ssi:
                    if i0 + 14 >= line_length:
                        if strict:
                            raise ValueError(f"Error parsing SSI flag for {sat_id} at index {i}: not enough characters in line")
                        else:
                            logging.warning(f"When parsing SSI flag for {sat_id} at index {i}: not enough characters in line")
                        pass
                    else:
                        ssi_str = line[i0 + 14]
                        if ssi_str != " ":
                            ssi = int(ssi_str)
                if parse_lli:
                    if i0 + 15 >= line_length:
                        if strict:
                            raise ValueError(f"Error parsing LLI flag for {sat_id} at index {i}: not enough characters in line")
                        else:
                            logging.warning(f"When parsing LLI flag for {sat_id} at index {i}: not enough characters in line")
                        pass
                    else:
                        lli_str = line[i0 + 15]
                        if lli_str != " ":
                            lli = int(lli_str)
                obs_val = (obs_val, ssi, lli)
        else:
            if obs_code[0] == "L":
                obs_val = (float("nan"), DEFAULT_SSI_VALUE, DEFAULT_LLI_VALUE)
        obs_vals.append(obs_val)
        i0 += 16

    return sat_id, obs_vals


def parse_observations(
    input: io.TextIOWrapper,
    system_obs_types: Dict[str, List[str]],
    parse_ssi: bool,
    parse_lli: bool,
    strict: bool = True,
) -> List[EpochRecord]:

    records: List[EpochRecord] = []

    current_record: Optional[EpochRecord] = None
    num_sats: int = 0

    for line in input:
        if line.startswith(">"):
            if current_record is not None:
                if strict:
                    raise ValueError(
                        "Unexpected start of new epoch while parsing epoch"
                    )
                records.append(current_record)
            epoch_header = parse_epoch_header(line, strict)
            if epoch_header is None:
                if strict:
                    raise ValueError("Error parsing epoch header")
                continue
            epoch, epoch_flag, num_sats = epoch_header
            current_record = EpochRecord(epoch, epoch_flag, {})
        else:
            # assume line contains transmitter observations
            if current_record is None:
                if strict:
                    raise ValueError("Observations found before epoch header")
                continue
            sat_id, obs_vals = parse_transmitter_observations(
                line, system_obs_types, parse_ssi, parse_lli, strict
            )
            current_record.transmitters[sat_id] = obs_vals
            num_sats -= 1
            if num_sats == 0:
                records.append(current_record)
                current_record = None

    if current_record is not None:
        records.append(current_record)

    return records


DEFAULT_SSI_VALUE = 0
DEFAULT_LLI_VALUE = 0


def format_observations(
    records: List[EpochRecord],
    system_obs_types: Dict[str, List[str]],
    format_ssi: bool = True,
    format_lli: bool = True,
    strict: bool = True,
) -> List[str]:

    lines = []
    for record in records:
        lines.append(format_epoch_header(record.epoch, record.epoch_flag, len(record.transmitters)))
        for sat_id, obs_vals in record.transmitters.items():
            system_letter = sat_id[0]
            obs_codes = system_obs_types[system_letter]
            sat_obs_str = f"{sat_id}"
            for i, obs_val in enumerate(obs_vals):
                obs_code = obs_codes[i]
                if obs_code[0] == "L":
                    # if format_ssi or format_lli are true, we enforce length-3 obs values
                    if not isinstance(obs_val, tuple) or len(obs_val) != 3:
                        raise ValueError(
                            f"Expected length-3 observation value (since `format_ssi` and/or `format_lli` is true) for {sat_id} at index {i}"
                        )
                    obs_val, ssi, lli = obs_val
                    ssi = int(ssi) if ssi is not None else DEFAULT_SSI_VALUE
                    lli = int(lli) if lli is not None else DEFAULT_LLI_VALUE
                    if format_ssi and format_lli:
                        sat_obs_str += f"{obs_val:14.3f}{ssi:01}{lli:01}"
                    elif format_ssi:
                        sat_obs_str += f"{obs_val:14.3f}{ssi:01} "
                    elif format_lli:
                        sat_obs_str += f"{obs_val:14.3f} {lli:01}"
                    else:
                        sat_obs_str += f"{obs_val:14.3f}  "
                else:
                    sat_obs_str += f"{obs_val:14.3f}  "
            lines.append(sat_obs_str)
    return lines



@dataclass
class RINEX_ObservationArrays:
    # sat_id -> obs_code -> [obs_values]
    epochs: np.ndarray
    records: Dict[str, Dict[str, np.ndarray]]



def parse_observations_into_observation_arrays(
    input: io.TextIOWrapper,
    system_obs_types: Dict[str, List[str]],
    observation_arrays: RINEX_ObservationArrays,
    parse_ssi: bool,
    parse_lli: bool,
    strict: bool = True,
) -> None:

    currently_parsing_record: bool = False
    num_sats: int = 0

    # keep track of obs index for each sat
    current_sat_index = {}
    max_num_epochs: int = -1
    CHECK_OBS_ARR_LENGTH: bool = True
    for sat_id in observation_arrays.records.keys():
        current_sat_index[sat_id] = 0
        if max_num_epochs < 0:
            for arr in observation_arrays.records[sat_id].values():
                max_num_epochs = max(max_num_epochs, len(arr))
        if CHECK_OBS_ARR_LENGTH:
            for arr in observation_arrays.records[sat_id].values():
                assert(max_num_epochs == len(arr))
    if max_num_epochs < 0:
        assert(False)

    current_epoch_index = 0

    for line in input:
        if line.startswith(">"):
            if currently_parsing_record:
                if strict:
                    raise ValueError(
                        "Unexpected start of new epoch while parsing epoch"
                    )
                current_epoch_index += 1
            epoch_header = parse_epoch_header(line, strict)
            if epoch_header is None:
                if strict:
                    raise ValueError("Error parsing epoch header")
                continue
            epoch, epoch_flag, num_sats = epoch_header
            observation_arrays.epochs[current_epoch_index] = epoch
            currently_parsing_record = True
        else:
            # assume line contains transmitter observations
            if not currently_parsing_record:
                if strict:
                    raise ValueError("Observations found before epoch header")
                continue
            sat_id, obs_vals = parse_transmitter_observations(
                line, system_obs_types, parse_ssi, parse_lli, strict
            )
            system_letter = sat_id[0]
            obs_types = system_obs_types[system_letter]
            sat_index = current_sat_index[sat_id]
            for obs_id, obs_val in zip(obs_types, obs_vals):
                observation_arrays.records[sat_id][obs_id][sat_index] = obs_val
            current_sat_index[sat_id] -= 1

            num_sats -= 1
            if num_sats == 0:
                current_epoch_index += 1
                currently_parsing_record = False

    return


class Dataset:

    def __init__(self, include_ssi: bool = True, include_lli: bool = True) -> None:
        self.header: Optional[Header] = None
        self.epoch_records: Optional[List[EpochRecord]] = None

        # These should not be overwritten once set, since they determine the content of the observations
        self._include_ssi = include_ssi
        self._include_lli = include_lli

        self._filepaths: List[str] = []

    def load(self, io: io.TextIOWrapper, strict: bool = True) -> "Dataset":
        # todo: add options for epochs, etc.
        header = parse_header(io, strict)
        if self.header is None:
            self.header = header
        else:
            # TODO check for header compatibility when merging datasets
            # check phase offsets, applied DCBs, etc
            if strict:
                assert(False)
            self.header = header
        epoch_records = parse_observations(
            io, self.header.system_obs_types, self._include_ssi, self._include_lli, strict
        )
        if self.epoch_records is None:
            self.epoch_records = epoch_records
        else:
            self.epoch_records += epoch_records
        return self

    def load_files(self, filepaths: List[str], strict: bool = True) -> "Dataset":
        for filepath in filepaths:
            with open(filepath, "r") as f:
                # print(filepath)
                self.load(f, strict)
        self._filepaths += filepaths
        return self

    def save(self, io: io.TextIOWrapper) -> None:
        if self.header is None or self.epoch_records is None:
            raise ValueError("Cannot save dataset without header and observations")
        lines = format_header(self.header)
        lines.extend(format_observations(self.epoch_records, self.header.system_obs_types))
        # writelines doesn't add newlines, so we need to add them ourselves
        lines = [line + "\n" for line in lines]
        io.writelines(lines)
    
    def get_observation_arrays(
        self,
        strip_all_nan: bool = True,
        sort_and_merge_epoch_records: bool = False
    ) -> RINEX_ObservationArrays:
        """
        Get observations as numpy arrays.
        
        Returns:
            obs_epochs: numpy array of observation epochs (GPS seconds)
            obs_arrays: dict of satellite ID to dict of observation code to numpy array of observation values
        """

        obs_epochs: List[float] = []
        obs_arrays: Dict[str, Dict[str, List[float]]] = {}
        # file_tracer: Dict[str, Dict[str, List[float]]] = {}

        GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

        if self.header is None:
            raise ValueError("Cannot get observation arrays without header")
        if self.epoch_records is None:
            raise ValueError("Cannot get observation arrays without observations")
        system_obs_types = self.header.system_obs_types

        # Iterate through each record and append values to the obs_arrays dict
        observations = self.epoch_records
        if sort_and_merge_epoch_records:
            observations_dict: Dict[datetime, EpochRecord] = {}
            # sorted_epoch_records = sorted(self.observations, lambda obs: obs.epoch)
            for obs in self.epoch_records:
                if obs.epoch not in observations_dict:
                    observations_dict[obs.epoch] = obs
                else:
                    observations_dict[obs.epoch].transmitters.update(obs.transmitters)
                    observations_dict[obs.epoch].epoch_flag |= obs.epoch_flag
            observations = sorted(list(observations_dict.values()),
                                  key = lambda obs: obs.epoch)
        
        for record_index, record in enumerate(observations):
            obs_epochs.append((record.epoch - GPS_EPOCH).total_seconds())
            for sat_id, obs_vals in record.transmitters.items():
                if sat_id[0] not in system_obs_types:
                    continue
                obs_codes = system_obs_types[sat_id[0]]
                if sat_id not in obs_arrays:
                    # When instantiating the obs_arrays entry for a new satellite, we need to create a list for each obs code, as well as for SSI and LLI flags when included
                    obs_arrays[sat_id] = {obs_code: [] for obs_code in obs_codes}
                    if self._include_ssi:
                        for obs_code in obs_codes:
                            if obs_code[0] == "L":
                                obs_arrays[sat_id][obs_code + "_SSI"] = []
                    if self._include_lli:
                        for obs_code in obs_codes:
                            if obs_code[0] == "L":
                                obs_arrays[sat_id][obs_code + "_LLI"] = []
                    obs_arrays[sat_id]["index"] = []
                obs_arrays[sat_id]["index"].append(record_index)
                for obs_index, obs_code in enumerate(obs_codes):
                    if obs_code[0] == "L":
                        assert(isinstance(obs_vals[obs_index], tuple))
                        assert(len(obs_vals[obs_index]) == 3)  # type: ignore
                        obs_val, ssi, lli = obs_vals[obs_index]  # type: ignore
                        obs_arrays[sat_id][obs_code].append(obs_val)
                        if self._include_ssi:
                            obs_arrays[sat_id][obs_code + "_SSI"].append(ssi)
                        if self._include_lli:
                            obs_arrays[sat_id][obs_code + "_LLI"].append(lli)
                    else:
                        obs_arrays[sat_id][obs_code].append(obs_vals[obs_index])  # type: ignore

        # Convert lists to numpy arrays
        obs_epochs_arr = np.array(obs_epochs)
        for sat_id, obs_dict in obs_arrays.items():
            for obs_code, obs_vals in obs_dict.items():
                val_arr = np.array(obs_vals)
                obs_dict[obs_code] = val_arr
        
        for sat_id, obs_dict in obs_arrays.items():
            for obs_code in list(obs_dict.keys()):
                if np.all(np.isnan(obs_dict[obs_code])):
                    del obs_dict[obs_code]
        
        return RINEX_ObservationArrays(obs_epochs_arr, obs_arrays)

        # create dict of sat ID
        # each has obs to value lists for each obs code
        # if all obs are nan, and prune, ignore that obs code for that satellite
        # keep track of epoch index; also have index list for each satellite
        # at end, convert to numpy arrays





def create_default_obs_header() -> Header:

    rinex_version: str = "3"
    file_type: str = "O"
    system_code: str = ""
    program_name: str = ""
    run_by: str = ""
    date: datetime = datetime.now()
    marker_name: str = ""
    marker_number: str = ""
    marker_type: str = ""
    observer: str = ""
    agency: str = ""
    receiver_number: str = ""
    receiver_type: str = ""
    receiver_version: str = ""
    antenna_number: str = ""
    antenna_type: str = ""
    approximate_position: List[float] = []
    antenna_offset: List[float] = []
    antenna_offset_frame: AntennaOffsetFrame = AntennaOffsetFrame.UNKNOWN
    antenna_phase_center_offsets: Optional[List[AntennaPhaseCenterOffsetEntry]] = None
    antenna_boresight: Optional[List[float]] = None
    antenna_zerodir_azi: Optional[float] = None
    antenna_zerodir_xyz: Optional[List[float]] = None
    vehicle_center_of_mass_xyz: Optional[List[float]] = None

    system_obs_types: Dict[str, List[str]] | None = {}

    signal_strength_unit: Optional[str] = None
    interval: Optional[float] = None

    time_system: str = "GPS"

    time_of_first_obs: Optional[datetime] = None
    time_of_last_obs: Optional[datetime] = None
    is_receiver_clock_offset_applied: Optional[bool] = None
    applied_dcbs: Optional[Dict[str, float]] = None  # system_code -> dcb
    applied_pcvs: Optional[Dict[str, float]] = None  # system_code -> pcv
    applied_scale_factors: Optional[Dict[str, Dict[str, float]]] = (
        None  # system_code -> obs_code -> scale_factor
    )
    applied_phase_shifts: (
        Dict[str, Dict[str, Tuple[float, Optional[List[str]]]]] | None
    ) = {}  # system_code -> obs_code -> (phase_shift, [satellite_ids])
    glonass_slot_frequencies: List[Tuple[int, int]] = []
    glonass_phase_bias_corrections: Dict[str, float] = {}
    leap_second_metadata: Optional[LeapSecondMetadata] = None
    number_of_satellites: Optional[int] = None
    number_of_obs: Optional[Dict[str, List[int]]] = (
        None  # satellite_id -> [number_of_obs]
    )

    comment_lines: List[str] = []

    return Header(
        rinex_version=rinex_version,
        file_type=file_type,
        system_code=system_code,
        program_name=program_name,
        run_by=run_by,
        date=date,
        comment_lines=comment_lines,
        marker_name=marker_name,
        marker_number=marker_number,
        marker_type=marker_type,
        observer=observer,
        agency=agency,
        receiver_number=receiver_number,
        receiver_type=receiver_type,
        receiver_version=receiver_version,
        antenna_number=antenna_number,
        antenna_type=antenna_type,
        approximate_position=approximate_position,
        antenna_offset=antenna_offset,
        antenna_offset_frame=antenna_offset_frame,
        antenna_phase_center_offsets=antenna_phase_center_offsets,
        antenna_boresight=antenna_boresight,
        antenna_zerodir_azi=antenna_zerodir_azi,
        antenna_zerodir_xyz=antenna_zerodir_xyz,
        vehicle_center_of_mass_xyz=vehicle_center_of_mass_xyz,
        system_obs_types=system_obs_types,
        signal_strength_unit=signal_strength_unit,
        interval=interval,
        time_system=time_system,
        time_of_first_obs=time_of_first_obs,
        time_of_last_obs=time_of_last_obs,
        is_receiver_clock_offset_applied=is_receiver_clock_offset_applied,
        applied_dcbs=applied_dcbs,
        applied_pcvs=applied_pcvs,
        applied_scale_factors=applied_scale_factors,
        applied_phase_shifts=applied_phase_shifts,
        glonass_slot_frequencies=glonass_slot_frequencies,
        glonass_phase_bias_corrections=glonass_phase_bias_corrections,
        leap_second_metadata=leap_second_metadata,
        number_of_satellites=number_of_satellites,
        number_of_obs=number_of_obs,
    )




def parse_rinex_file_epochs(f: io.TextIOWrapper) -> List[Tuple[datetime, int, int]]:
    """
    Parse RINEX file epochs from a file object.
    This is useful for quickly determining the epochs in the file
    """
    epochs = []
    for line in f:
        if line.startswith("> "):
            year = int(line[2:6])
            month = int(line[7:9])
            day = int(line[9:11])
            hour = int(line[12:14])
            minute = int(line[15:17])
            second = float(line[20:29])
            flag_1 = int(line[30:32])
            flag_2 = int(line[33:35])
            second, microsecond = divmod(second, 1)
            microsecond = int(microsecond * 1e6)
            epoch = datetime(year + 2000, month, day, hour, minute, int(second), microsecond)
            epochs.append((epoch, flag_1, flag_2))
    return epochs