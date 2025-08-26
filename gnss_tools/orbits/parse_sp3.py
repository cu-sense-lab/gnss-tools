from functools import lru_cache
from datetime import datetime, timezone, timedelta
import io
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

GPS_EPOCH = datetime(
    year=1980, month=1, day=6, hour=0, minute=0, second=0
)
ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)


@dataclass
class SP3Header:
    version: str
    position_velocity_flag: str
    start_time: float
    number_of_epochs: int
    data_used: str
    coordinate_sys: str
    orbit_type: str
    agency: str
    gps_week: int
    seconds_of_week: float
    epoch_interval: float
    mod_jul_day_start: int
    fractional_day: float

def parse_header(
        input: io.TextIOWrapper,
        strict: bool = True
    ) -> SP3Header:
    """
    Parse SP3 header from file.
    """
    line = input.readline()
    version = line[1:2]
    position_velocity_flag = line[2]
    start_time = parse_time_epoch(line[3:31])
    number_of_epochs = int(line[32:39])
    data_used = line[40:45].strip()
    coordinate_sys = line[46:51]
    orbit_type = line[52:55]
    agency = line[56:60].strip()

    line = input.readline()
    gps_week = int(line[3:7])
    seconds_of_week = float(line[8:23])
    epoch_interval = float(line[24:38])
    mod_jul_day_start = int(line[39:44])
    fractional_day = float(line[45:60])

    return SP3Header(
        version,
        position_velocity_flag,
        start_time,
        number_of_epochs,
        data_used,
        coordinate_sys,
        orbit_type,
        agency,
        gps_week,
        seconds_of_week,
        epoch_interval,
        mod_jul_day_start,
        fractional_day,
    )


def format_header(
    header: SP3Header
) -> List[str]:
    """
    Format SP3 Header into 2 header lines
    """
    
    dt = GPS_EPOCH + timedelta(seconds=header.start_time)
    # start_time_str = start_time_dt.strftime("%Y %m %d %H %M %S.%f")
    dt_seconds = dt.second + dt.microsecond * 1e-6
    line1 = (
        f"#{header.version:01}{header.position_velocity_flag:01}"
        f"{dt.year:04} {dt.month: >2} {dt.day: >2} {dt.hour: >2} "
        f"{dt.minute: >2} {dt_seconds: >11.8f} {header.number_of_epochs:>7} "
        f"{header.data_used:<5} {header.coordinate_sys:<5} "
        f"{header.orbit_type:<3} {header.agency:<4}"
    )
    
    line2 = (
        f"## {header.gps_week:04} {header.seconds_of_week: >15.8f}"
        f"{header.epoch_interval: >15.8f} {header.mod_jul_day_start:>5}"
        f" {header.fractional_day:<15.13f}"
    )
    return [line1, line2]


def test_nan(x: float, nan_value=999999.999999, eps=1e-3) -> bool:
    """Tests if `x` is NaN according to SP3 spec, i.e. is within `eps` of `nan_value`"""
    return abs(x - nan_value) < eps


def parse_time_epoch(timestr: str) -> float:
    """
    Takes SP3 file epoch date string format and returns GPS seconds.
    Because datatime only tracks up to microsecond accuracy, we cannot use
    the last 2 digits in the seconds decimal.  We will throw an error if the
    last two digits are not 0.  Also, the times in SP3 files are given in GPS time, even
    thought the format is YYYY MM DD HH MM SS.  This means that if we subtract
    the GPS epoch using two UTC datetimes, we'll get the correct time in GPS
    seconds (note, datetime is not leap-second aware, which is why this works).
    """
    if int(timestr[26:]) != 0:
        raise Exception(
            "`datetime` cannot handle sub-microsecond precision, but epoch in file appears to specify this level of precision."
        )
    time = datetime.strptime(timestr[:26], "%Y %m %d %H %M %S.%f")
    return (time - GPS_EPOCH).total_seconds()  # GPS time


def format_time_epoch(gps_seconds: float) -> str:
    """
    Takes GPS seconds and returns SP3 file epoch date string format.
    """
    dt = GPS_EPOCH + timedelta(seconds=gps_seconds)
    timestr = (
        f"{dt.year:04} {dt.month: >2} {dt.day: >2} {dt.hour: >2} "
        f"{dt.minute: >2} {dt.second: >2}.{dt.microsecond:0>6}00"
    )
    return timestr


def parse_position_and_clock(line: str) -> Tuple[str, float, float, float, float]:
    """
    Returns <vehicle id>, <x-coordinate>, <y-coordinate>, <z-coordinate>, <clock>
    x, y, z coordinates are in units of km and clock offset is in units of microseconds
    """
    veh_id, x, y, z, c = (
        line[1:4],
        float(line[4:18]),
        float(line[18:32]),
        float(line[32:46]),
        float(line[46:60]),
    )
    x = np.nan if test_nan(x) else x * 1e3
    y = np.nan if test_nan(y) else y * 1e3
    z = np.nan if test_nan(z) else z * 1e3  # convert from km to m
    clock = np.nan if test_nan(c) else c
    return veh_id, x, y, z, clock


def parse_velocity_and_clock(line: str) -> Tuple[str, float, float, float, float]:
    """
    Returns <vehicle id>, <x-velocity>, <y-velocity>, <z-velocity>, <clock-rate-change>
    x, y, z velocities are in units of dm/s and clock rate is in units of s/s
    """
    return parse_position_and_clock(line)


def format_position_and_clock(vehicle_id: str, x: float, y: float, z: float, clock: float) -> str:
    """
    Takes vehicle id, x, y, z coordinates (in km), and clock offset (in microseconds),
    and returns the formatted string for SP3 file.
    """
    # convert from m to km
    x = x / 1000.0
    y = y / 1000.0
    z = z / 1000.0
    return f"{vehicle_id:>3}{x:>14.6f}{y:>14.6f}{z:>14.6f}{clock:>14.6f}"


@dataclass
class SP3Record:
    epoch: float
    p_entries: Dict[str, Optional[Tuple[float, float, float, float]]]
    v_entries: Dict[str, Optional[Tuple[float, float, float, float]]]

def parse_records(
        input: io.TextIOWrapper,
        parse_position: bool = True,
        parse_velocity: bool = False,
        strict: bool = True
    ) -> List[SP3Record]:
    """
    Parse SP3 records from iterable lines read from a file.
    """
    records: List[SP3Record] = []
    current_record: Optional[SP3Record] = None
    for line in input.readlines():
        if line.startswith("*"):
            if current_record is not None:
                records.append(current_record)
            epoch = parse_time_epoch(line[2:].strip())
            current_record = SP3Record(epoch, {}, {})
        elif line.startswith("P") and parse_position:
            if current_record is None:
                raise Exception("Position record found before epoch")
            veh_id, x, y, z, c = parse_position_and_clock(line)
            current_record.p_entries[veh_id] = (x, y, z, c)
        elif line.startswith("V") and parse_velocity:
            if current_record is None:
                raise Exception("Velocity record found before epoch")
            veh_id, x, y, z, c = parse_velocity_and_clock(line)
            current_record.v_entries[veh_id] = (x, y, z, c)
    if current_record is not None:
        records.append(current_record)
    return records


def format_records(records: List[SP3Record]) -> List[str]:
    formatted_lines = []
    
    for record in records:
        # Format the epoch line
        epoch_line = f"* {format_time_epoch(record.epoch)}"
        formatted_lines.append(epoch_line)
        
        # Format the position entries
        for veh_id, entry in record.p_entries.items():
            if entry is None:
                continue
            (x, y, z, c) = entry
            position_line = f"P{format_position_and_clock(veh_id, x, y, z, c)}"
            formatted_lines.append(position_line)
            if veh_id in record.v_entries:
                entry = record.v_entries[veh_id]
                if entry is None:
                    continue
                (vx, vy, vz, vc) = entry
                velocity_line = f"V{format_position_and_clock(veh_id, vx, vy, vz, vc)}"
                formatted_lines.append(velocity_line)
    
    return formatted_lines


@dataclass
class SP3Arrays:
    epochs: np.ndarray
    position: Dict[str, np.ndarray]
    velocity: Dict[str, np.ndarray]
    clock: Dict[str, np.ndarray]
    clock_rate: Dict[str, np.ndarray]

    _filepaths: Optional[List[str]] = None


class Dataset:

    def __init__(self) -> None:
        self.header: Optional[SP3Header] = None
        self.records: List[SP3Record] = []

        self._filepaths: List[str] = []
    
    def load_file(
            self,
            input: io.TextIOWrapper,
            parse_position: bool = True,
            parse_velocity: bool = False,
            strict: bool = True
        ) -> "Dataset":
        # todo: add options for epochs, etc.
        header = parse_header(input, strict)
        if self.header is None:
            self.header = header
        else:
            # TODO check for header compatibility when merging datasets
            # check phase offsets, applied DCBs, etc
            if strict:
                assert(False)
            self.header = header
        records = parse_records(input, parse_position, parse_velocity, strict)
        self.records += records

        self._filepaths.append(input.name)
        return self

    def load_files(
            self,
            filepaths: List[str],
            parse_position: bool = True,
            parse_velocity: bool = True,
            strict: bool = True,
            verbose: bool = False
        ) -> "Dataset":
        num_files = len(filepaths)
        for i, filepath in enumerate(filepaths):
            if verbose:
                print(f"\rLoading {i: 3} / {num_files}", end="")
            with open(filepath, "r") as f:
                self.load_file(f, parse_position, parse_velocity, strict)
        if verbose:
            print("...Done")
        return self
    
    def save_file(
            self,
            output: io.TextIOWrapper,
            strict: bool = True
    ) -> None:
        if self.header is not None:
            header_lines = format_header(self.header)
            # append line separators
            header_lines = [line + "\n" for line in header_lines]
            output.writelines(header_lines)
        else:
            if strict:
                raise Exception("SP3 Dataset cannot write file when `self.header` is None.")
        record_lines = format_records(self.records)
        record_lines = [line + "\n" for line in record_lines]
        output.writelines(record_lines)
        
    
    def get_sp3_arrays(
            self,
            merge_duplicates: bool = False,
            convert_keys: Optional[Callable[[str], str]] = None
    ) -> SP3Arrays:
        if merge_duplicates:
            records_dict: dict[float, SP3Record] = {}
            for record in self.records:
                if record.epoch not in records_dict:
                    records_dict[record.epoch] = record
                else:
                    existing_record = records_dict[record.epoch]
                    existing_record.p_entries.update(record.p_entries)
                    existing_record.v_entries.update(record.v_entries)
            records: list[SP3Record] = list(records_dict.values())
        else:
            records: list[SP3Record] = self.records

        records = sorted(records, key=lambda record: record.epoch)
        num_records = len(records)
        epochs = np.array([record.epoch for record in records])
        positions = {}
        velocities = {}
        clocks = {}
        clock_rates = {}
        
        for index, record in enumerate(records):
            for veh_id, entry in record.p_entries.items():
                veh_id = veh_id if convert_keys is None else convert_keys(veh_id)
                if entry is None:
                    continue
                if veh_id not in positions:
                    positions[veh_id] = np.nan * np.zeros((num_records, 3))
                    clocks[veh_id] = np.nan * np.zeros(num_records)
                positions[veh_id][index, :] = entry[:3]
                clocks[veh_id][index] = entry[3]
            for veh_id, entry in record.v_entries.items():
                veh_id = veh_id if convert_keys is None else convert_keys(veh_id)
                if entry is None:
                    continue
                if veh_id not in velocities:
                    velocities[veh_id] = np.nan * np.zeros((num_records, 3))
                    clock_rates[veh_id] = np.nan * np.zeros(num_records)
                velocities[veh_id][index, :] = entry[:3]
                clock_rates[veh_id][index] = entry[3]
        
        return SP3Arrays(
            epochs=epochs,
            position=positions,
            velocity=velocities,
            clock=clocks,
            clock_rate=clock_rates,
            _filepaths=self._filepaths
        )
