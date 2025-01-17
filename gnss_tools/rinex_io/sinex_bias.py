"""
Author Brian Breitsch
Date: 2025-01-02
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
import re
import io
from typing import Dict, List, Optional


@dataclass
class SINEXShortNameMetadata:
    ac: str
    gps_week: int
    dow: int


def parse_sinex_short_name(filename: str) -> Optional[SINEXShortNameMetadata]:
    """
    The SINEX BIAS files are named:
    CCCWWWWD.BIA
    where
    CCC: 3-figure Analysis Center (AC) designator
    WWWW: GPS week
    D: Day of week (0-6) or 7 for a weekly file
    """
    pattern = r"^(\w{3})(\d{4})(\d)\.BIA$"
    match = re.match(pattern, filename)
    if match:
        return SINEXShortNameMetadata(
            ac=match.group(1), gps_week=int(match.group(2)), dow=int(match.group(3))
        )
    else:
        return None


@dataclass
class SINEXLongNameMetadata:
    ac: str
    version: int
    campaign: str
    product_type: str
    year: int
    day_of_year: int
    hour: int
    minute: int
    intended_period: int
    separator1: str
    sampling_interval: int
    separator2: str
    content_type: str
    separator3: str
    format_extension: int
    extension: str


def parse_long_filename(filename: str) -> Optional[SINEXLongNameMetadata]:
    """
    Parse the long filename and extract relevant information.
    Returns a dictionary with the parsed values.
    """
    pattern = r"^(\w{3})(\d{1})(\w{3})(\w{3})_(\d{4})(\d{3})(\d{2})(\d{2})_(\d{2})(\w{1})_(\d{2})(\d{2})(\d{2})_(\d{2})(\w{1})_(\d{3})(\w{3})\.*(\w*)$"
    match = re.match(pattern, filename)
    if match:
        parsed_data = SINEXLongNameMetadata(
            ac=match.group(1),
            version=int(match.group(2)),
            campaign=match.group(3),
            product_type=match.group(4),
            year=int(match.group(5)),
            day_of_year=int(match.group(6)),
            hour=int(match.group(7)),
            minute=int(match.group(8)),
            intended_period=int(match.group(9)),
            separator1=match.group(10),
            sampling_interval=int(match.group(11)),
            separator2=match.group(12),
            content_type=match.group(13),
            separator3=match.group(14),
            format_extension=int(match.group(15)),
            extension=match.group(16),
        )
        return parsed_data
    else:
        return None


@dataclass
class SINEX_HeaderLine:
    format_version: str
    file_agency_code: str
    creation_time: datetime
    agency_code: str
    start_time: datetime
    end_time: datetime
    bias_mode: str
    number_of_estimates: int

    @staticmethod
    def parse(line: str) -> Optional["SINEX_HeaderLine"]:
        pattern = re.compile(r"(\w5) (\d[.]\d{2}) (\w{3}) (\d{4}:\d{3}:\d{5}) (\w{3}) (\d{4}:\d{3}:\d{5}) (\w{1}) (\d{8})")
        match = pattern.match(line)
        if match:
            format_version = match.group(1)
            file_agency_code = match.group(2)
            creation_time = datetime.strptime(match.group(3), "%Y:%j:%H%M%S")
            agency_code = match.group(4)
            start_time = datetime.strptime(match.group(5), "%Y:%j:%H%M%S")
            end_time = datetime.strptime(match.group(6), "%Y:%j:%H%M%S")
            bias_mode = match.group(7)
            number_of_estimates = int(match.group(8))
            return SINEX_HeaderLine(
                format_version,
                file_agency_code,
                creation_time,
                agency_code,
                start_time,
                end_time,
                bias_mode,
                number_of_estimates,
            )
        else:
            return None


@dataclass
class SINEX_FileReference:
    description: str
    output: str
    contact: str
    software: str
    hardware: str
    input: str
    reference_frame: str

    @staticmethod
    def parse(lines: List[str]) -> "SINEX_FileReference":
        reference_frame = None
        description = None
        input = None
        output = None
        contact = None
        hardware = None
        software = None
        for line in lines:
            if line.startswith("REFERENCE FRAME"):
                reference_frame = line[15].strip()
            elif line.startswith("DESCRIPTION"):
                description = line[11].strip()
            elif line.startswith("INPUT"):
                input = line[5].strip()
            elif line.startswith("OUTPUT"):
                output = line[6].strip()
            elif line.startswith("CONTACT"):
                contact = line[7].strip()
            elif line.startswith("HARDWARE"):
                hardware = line[8].strip()
            elif line.startswith("SOFTWARE"):
                software = line[8].strip()
        return SINEX_FileReference(
            reference_frame, description, input, output, contact, hardware, software
        )


@dataclass
class SINEX_BiasReceiverInformation:
    station_name: str
    constellation: str
    receiver_group_identifier: str
    start_time: datetime
    end_time: datetime
    receiver_type: str
    receiver_firmware: Optional[str]

    @staticmethod
    def parse(lines: List[str]) -> "SINEX_BiasReceiverInformation":
        station_name = None
        constellation = None
        receiver_group_identifier = None
        start_time = None
        end_time = None
        receiver_type = None
        receiver_firmware = None
        for line in lines:
            if line.startswith("Station Name"):
                station_name = line[1:].strip()
            elif line.startswith("Constellation"):
                constellation = line[1:].strip()
            elif line.startswith("Receiver Group Identifier"):
                receiver_group_identifier = line[1:].strip()
            elif line.startswith("Start Time"):
                start_time = datetime.strptime(line[1:].strip(), "%Y:%j:%H:%M:%S")
            elif line.startswith("End Time"):
                end_time = datetime.strptime(line[1:].strip(), "%Y:%j:%H:%M:%S")
            elif line.startswith("Receiver Type"):
                receiver_type = line[1:].strip()
            elif line.startswith("Receiver Firmware"):
                receiver_firmware = line[1:].strip()
        return SINEX_BiasReceiverInformation(
            station_name,
            constellation,
            receiver_group_identifier,
            start_time,
            end_time,
            receiver_type,
            receiver_firmware,
        )

@dataclass
class SINEX_BiasDescription:
    observation_sampling: int
    parameter_spacing: int
    determination_method: str
    bias_mode: str
    time_system: str
    receiver_clock_reference_gnss: str
    satellite_clock_reference_observables: Dict[str, List[str]]

    @staticmethod
    def parse(lines: List[str]) -> "SINEX_BiasDescription":
        observation_sampling = None
        parameter_spacing = None
        determination_method = None
        bias_mode = None
        time_system = None
        receiver_clock_reference_gnss = None
        satellite_clock_reference_observables = {}
        for line in lines:
            if line.startswith("OBSERVATION_SAMPLING"):
                observation_sampling = int(line[20:].strip())
            elif line.startswith("PARAMETER_SPACING"):
                parameter_spacing = int(line[17:].strip())
            elif line.startswith("DETERMINATION_METHOD"):
                determination_method = line[20:].strip()
            elif line.startswith("BIAS_MODE"):
                bias_mode = line[9:].strip()
            elif line.startswith("TIME_SYSTEM"):
                time_system = line[11:].strip()
            elif line.startswith("RECEIVER_CLOCK_REFERENCE_GNSS"):
                receiver_clock_reference_gnss = line[29:].strip()
            elif line.startswith("SATELLITE_CLOCK_REFERENCE_OBSERVABLES"):
                vars = line[37:].strip().split()
                sat_id = vars[0]
                observables = vars[1:]
                satellite_clock_reference_observables[sat_id] = observables
        return SINEX_BiasDescription(
            observation_sampling,
            parameter_spacing,
            determination_method,
            bias_mode,
            time_system,
            receiver_clock_reference_gnss,
            satellite_clock_reference_observables,
        )


class SINEX_BiasType(Enum):
    DSB = auto()
    ISB = auto()
    OSB = auto()


@dataclass
class SINEX_BiasSolutionEntry:
    bias: SINEX_BiasType
    svn: str
    prn: str
    station: str
    obs1: str
    obs2: str
    bias_start: datetime
    bias_end: datetime
    unit: str
    estimated_value: Optional[float]
    std_dev: Optional[float]
    estimated_slope: Optional[float]
    std_dev_slope: Optional[float]

    @staticmethod
    def parse(line: str, strict: bool = True) -> Optional["SINEX_BiasSolutionEntry"]:
        bias = line[1:5].strip()
        svn = line[6:10].strip()
        prn = line[11:14].strip()
        station = line[15:24].strip()
        obs1 = line[25:29].strip()
        obs2 = line[30:34].strip()
        bias_start_str = line[35:49].strip()
        bias_end_str = line[50:64].strip()
        unit = line[65:69].strip()
        estimated_value_str = line[70:91].strip()
        std_dev_str = line[92:103].strip()

        estimated_slope = None
        std_dev_slope = None
        # TODO check for slope and slope STD

        try:
            year, day_of_year, seconds = map(int, bias_start_str.split(":"))
            bias_start = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, seconds=seconds)
            year, day_of_year, seconds = map(int, bias_end_str.split(":"))
            bias_end = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, seconds=seconds)
        except ValueError as e:
            print("Error parsing bias solution entry start/end times: ", line)
            if strict:
                raise e
            return None
        
        estimated_value = float("nan")
        std_dev = float("nan")
        try:
            estimated_value = float(estimated_value_str)
        except ValueError as e:
            pass
        try:
            std_dev = float(std_dev_str)
        except ValueError as e:
            pass

        return SINEX_BiasSolutionEntry(
            bias,
            svn,
            prn,
            station,
            obs1,
            obs2,
            bias_start=bias_start,
            bias_end=bias_end,
            unit=unit,
            estimated_value=estimated_value,
            std_dev=std_dev,
            estimated_slope=estimated_slope,
            std_dev_slope=std_dev_slope,
        )
    
    def format(self) -> str:
        bias_start_seconds = self.bias_start.hour * 3600 + self.bias_start.minute * 60 + self.bias_start.second
        bias_start_str = self.bias_start.strftime("%Y:%j:") + f"{bias_start_seconds:05d}"
        bias_end_seconds = self.bias_end.hour * 3600 + self.bias_end.minute * 60 + self.bias_end.second
        bias_end_str = self.bias_end.strftime("%Y:%j:") + f"{bias_end_seconds:05d}"
        return f" {self.bias: <4} {self.svn: <4} {self.prn: <3} {self.station: <9} {self.obs1: <4} {self.obs2: <4} {bias_start_str} {bias_end_str} {self.unit <4} {self.estimated_value: >21.15f} {self.std_dev: >11.6f} {self.estimated_slope: >21.15f} {self.std_dev_slope: >11.6f}"


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
    Further refine sorted bias solution entries (from `sort_gnss_bias_solutions`) to only contain satellite code biases for a particular
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


class SINEX_Dataset:

    def __init__(self):
        self.header: Optional[SINEX_HeaderLine] = None
        self.file_reference: Optional[SINEX_FileReference] = None
        self.file_comments: Optional[List[str]] = None
        self.input_acknowledgments: Optional[List[str]] = None
        self.receiver_information: Optional[SINEX_BiasReceiverInformation] = None
        self.bias_description: Optional[SINEX_BiasDescription] = None
        self.bias_solutions: List[SINEX_BiasSolutionEntry] = []

        self._filepaths: List[str] = []
    
    def parse(self, text_input: io.StringIO, strict: bool = True, verbose: bool = False) -> None:
        current_section: Optional[str] = None
        section_lines: List[str] = []
        comment_lines: List[str] = []
        for line in text_input:
            # TODO: LINE FOR TEST DATASETS ONLY
            # skip ellipses in test datasets
            if line.strip() == "...":
                continue
            if line.startswith("%"):
                # Should be either start of file (=BIA) or end of file (=ENDBIA)
                if line.startswith("%=BIA"):
                    self.header = SINEX_HeaderLine.parse(line)
                elif line.startswith("%=ENDBIA"):
                    if verbose:
                        print("Reached end of bias file.")
                else:
                    print("Warning: unknown header line: ", line)
            elif line.startswith("*"):
                comment_lines.append(line)
            elif line.startswith("+"):
                if section_lines:
                    print("Warning: last section never ended: ", current_section)
                    self.parse_section(current_section, section_lines)
                section_lines = []
                current_section = line[1:].strip()
            elif line.startswith("-"):
                if line[1:].strip() != current_section:
                    # If current section is FILE/COMMENT, we need to ignore lines that start with '-'
                    if current_section == "FILE/COMMENT":
                        section_lines.append(line)
                        continue
                    print("Warning: section end does not match current section: ", line[1:].strip(), current_section)
                
                self.parse_section(current_section, section_lines)
                section_lines = []
                current_section = None
            else:
                section_lines.append(line)
            
        if section_lines:
            self.parse_section(current_section, section_lines)
    
    def parse_section(self, section_header: str, lines: List[str]) -> None:
        if section_header == "FILE/REFERENCE":
            self.file_reference = SINEX_FileReference.parse(lines)
        elif section_header == "FILE/COMMENT":
            if self.file_comments is None:
                self.file_comments = []
            self.file_comments += lines
        elif section_header == "INPUT/ACKNOWLEDGMENTS":
            if self.input_acknowledgments is None:
                self.input_acknowledgments = []
            self.input_acknowledgments += lines
        elif section_header == "BIAS/RECEIVER":
            self.receiver_information = SINEX_BiasReceiverInformation.parse(lines)
        elif section_header == "BIAS/DESCRIPTION":
            self.bias_description = SINEX_BiasDescription.parse(lines)
        elif section_header == "BIAS/SOLUTION":
            for line in lines:
                self.bias_solutions.append(SINEX_BiasSolutionEntry.parse(line))
        else:
            print("Warning: unknown section: ", section_header, len(lines))

    def extract_satellite_code_biases(self, obs1: str, obs2: str, strict: bool = True) -> Dict[datetime, Dict[str, float]]:
        sorted_entries = sort_gnss_bias_solutions(self.bias_solutions)
        return extract_gnss_satellite_code_biases_from_sorted_entries(sorted_entries, obs1, obs2, strict)

