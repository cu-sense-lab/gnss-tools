"""
Author Brian Breitsch
Date: 2025-01-02
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import io
import logging
import re
from typing import Dict, List, Optional
import numpy as np


@dataclass
class IonexFilenameMetadata:
    analysis_center: str
    region_code: str
    day_of_year: int
    sequence_number: int
    year: int

    @staticmethod
    def parse(filename: str, century: int = 2000) -> Optional['IonexFilenameMetadata']:
        if len(filename) != 12:
            raise Exception("Invalid IONEX filename length")
        pattern = re.compile(r"^(\w{3})(\w{1})(\d{3})(\d{1})[.](\d{2})I$")
        match = pattern.match(filename)
        if not match:
            return None
        analysis_center = match.group(1)
        region_code = match.group(2)
        day_of_year = int(match.group(3))
        sequence_number = int(match.group(4))
        year = int(match.group(5)) + century
        return IonexFilenameMetadata(analysis_center, region_code, day_of_year, sequence_number, year)


IONEX_MODEL_DEFINITIONS = {
    "BEN": "Bent",
    "ENV": "Envisat",
    "ERS": "ERS",
    "GEO": "Geostationary",
    "GNS": "GNSS",
    "IRI": "IRI",
    "MIX": "Mixed",
    "NNS": "NNS",
    "TOP": "Topex",
    "UNKNOWN": "Unknown"
}

class IonexModel(Enum):
    BEN = "BEN"
    ENV = "ENV"
    ERS = "ERS"
    GEO = "GEO"
    GNS = "GNS"
    IRI = "IRI"
    MIX = "MIX"
    NNS = "NNS"
    TOP = "TOP"
    UNKNOWN = "UNKNOWN"

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True 

class IonexMappingFunction(Enum):
    NONE = "None"
    COSZ = "COSZ"
    QFAC = "QFAC"
    OTHER = "Other"

class IonexSystemCodes(Enum):
    GPS = "G"
    GLONASS = "R"
    GALILEO = "E"
    BEIDOU = "C"
    QZSS = "J"
    SBAS = "S"
    IRNSS = "I"
    MIXED = "M"

@dataclass
class SatelliteDCBEntry:
    satellite_id: str
    dcb: float
    rms: Optional[float]

    @staticmethod
    def parse(line: str) -> 'SatelliteDCBEntry':
        satellite_id_str = line[3:6].strip()
        dcb_str = line[7:17].strip()
        rms_str = line[17:27].strip()
        if not satellite_id_str:
            raise Exception("Satellite ID not present for `PRN / BIAS / RMS` entry")
        if len(satellite_id_str) != 3:
            raise Exception("Satellite ID must be 3 characters long")
        satellite_id = satellite_id_str
        if dcb_str:
            dcb = float(dcb_str)
        else:
            raise Exception("DCB not present for `PRN / BIAS / RMS` entry")
        if rms_str:
            rms = float(rms_str)
        else:
            rms = None
        return SatelliteDCBEntry(satellite_id, dcb, rms)
    
    def format(self) -> str:
        line = f"   {self.satellite_id}{self.dcb: >10.3f}{self.rms: >10.3f}"
        return line.ljust(60)

@dataclass
class StationDCBEntry:
    station_id: str
    dcb: float
    rms: float
    system_code: Optional[IonexSystemCodes]
    gfz_id: Optional[str]  # TODO this is not specified in spec -- might be in SINEX Bias spec

    @staticmethod
    def parse(line: str) -> 'StationDCBEntry':
        system_code_str = line[3:6].strip()
        station_id_str = line[6:10].strip()
        gfz_id_str = line[11:20].strip()
        dcb_str = line[26:36].strip()
        rms_str = line[36:46].strip()
        if not station_id_str:
            raise Exception("Station ID not present for `PRN / BIAS / RMS` entry")
        if len(station_id_str) != 4:
            raise Exception("Station ID must be 4 characters long")
        station_id = station_id_str
        if dcb_str:
            dcb = float(dcb_str)
        else:
            raise Exception("DCB not present for `PRN / BIAS / RMS` entry")
        if rms_str:
            rms = float(rms_str)
        else:
            raise Exception("RMS not present for `PRN / BIAS / RMS` entry")
        system_code = IonexSystemCodes(system_code_str) if system_code_str else None
        gfz_id = gfz_id_str if gfz_id_str else None
        return StationDCBEntry(station_id, dcb, rms, system_code, gfz_id)
    
    def format(self) -> str:
        line = f"   {self.system_code.value: <3}{self.station_id} {self.gfz_id: <9}      {self.dcb: >10.3f}{self.rms: >10.3f}"
        return line.ljust(60)


@dataclass
class Header:

    ionex_version: str
    ionex_type: str
    ionex_model: IonexModel
    ionex_program: Optional[str]
    run_by: Optional[str]
    run_date: Optional[datetime]
    header_description: Optional[str]

    epoch_of_first_map: datetime
    epoch_of_last_map: datetime
    interval: int
    number_of_maps: int
    mapping_function: str
    elevation_cutoff: float

    observables_used: Optional[str]  # TODO check spec
    number_of_stations: int
    number_of_satellites: int
    base_radius: float
    map_dimension: int

    height_1: float
    height_2: float
    dheight: float
    latitude_1: float
    latitude_2: float
    dlatitude: float
    longitude_1: float
    longitude_2: float
    dlongitude: float

    exponent: float
    
    # Auxilliary data
    gnss_dcbs: Dict[str, SatelliteDCBEntry]
    station_dcbs: Dict[str, StationDCBEntry]

    comments: List[str]


    @staticmethod
    def parse(input: io.TextIOWrapper):

        ionex_version: Optional[str] = None
        ionex_type: Optional[str] = None
        ionex_model: IonexModel = IonexModel.UNKNOWN
        ionex_program: Optional[str] = None
        run_by: Optional[str] = None
        run_date: Optional[datetime] = None
        header_description: Optional[str] = None
        epoch_of_first_map: Optional[datetime] = None
        epoch_of_last_map: Optional[datetime] = None
        interval: Optional[int] = None
        number_of_maps: Optional[int] = None
        mapping_function: Optional[str] = None
        elevation_cutoff: Optional[float] = None
        observables_used: Optional[str] = None
        number_of_stations: Optional[int] = None
        number_of_satellites: Optional[int] = None
        base_radius: Optional[float] = None
        map_dimension: Optional[int] = None
        height_1: Optional[float] = None
        height_2: Optional[float] = None
        dheight: Optional[float] = None
        latitude_1: Optional[float] = None
        latitude_2: Optional[float] = None
        dlatitude: Optional[float] = None
        longitude_1: Optional[float] = None
        longitude_2: Optional[float] = None
        dlongitude: Optional[float] = None
        exponent: Optional[float] = None

        # Auxilliary DCB data
        gnss_dcbs: Dict[str, SatelliteDCBEntry] = {}
        station_dcbs: Dict[str, StationDCBEntry] = {}

        currently_parsing_auxilliary_data = False
        current_auxilliary_data_section: Optional[str] = None

        description_lines: List[str] = []

        comments: List[str] = []
        
        while line := input.readline():

            if line[60:].strip() == "END OF HEADER":
                break
            line_label = line[60:].strip()
            if line_label == "START OF AUX DATA":
                currently_parsing_auxilliary_data = True
                current_auxilliary_data_section = line[:60].strip()
                continue
            if line_label == "END OF AUX DATA":
                if not currently_parsing_auxilliary_data:
                    # raise Exception("END OF AUX DATA without START OF AUX DATA")
                    logging.warning("Found `END OF AUX DATA` without `START OF AUX DATA`")
                currently_parsing_auxilliary_data = False
                current_auxilliary_data_section = None
                continue

            if line_label == "IONEX VERSION / TYPE":
                ionex_version = line[0:20].strip()
                ionex_type = line[20:40].strip()
                ionex_model_str = line[40:60].strip()
                if ionex_model_str not in IonexModel.__members__:
                    logging.warning(f"Unknown IONEX model: {ionex_model_str}")
                    ionex_model = IonexModel.UNKNOWN
                else:
                    ionex_model = IonexModel(ionex_model_str)
                continue

            if line_label == "PGM / RUN BY / DATE":
                ionex_program = line[0:20].strip()
                run_by = line[20:40].strip()
                run_date_str = line[40:60].strip()
                if run_date_str:
                    DATETIME_FORMAT_STRINGS = [
                        "%y %m %d %H %M %S",
                        "%d-%b-%y %H:%M",
                    ]
                    run_date = None
                    for format_string in DATETIME_FORMAT_STRINGS:
                        try:
                            run_date = datetime.strptime(run_date_str, format_string)
                        except ValueError:
                            continue
                    if run_date is None:
                        logging.warning(f"Unhandled date format in `PGM / RUN BY / DATE`: {run_date_str}")
                continue

            if line_label == "DESCRIPTION":
                description_lines.append(line[:60].strip())
                continue

            if line_label == "COMMENT":
                comments.append(line[:60].strip())
                continue

            if line_label == "EPOCH OF FIRST MAP":
                pass

            # catch map-related header labels for robustness
            if line_label in ["START OF TEC MAP", "END OF TEC MAP", "START OF RMS MAP", "END OF RMS MAP", "EPOCH OF CURRENT MAP", "LAT/LON1/LON2/DLON/H"]:
                logging.warning(f"Found `{line_label}` in header section")
                break
            
            if currently_parsing_auxilliary_data:
                if current_auxilliary_data_section in ["GNSS DCBS"]:
                    satellite_dcb_entry = SatelliteDCBEntry.parse(line)
                    gnss_dcbs[satellite_dcb_entry.satellite_id] = satellite_dcb_entry
                elif current_auxilliary_data_section in ["STATION DCBS"]:
                    station_dcb_entry = StationDCBEntry.parse(line)
                    station_dcbs[station_dcb_entry.station_id] = station_dcb_entry
                elif current_auxilliary_data_section in ["DIFFERENTIAL CODE BIASES"]:
                    line_label = line[60:].strip()
                    if line_label == "PRN / BIAS / RMS":
                        satellite_dcb_entry = SatelliteDCBEntry.parse(line)
                        gnss_dcbs[satellite_dcb_entry.satellite_id] = satellite_dcb_entry
                    elif line_label == "STATION / BIAS / RMS":
                        station_dcb_entry = StationDCBEntry.parse(line)
                        station_dcbs[station_dcb_entry.station_id] = station_dcb_entry
                else:
                    logging.warning(f"Unknown auxilliary data section: {current_auxilliary_data_section}")
                continue
        
        return Header(ionex_version, ionex_type, ionex_model, ionex_program, run_by, run_date, 
                      header_description, epoch_of_first_map, epoch_of_last_map, interval, number_of_maps, 
                      mapping_function, elevation_cutoff, observables_used, number_of_stations, number_of_satellites, 
                      base_radius, map_dimension, height_1, height_2, dheight, latitude_1, latitude_2, dlatitude, 
                      longitude_1, longitude_2, dlongitude, exponent, gnss_dcbs, station_dcbs, comments)
        
    def format_header(self, output: io.TextIOWrapper) -> None:
        header_lines = []
        line = f"     {self.ionex_version: <15}{self.ionex_type.value: <20}{self.ionex_type.value: <20}"
        header_lines.append(line.ljust(60) + "IONEX VERSION / TYPE")


@dataclass
class MapLatitudeBand:
    latitude: float
    lon0: float
    lon1: float
    dlon: float
    height: float
    values: List[float]

@dataclass
class IonexMap:
    map_type: str
    epoch: datetime
    map_latitude_bands: List[MapLatitudeBand]

    def get_lats(self) -> np.ndarray:
        return np.array([band.latitude for band in self.map_latitude_bands])

    def get_lons(self) -> np.ndarray:
        band = self.map_latitude_bands[0]
        lon0 = band.lon0
        lon1 = band.lon1
        dlon = band.dlon
        # Check that all other bands have the same longitudes
        for band in self.map_latitude_bands:
            if band.lon0 != lon0 or band.lon1 != lon1 or band.dlon != dlon:
                raise Exception("Inconsistent longitudes in map")
        return np.arange(lon0, lon1 + dlon, dlon)

    def get_map_array(self, sort_lat: bool = False) -> np.ndarray:
        if sort_lat:
            map_latitude_bands = sorted(self.map_latitude_bands, key=lambda x: x.latitude)
        else:
            map_latitude_bands = self.map_latitude_bands
        num_lats = len(map_latitude_bands)
        band = map_latitude_bands[0]
        num_lons = len(band.values)
        for band in map_latitude_bands:
            if len(band.values) != num_lons:
                raise Exception(f"Inconsistent number of longitudes in map: {num_lons} != {len(band.values)}")
        map_array = np.zeros((num_lats, num_lons))
        for i, band in enumerate(map_latitude_bands):
            map_array[i, :] = band.values
        return map_array
        
    # this function receives a list of lines rather than a TextIOWrapper
    # since the function that parses all the maps will agregate lines for a single map
    @staticmethod
    def parse(lines: List[str], map_type: str = "TEC", strict: bool = False) -> "IonexMap":
        
        if (lines[0][60:].strip() != f"START OF {map_type} MAP"):
            print(lines[0])
            assert(False)
        if (lines[-1][60:].strip() != f"END OF {map_type} MAP"):
            print(lines[-1])
            assert(False)

        lines = iter(lines)

        line = next(lines)
        line = next(lines)
        line_label = line[60:].strip()
        if line_label != "EPOCH OF CURRENT MAP":
            raise Exception("Expected `EPOCH OF CURRENT MAP` label")
        epoch_str = line[:60].strip()
        ymdhms = map(int, epoch_str.split())
        epoch = datetime(*ymdhms)
        map_latitude_bands = []
        line = next(lines, None)
        while line is not None:
            line_label = line[60:].strip()
            if line_label == f"END OF {map_type} MAP":
                break
            elif line_label == "END OF TEC MAP":
                raise Exception(f"Found `END OF TEC MAP` before `END OF {map_type} MAP`")
            elif line_label == "END OF RMS MAP":
                raise Exception(f"Found `END OF RMS MAP` before `END OF {map_type} MAP`")
            elif line_label == "LAT/LON1/LON2/DLON/H":
                lat = float(line[2:8])
                lon0 = float(line[8:14])
                lon1 = float(line[14:20])
                dlon = float(line[20:26])
                height = float(line[26:32])
            
                num_values = int((lon1 - lon0) / dlon) + 1
                values = []
                done_parsing = False

                line = next(lines, None)
                while line is not None:

                    line_label = line[60:].strip()
                    if line_label == f"END OF {map_type} MAP":
                        logging.warning(f"Unexpected end of map data (inner)")
                        break
                    i = 0
                    while i + 5 < len(line):
                        val_str = line[i:i+5].strip()
                        if val_str:
                            values.append(float(val_str))
                        if len(values) == num_values:
                            done_parsing = True
                            break
                        i += 5
                    if done_parsing:
                        break
                    line = next(lines, None)
                
                if line is None:
                    raise Exception("Unexpected end of map data (outer)")
                
                latitude_band = MapLatitudeBand(lat, lon0, lon1, dlon, height, values)

                map_latitude_bands.append(latitude_band)
                
                line = next(lines, None)
            else:
                if strict:
                    raise Exception(f"Unknown label in {map_type} map: {line_label}")
                else:
                    logging.warning(f"Unknown label in {map_type} map: {line_label}")
                    line = next(lines, None)
        
        return IonexMap(map_type, epoch, map_latitude_bands)

        
        

class Dataset:

    def __init__(self) -> None:
        self.header: Optional[Header] = None
        self.tec_maps: List[IonexMap] = []
        self.rms_maps: List[IonexMap] = []

        # self._filepaths: List[str] = []
    
    def load(self, input: io.TextIOWrapper, strict: bool = True) -> None:
        header = None
        tec_maps = []
        rms_maps = []

        header = Header.parse(input)

        current_map_lines = None
        current_map_type = None
        while line := input.readline():
            line_label = line[60:].strip()
            if line_label == "START OF TEC MAP":
                if current_map_lines:
                    raise Exception("Found `START OF TEC MAP` before `END OF ... MAP`")
                    # tec_map = IonexMap.parse(current_map_lines, "TEC")
                    # tec_maps.append(tec_map)
                current_map_lines = [line]
                current_map_type = "TEC"
            elif line_label == "START OF RMS MAP":
                if current_map_lines:
                    raise Exception("Found `START OF RMS MAP` before `END OF ... MAP`")
                current_map_lines = [line]
                current_map_type = "RMS"
            elif line_label == "END OF TEC MAP":
                if current_map_lines:
                    assert(current_map_type == "TEC")
                    current_map_lines.append(line)
                    tec_map = IonexMap.parse(current_map_lines, "TEC", strict)
                    tec_maps.append(tec_map)
                    current_map_lines = None
                    current_map_type = None
            elif line_label == "END OF RMS MAP":
                if current_map_lines:
                    assert(current_map_type == "RMS")
                    current_map_lines.append(line)
                    rms_map = IonexMap.parse(current_map_lines, "RMS", strict)
                    rms_maps.append(rms_map)
                    current_map_lines = None
                    current_map_type = None
            elif line_label == "END OF FILE":
                break
            else:
                if current_map_lines:
                    current_map_lines.append(line)
                else:
                    if strict:
                        raise Exception("Found map data without `START OF ... MAP`")
                    else:
                        logging.warning("Found map data without `START OF ... MAP`")
                        print(line)
                        continue

        self.header = header
        self.tec_maps.extend(tec_maps)
        self.rms_maps.extend(rms_maps)







