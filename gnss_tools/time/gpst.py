"""
Utilities for doing GPST time conversions.

Author: Brian Breitsch
Date: 2025-01-02
"""

import numpy as np
from datetime import datetime, timedelta, timezone
from .leap_seconds import utc_tai_offset
from .gtime import GTIME_DTYPE, GTime
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


GPS_EPOCH = datetime(year=1980, month=1, day=6, hour=0, minute=0, second=0)
GPS_TAI_OFFSET = utc_tai_offset(GPS_EPOCH)
SECONDS_IN_WEEK = 3600 * 24 * 7

def convert_datetime_to_gps_seconds(dt: datetime) -> float:
    # if not hasattr(dt, "tzinfo") or dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
    #     dt = dt.replace(tzinfo=timezone.utc)
    time_gps_offset = utc_tai_offset(dt) - GPS_TAI_OFFSET
    timedelta = dt - GPS_EPOCH + time_gps_offset
    return timedelta.days * 86400 + timedelta.seconds + timedelta.microseconds / 1e6

def convert_gps_seconds_to_datetime(gps_seconds: float) -> datetime:
    microseconds = (gps_seconds % 1) * 1e6
    return GPS_EPOCH + timedelta(seconds=gps_seconds, microseconds=microseconds) \
              - GPS_TAI_OFFSET + utc_tai_offset(GPS_EPOCH)


def convert_datetime_array_to_gps_seconds_array(dt_array: Iterable[datetime]) -> np.ndarray:
    return np.array([convert_datetime_to_gps_seconds(dt) for dt in dt_array], dtype=np.float64)

def convert_gps_seconds_array_to_datetime64_array(gps_seconds_array: Iterable[float]) -> np.ndarray:
    return np.array([convert_gps_seconds_to_datetime(gpst) for gpst in gps_seconds_array], dtype=np.datetime64)

def convert_gps_seconds_to_datetime_list(gps_seconds_array: Iterable[float]) -> List[datetime]:
    return [convert_gps_seconds_to_datetime(gpst) for gpst in gps_seconds_array]


@dataclass(slots=True)
class GPSTime(GTime):
    # whole_seconds: int
    # frac_seconds: float

    @property
    def week_num(self) -> int:
        return self.whole_seconds // SECONDS_IN_WEEK
    
    @property
    def tow(self) -> GTime:
        return GTime(self.whole_seconds - self.week_num * SECONDS_IN_WEEK, self.frac_seconds)

    def to_datetime(self) -> datetime:
        """
        Converts GPSTime to `datetime` object.

        Returns:
            datetime object
        """
        microseconds = int(self.frac_seconds * 1e6)
        return GPS_EPOCH + timedelta(seconds=float(self.whole_seconds), microseconds=microseconds) \
              - GPS_TAI_OFFSET + utc_tai_offset(GPS_EPOCH)
    
    @staticmethod
    def from_datetime(dt: datetime) -> "GPSTime":
        """
        Computes GPSTime from `datetime` object.

        Inputs:
            dt: datetime object
        Returns:
            GPSTime object
        """
        # if not hasattr(dt, "tzinfo") or dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        #     dt = dt.replace(tzinfo=timezone.utc)
        time_gps_offset = utc_tai_offset(dt) - GPS_TAI_OFFSET
        timedelta = dt - GPS_EPOCH + time_gps_offset
        # print(time_gps_offset, timedelta, dt, GPS_EPOCH)
        return GPSTime(timedelta.days * 86400 + timedelta.seconds, timedelta.microseconds / 1e6)
    
    @staticmethod
    def from_week_and_tow(week_num: int, tow: GTime) -> "GPSTime":
        """
        Computes GPSTime from GPS week number and time of week.
        
        Inputs:
            week_num: int - GPS week number
            tow: GTime - time of week
        
        Returns:
            GPSTime object
        """
        return GPSTime(week_num * SECONDS_IN_WEEK + tow.whole_seconds, tow.frac_seconds)
    
    @staticmethod
    def arange(start: "GPSTime", end: "GPSTime", step: Optional[float] = None) -> np.ndarray:
        """
        Creates GPSTime array from start to end with given step.
        
        Inputs:
            start: GPSTime - start time
            end: GPSTime - end time
            step: float - step in seconds
        
        Returns:
            GPSTime array
        """
        if step is None:
            step = 1
        delta = np.arange(0, end.whole_seconds - start.whole_seconds + end.frac_seconds, step, dtype=np.float64)
        delta_frac, delta_int = np.modf(delta)
        whole_seconds = start.whole_seconds + delta_int
        frac_seconds = start.frac_seconds + delta_frac
        delta_frac, delta_ind = np.modf(frac_seconds)
        whole_seconds += delta_ind
        frac_seconds = delta_frac
        result = np.zeros(whole_seconds.shape, dtype=GTIME_DTYPE)
        result["whole_seconds"] = whole_seconds
        result["frac_seconds"] = frac_seconds
        return result

