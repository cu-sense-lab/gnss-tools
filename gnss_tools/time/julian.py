"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from typing import Iterable, List
from datetime import datetime
import numpy as np
import logging


def datetime_to_julian(time: datetime) -> float:
    """Return Julian day for given UTC datetime
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    """
    logging.warning("This function is deprecated. Use `datetime_to_julian_day` instead.")
    year, month = time.year, time.month
    days = (time - datetime(year, month, 1)).total_seconds() / 86400
    return 367 * year - int(7 * int((month + 9) / 12) / 4) + int(275 * month / 9) + days


def datetime_to_julian_array(times: Iterable[datetime]) -> List[float]:
    """Return Julian day for given UTC datetimes
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    """
    logging.warning("This function is deprecated. Use `datetime_to_julian_day` instead.")
    return [datetime_to_julian(time) for time in times]


def datetime_to_julian_day_dt(epoch: datetime) -> float:
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    days = (epoch - j2000).total_seconds() / 86400
    return days + 2451545.0


def days_since_j2000_dt(epoch: datetime) -> float:
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    return (epoch - j2000).total_seconds() / 86400


def datetime_to_julian_day(epoch: datetime) -> float:
    julian_day = (
        367 * epoch.year
        - np.floor(7 * (epoch.year + np.floor((epoch.month + 9) / 12)) / 4)
        + np.floor(275 * epoch.month / 9)
        + epoch.day
        + 1721013.5
        + epoch.hour / 24
        + epoch.minute / 1440
        + epoch.second / 86400
    )
    return julian_day

def datetime_to_julian_day_array(epochs: Iterable[datetime]) -> np.ndarray:
    return np.array([datetime_to_julian_day(epoch) for epoch in epochs])


def days_since_j2000(epoch: datetime) -> float:
    julian_day = datetime_to_julian_day(epoch)
    return julian_day - 2451545.0

def days_since_j2000_array(epochs: Iterable[datetime]) -> np.ndarray:
    return datetime_to_julian_day_array(epochs) - 2451545.0