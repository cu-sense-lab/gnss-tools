"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from datetime import datetime, timezone
from .gpst import GPSTime
import numpy as np


def gpst2gmst(gps_time: GPSTime) -> float:
    """Converts GPSTime object into GMST--Greenwich Mean Standard Time.
    
    Inputs:
        gps_time: GPSTime object
    
    Returns:
        GMST in hours
    """
    delta_days = (gps_time.whole_seconds + gps_time.frac_seconds - 630763213.0) / (3600 * 24)
    return 18.697374558 + 24.065709824419 * delta_days


def gpst2gmst_vec(
        gps_time: np.ndarray,  # dtype: numpy GTime or np.float64
        gtime_dtype: bool = False,
        ) -> np.ndarray:  # dtype: numpy float64
    """Converts GPSTime object into GMST--Greenwich Mean Sidereal Time.
    
    Inputs:
        gps_time: GPSTime object
    
    Returns:
        GMST in hours
    """
    if gtime_dtype:
        delta_days = ((gps_time['whole_seconds'] - 630763213) + gps_time['frac_seconds']) / (3600 * 24)
    else:
        delta_days = (gps_time - 630763213.0) / (3600 * 24)
    return 18.697374558 + 24.065709824419 * delta_days


def dt2gmst(dt: datetime) -> float:
    """Converts UTC datetime object into GMST--Greenwich Mean Sidereal Time.
    
    Inputs:
        dt: datetime object
    
    Returns:
        GMST in hours
    """
    delta_days = (dt - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / (3600 * 24)
    return 18.697374558 + 24.065709824419 * delta_days