"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from datetime import datetime
from typing import Iterable
import numpy as np
from gnss_tools.time.julian import datetime_to_julian_day_array, datetime_to_julian_day, days_since_j2000

def compute_sun_eci_coordinates(times: Iterable[datetime]) -> np.ndarray:
    '''Return sun ECI coordinates for `times`.
    `times` can be any format accepted by `time2jd` from `time_utils` package
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    '''
    jd = np.asarray(datetime_to_julian_day_array(times))
    T_ut1 = (jd - 2451545) / 36525
    T_tdb = T_ut1
    lamb_sun = 280.4606184 + 36000.77005361 * T_tdb  # mean longitude
    M_sun = 357.5277233 + 35999.05034 * T_tdb  # mean anomaly
    lamb_ecli = lamb_sun + 1.914666471 * np.sin(np.radians(M_sun)) + 0.918994643 * np.sin(np.radians(2 * M_sun))  # ecliptic longitude
    dist = 1.000140612 - 0.016708617 * np.cos(np.radians(M_sun)) - 0.000139589 * np.cos(np.radians(2 * M_sun))  # distance to sun
    eps = 23.438291 - 0.0130042 * T_tdb
    return np.stack((dist * np.cos(np.radians(lamb_ecli)),
                  dist * np.cos(np.radians(eps)) * np.sin(np.radians(lamb_ecli)),
                  dist * np.sin(np.radians(eps)) * np.sin(np.radians(lamb_ecli)))).T



def get_sun_position(epoch: datetime) -> np.ndarray:
    """
    From https://astronomy.stackexchange.com/questions/28802/calculating-the-sun-s-position-in-eci
    """
    days = days_since_j2000(epoch)
    # Compute mean longitude of the sun in degrees
    L = 280.4606184 + (36000.77005361 / 36525) * days
    # Compute mean anomaly in degrees
    g = 357.5277233 + (35999.05034 / 36525) * days
    # Compute ecliptic longitude in degrees
    p = L + 1.914666471 * np.sin(g * np.pi / 180) + 0.918994643 * np.sin(2 * g * np.pi / 180)
    # Compute obliquity of the ecliptic in degrees
    q = 23.43929 - (46.8093 / 3600) * (days / 36525)

    u_x = np.cos(p * np.pi / 180)
    u_y = np.cos(q * np.pi / 180) * np.sin(p * np.pi / 180)
    u_z = np.sin(q * np.pi / 180) * np.sin(p * np.pi / 180)

    sun_distance_AU = 1.000140612 - 0.016708617 * np.cos(g * np.pi / 180) - 0.000139589 * np.cos(2 * g * np.pi / 180)
    sun_distance_m = sun_distance_AU * 149597870700.0

    return sun_distance_m * np.array([u_x, u_y, u_z])

def get_sun_direction(epoch: datetime) -> np.ndarray:
    # JD = time_utils.datetime_to_julian(epoch)
    JD = datetime_to_julian_day(epoch)

    # pi = np.pi
    pi = 3.14159265359
    UT1 = (JD - 2451545) / 36525
    longMSUN = 280.4606184 + 36000.77005361 * UT1
    mSUN = 357.5277233 + 35999.05034 * UT1
    ecliptic = (
        longMSUN
        + 1.914666471 * np.sin(mSUN * pi / 180)
        + 0.918994643 * np.sin(2 * mSUN * pi / 180)
    )
    eccen = 23.439291 - 0.0130042 * UT1

    ecliptic_rad = ecliptic * pi / 180
    eccen_rad = eccen * pi / 180

    x = np.cos(ecliptic_rad)
    y = np.cos(eccen_rad) * np.sin(ecliptic_rad)
    z = np.sin(eccen_rad) * np.sin(ecliptic_rad)

    return np.array([x, y, z])


def get_sun_position_2(epoch: datetime) -> np.ndarray:
    sun_distance = 0.989 * 1.496e8
    sun_position = get_sun_direction(epoch)

    sun_position = sun_position * sun_distance

    return sun_position