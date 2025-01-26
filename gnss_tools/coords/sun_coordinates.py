"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from datetime import datetime
from typing import Iterable
import numpy as np
from gnss_tools.time.julian import datetime_to_julian_array

def compute_sun_eci_coordinates(times: Iterable[datetime]) -> np.ndarray:
    '''Return sun ECI coordinates for `times`.
    `times` can be any format accepted by `time2jd` from `time_utils` package
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    '''
    jd = np.asarray(datetime_to_julian_array(times))
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