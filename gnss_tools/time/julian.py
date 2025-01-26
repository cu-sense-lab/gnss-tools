"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from typing import Iterable, List
from datetime import datetime, timezone

def datetime_to_julian(time: datetime) -> float:
    '''Return Julian day for given UTC datetime
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    ''' 
    year, month = time.year, time.month
    days = (time - datetime(year, month, 1, tzinfo=timezone.utc)).total_seconds() / 86400
    return 367 * year - int(7 * int((month + 9) / 12) / 4) + int(275 * month / 9) + days

def datetime_to_julian_array(times: Iterable[datetime]) -> List[float]:
    '''Return Julian day for given UTC datetimes
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    ''' 
    return [datetime_to_julian(time) for time in times]