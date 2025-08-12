"""
leap_seconds.py

Utilities for computing number of elapsed leap seconds.

@author Brian Breitsch
"""

from datetime import datetime, timedelta, timezone
from urllib.request import urlretrieve  # python 3
from os.path import isfile, dirname, join
from collections import namedtuple
from typing import List

OffsetEpoch = namedtuple('OffsetEpoch', ['epoch', 'offset'])
NTP_EPOCH = datetime(year=1900, month=1, day=1, hour=0, minute=0, second=0)
LEAP_SECOND_EPOCHS: List[OffsetEpoch] = []
_LEAP_SECONDS_FILEPATH = join(dirname(__file__), './leap_second_epochs.txt')


def download_tai_leap_seconds(filepath: str) -> None:
    """Uses `urlretrieve` to get leap seconds list from:
        http://www.ietf.org/timezones/data/leap-seconds.list
    """
    leap_seconds_list_url = 'http://www.ietf.org/timezones/data/leap-seconds.list'
    _ = urlretrieve(leap_seconds_list_url, filepath)


def parse_tai_leap_seconds(filepath: str) -> List[OffsetEpoch]:
    leap_seconds_epochs = []
    with open(filepath, 'r') as leap_seconds_data:
        for line in leap_seconds_data.readlines():
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            if line.startswith('#$'):
                file_update_ntp = int(line.split()[1])
            elif line.startswith('#@'):
                file_expiration_ntp = int(line.split()[1])
            elif line.startswith('#') or line == '': 
                # if line is comment or blank, ignore
                continue
            else:
                ntp_timestamp = int(line.split()[0])
                offset = int(line.split()[1])
                epoch = NTP_EPOCH + timedelta(seconds=ntp_timestamp)
                leap_seconds_epochs.append(OffsetEpoch(epoch, offset))
    return leap_seconds_epochs

# def update_leap_seconds_data():
#     download_tai_leap_seconds(_LEAP_SECONDS_FILEPATH)
#     LEAP_SECOND_EPOCHS = parse_tai_leap_seconds(_LEAP_SECONDS_FILEPATH)

# need to download/parse leap second epochs if not already done
if not LEAP_SECOND_EPOCHS:
    if not isfile(_LEAP_SECONDS_FILEPATH):
        download_tai_leap_seconds(_LEAP_SECONDS_FILEPATH)
        print('downloaded leap seconds file')
    LEAP_SECOND_EPOCHS = parse_tai_leap_seconds(_LEAP_SECONDS_FILEPATH)
    # print('there are {0} leap second epochs'.format(len(LEAP_SECOND_EPOCHS)))


def utc_tai_offset(time: datetime) -> timedelta:
    """
    Calculates the offset (number of leap seconds) between a
    given time and TAI. If `time` is before the first leap
    seconds were introduced in 1972, returns 10--which is the
    original offset introduced in 1972. Otherwise, returns 
    the offset corresponding to the last offset before
    `time`.

    Note: `time` must be timezone aware.
    
    Inputs
    -----
    time: datetime
        the time for which to find leap seconds
    
    Returns
    ------
    offset: timedelta
        the total leap second offset
    """
    offset = 0
    for i in range(len(LEAP_SECOND_EPOCHS)):
        if LEAP_SECOND_EPOCHS[i].epoch > time:
            break
        offset = LEAP_SECOND_EPOCHS[i].offset
    return timedelta(seconds=offset)
