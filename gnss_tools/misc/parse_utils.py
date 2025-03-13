import re
from typing import List, Dict, Any
from datetime import datetime, timedelta



def parse_pattern_info(
        str_list: List[str],
        pattern: re.Pattern,
        parse_datetime: bool = True,
        keep_original_str: bool = False,
        original_str_key: str = "original_str"
) -> List[Dict[str, Any]]:
    """
    Parse a list of strings using a regular expression pattern.

    Args:
        str_list: List of strings to parse.
        pattern: Regular expression pattern to use for parsing.
        parse_datetime: If True, convert the parsed year, month, and day to a datetime object.

    Returns:
        List of dictionaries containing the parsed information.
    """
    parsed_info: List[Dict[str, Any]] = []
    for str_ in str_list:
        match = pattern.match(str_)
        if match:
            parsed_info.append(match.groupdict())
    for i, info in enumerate(parsed_info):
        if parse_datetime:
            year = info.get("year")
            month = info.get("month")
            day = info.get("day")
            doy = info.get("doy")
            hour = info.get("hour", 0)
            minute = info.get("minute", 0)
            second = info.get("second", 0)
            microsecond = info.get("microsecond", 0)
            if year and doy:
                info["datetime"] = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
            elif year and month and day:
                info["datetime"] = datetime(
                    int(year), int(month), int(day), int(hour), int(minute), int(second), int(microsecond)
                )
        if keep_original_str:
            info[original_str_key] = str_list[i]
    return parsed_info

def sort_info_by_date(
        info_list: List[Dict[str, Any]],
        date_key: str = "datetime"
) -> Dict[datetime, Dict[str, Any]]:
    """
    Sort a list of dictionaries by date, as specified by datetime value.

    Args:
        info_list: List of dictionaries containing date information.
        date_key: Key in the dictionary containing the date information.

    Returns:
        Dictionary with datetime objects as keys and the corresponding dictionary as values.
    """
    sorted_info: Dict[datetime, Dict[str, Any]] = {}
    for info in info_list:
        date = info.get(date_key)
        if date:
            sorted_info[date] = info
    return sorted_info

def get_info_by_date(
        str_list: List[str],
        pattern: re.Pattern,
        keep_original_str: bool = False,
        original_str_key: str = "original_str"
) -> Dict[datetime, Dict[str, Any]]:
    """
    Parse a list of strings using a regular expression pattern and sort the parsed information by date.

    Args:
        str_list: List of strings to parse.
        pattern: Regular expression pattern to use for parsing.

    Returns:
        Dictionary with datetime objects as keys and the corresponding dictionary as values.
    """
    parsed_info = parse_pattern_info(str_list, pattern, keep_original_str=keep_original_str, original_str_key=original_str_key)
    sorted_info = sort_info_by_date(parsed_info)
    return sorted_info