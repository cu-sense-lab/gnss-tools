"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from typing import List


def print_columns(
        string_list: List[str],
        ncol: int = 2,
        colsep: str = "     ",
    ) -> None:
    """
    Print the strings from `string_list` in `ncol` columns, with entries going down rows in each column first.
    """
    nrow = max(len(string_list) // ncol + 1, min(len(string_list), ncol))
    for r in range(nrow):
        print(
            colsep.join(
                [
                    string_list[r + c * nrow]
                    for c in range(ncol)
                    if (r + c * nrow) < len(string_list)
                ]
            )
        )

def print_to_columns(
        lines: List[str],
        break_seq: str = "\n",
        colsep: str = "     ",
    ) -> None:
    """
    Print the strings from `lines` in columns, with entries going down rows in each column first.
    """
    rows = []
    i = 0
    j = 0
    ncol = 1
    while i < len(lines):
        line = lines[i]
        if line == break_seq:
            i += 1
            j = 0
            continue
        while len(rows) <= j:
            rows.append(["" for k in range(ncol - 1)])
        while i < len(string_list) and string_list[i] != break_seq:
            row.append(string_list[i])
            i += 1
        rows.append(row)
        i += 1


import calendar
from datetime import datetime
from typing import List


def print_calendar_with_datetimes(datetimes: List[datetime], ncol: int = 1):
    """Return lines of an ASCII calendar highlighting given datetimes.

    Args:
      datetimes: A list of datetime objects.
    """
    datetimes = sorted(datetimes)
    lines = []

    # Find the first and last year in the list
    first_year = datetimes[0].year
    last_year = datetimes[-1].year

    for year in range(first_year, last_year + 1):
        lines.append(f"\n{year}")
        for month in range(1, 13):
            # Get calendar for the month
            cal = calendar.monthcalendar(year, month)

            # Print month and weekdays
            lines.append(calendar.month_name[month])
            lines.append("Mo Tu We Th Fr Sa Su")

            for week in cal:
                line = ""
                for day in week:
                    if day == 0:
                        line += "   "  # Empty space for days not in the month
                    else:
                        date = datetime(year, month, day).date()
                        if date in [dt.date() for dt in datetimes]:
                            line += f"\033[92m{day:2d}\033[0m " # Highlight the day in green
                        else:
                            line += f"{day:2d} "
                lines.append(line)  # Newline after each week
    print_columns(lines, ncol=ncol)
