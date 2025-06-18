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
        string_list: List[str],
        num_columns: int = 2,
        colsep: str = "  ",
        row_first: bool = True,
        pad_along_columns: bool = True,
    ) -> None:
    """
    Print the strings from `string_list` in columns.
    """
    assert num_columns > 0, "Number of columns must be greater than 0."
    # First, split the string list into rows and columns
    num_rows = (len(string_list) + num_columns - 1) // num_columns
    rows: list[list[str]] = []
    if row_first:
        # Split by rows first
        current_row = []
        for i, item in enumerate(string_list):
            current_row.append(item)
            if len(current_row) == num_columns:
                rows.append(current_row)
                current_row = []
        if current_row:
            if len(current_row) < num_columns:
                # Pad the current row if necessary
                current_row += [""] * (num_columns - len(current_row))
            rows.append(current_row)
    else:
        # Split by columns first
        for i, item in enumerate(string_list):
            row_index = i % num_rows
            if len(rows) <= row_index:
                rows.append([])
            rows[row_index].append(item)
        for row in rows:
            if len(row) < num_columns:
                # Pad the row if necessary
                row += [""] * (num_columns - len(row))
    # Pad strings if necessary
    if pad_along_columns:
        for j in range(num_columns):
            # Find the maximum length of the strings in each column
            max_column_string_length = max(
                len(row[j]) for row in rows
            )
            for row in rows:
                row[j] = row[j].ljust(max_column_string_length)
    # Concatenate the rows into a single string with column separation
    row_strings = []
    for row in rows:
        row_string = colsep.join(row)
        row_strings.append(row_string)
    # Print the rows
    for row_string in row_strings:
        print(row_string)


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
