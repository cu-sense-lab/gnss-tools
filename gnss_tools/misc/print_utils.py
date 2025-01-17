"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from typing import List


def print_columns(string_list: List[str], ncol: int = 2, colsep: str = '     ') -> None:
    '''
    Print the strings from `string_list` in `ncol` columns, with entries going down rows in each column first.
    '''
    nrow = max(len(string_list) // ncol + 1, min(len(string_list), ncol))
    for r in range(nrow):
        print(colsep.join([string_list[r + c * nrow] for c in range(ncol) if (r + c * nrow) < len(string_list)]))