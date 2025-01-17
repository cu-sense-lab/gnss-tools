"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from typing import List, Tuple
import numba
import numpy as np
from numpy.typing import NDArray


def find_contiguous_segments(
    valid: NDArray[np.bool_],
    allow_gap_size: int = 0,
) -> List[Tuple[int, int]]:
    """Find contiguous segments of valid data.

    Args:
        valid: Boolean array indicating valid data points.
        allow_gap_size: Maximum gap size to allow within a segment.

    Returns:
        List of tuples containing the start and end indices of each contiguous segment.
    """
    return numba_find_contiguous_segments(len(valid), valid, allow_gap_size)

@numba.njit
def numba_find_contiguous_segments(
    N: int,
    valid: NDArray[np.bool_],
    allow_gap_size: int = 0,
) -> List[Tuple[int, int]]:
    
    segments: List[Tuple[int, int]] = []
    start: int = -1  # use -1 to indicate that we are not in a segment yet
    gap_count: int = 0
    for i in range(N):
        if valid[i]:
            if start == -1:
                start = i
            gap_count = 0
        else:
            gap_count += 1
            if start != -1 and gap_count > allow_gap_size:
                segments.append((start, i - gap_count + 1))
                start = -1
    if start != -1:
        segments.append((start, N - gap_count))
    return segments