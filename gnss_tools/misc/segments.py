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


def debounce_integer_sequence(
        sequence: np.ndarray,
        debounce_length: int,
        edge: str = "rising",
) -> np.ndarray:
    """
    Debounce an integer sequence by removing any sequences of integers that are
    less than the debounce_length.
    """
    output_sequence = np.zeros_like(sequence)
    previous_val = sequence[0]
    # change_index = {}
    # change_index[previous_val] = 0
    # for i, val in enumerate(sequence):
    #     if val == previous_val:
    #         output_sequence[i] = val
    #     else:
    #         output_sequence[i] = val
    #         change_index[previous_val] = i
    #         if val in change_index and i - change_index[val] < debounce_length:
    #             output_sequence[change_index[val]:i] = val
    #         previous_val = val
    change_start_index = 0
    change_stop_index = 0
    for i, val in enumerate(sequence):
        if val == previous_val:
            output_sequence[i] = val
        else:
            output_sequence[i] = val
            # Cases:
            # 1. The previous segment was too short to count as a change
            # 2. The previous segment was long enough to count as a change
    return output_sequence
            
def find_threshold_segments(
        data: np.ndarray,
        thresholds: np.ndarray,
        debounce_length: int = 1,
) -> np.ndarray:
    bin_index = np.searchsorted(thresholds, data)
    return debounce_integer_sequence(bin_index, debounce_length)