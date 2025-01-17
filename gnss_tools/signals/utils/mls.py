import numpy as np
from typing import List


def generate_mls(
    register_length: int, feedback_taps: List[int], output_taps: List[int]
):
    """Generates Maximum Length Sequence (length-(2**N - 1) binary sequence) for the given feedback and output taps.

    Parameters
    ----------
    `register_length` : number of bits to use in the feedback register
    `feedback_taps` : list of integer feedback tap indices
    `output_taps` : list of integer output tap indices

    Returns
    -------
    output : NDArray of shape (2**register_length - 1,)
        the code sequence
    """
    shift_register = np.ones((register_length,), dtype=np.int8)
    seq = np.zeros((2**register_length - 1,), dtype=np.int8)
    for i in range(2**register_length - 1):
        seq[i] = np.sum(shift_register[output_taps]) % 2
        first = np.sum(shift_register[feedback_taps]) % 2
        shift_register[1:] = shift_register[:-1]
        shift_register[0] = first
    return seq
