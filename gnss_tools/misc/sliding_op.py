import numpy as np
from typing import Callable, Optional

def compute_sliding_op(
    value_arr: np.ndarray, window_length: int, op: Callable[[np.ndarray], np.ndarray],
    centered: bool = True
) -> np.ndarray:
    # window_shape = (1,) * (value_arr.ndim - 1) + (window_length,)
    window_shape = (window_length,)
    window_axis = value_arr.ndim - 1
    # print(window_shape, window_axis)
    value_sliding_window_view = np.lib.stride_tricks.sliding_window_view(
        value_arr, window_shape, axis=window_axis
    )
    # print("Sliding window view shape:", value_sliding_window_view.shape)
    output = op(value_sliding_window_view)
    # print("Output shape:", output.shape)
    result = np.zeros_like(value_arr)
    if centered:
        # Center the output in the result array
        i0 = window_length // 2
        i1 = i0 + output.shape[-1]
        result[..., :i0] = output[..., 0:1]
        result[..., i0:i1] = output
        result[..., i1:] = output[..., -1:]
    else:
        # Align full windows to start of the result array
        result[..., :output.shape[-1]] = output
    return result