"""
Author: Brian Breitsch
Date: 2025-01-02
"""

import numpy as np

def cascading_outlier_mask(
        arr: np.ndarray,
        magnitude_start: float = 1e6,
        magntiude_end: float = 1e1,
        num_steps: int = 10
    ) -> np.ndarray:
    mask = np.ones(arr.shape[0], dtype=bool)
    for threshold in np.logspace(np.log(magnitude_start), np.log(magntiude_end), num_steps):
        mean = np.nanmean(arr[mask])
        mask = mask * (np.abs(arr - mean) < threshold)
        if not np.any(mask):
            break
    return mask