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
    # Default base of logspace is 10 (for some dumb reason...) so that's why log10 used below...
    for threshold in np.logspace(np.log10(magnitude_start), np.log10(magntiude_end), num_steps):
        mean = np.nanmean(arr[mask])
        mask = mask * (np.abs(arr - mean) < threshold)
        if not np.any(mask):
            break
    return mask