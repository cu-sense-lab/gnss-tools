from typing import List, Tuple
import numpy as np

def get_merged_epoch_array(*epoch_arrays) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Get a merged epoch array from multiple arrays of epochs.

    Returns:
    - unique epochs
    - list of indices (into unique epoch array) for each input array
    """
    unique_epochs, inverse_indices = np.unique(
        np.concatenate(epoch_arrays), return_inverse=True
    )
    split_sizes = [len(epoch_array) for epoch_array in epoch_arrays]
    return unique_epochs, np.split(inverse_indices, np.cumsum(split_sizes[:-1]))
