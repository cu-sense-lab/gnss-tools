
import numpy as np
from typing import Optional
import scipy.interpolate

NDARRAY_FLOAT64 = np.ndarray[tuple[int], np.dtype[np.float64]]

def create_knotted_spline(
    epochs: NDARRAY_FLOAT64,
    values: NDARRAY_FLOAT64,
    # knot_epochs_arg: Optional[NDARRAY_FLOAT64] = None,
    knot_spacing: float,
    spline_k: int = 3,
    start_epoch_arg: Optional[float] = None,
    end_epoch_arg: Optional[float] = None,
    weights: Optional[NDARRAY_FLOAT64] = None,
) -> scipy.interpolate.BSpline:
    """
    Interpolates the given values at the specified epochs using a knotted spline.
    """

    epochs = epochs - epochs[0]
    start_epoch = start_epoch_arg if start_epoch_arg is not None else epochs[0]
    end_epoch = end_epoch_arg if end_epoch_arg is not None else epochs[-1]
    spline_knot0 = (
        start_epoch // knot_spacing
    ) * knot_spacing
    spline_knot1 = (
        end_epoch // knot_spacing + 1
    ) * knot_spacing
    if spline_knot0 + knot_spacing > spline_knot1:
        raise ValueError(
            "Spline knot spacing is too large for the given epochs."
        )
    spline_knots = np.r_[
        [epochs[0]] * (spline_k + 1),
        np.arange(spline_knot0, spline_knot1, knot_spacing),
        [epochs[-1]] * (spline_k + 1),
    ]
    # print(spline_knots)
    
    spline = scipy.interpolate.make_lsq_spline(
        epochs, values, t=spline_knots, k=spline_k, w=weights
    )
    return spline

def knotted_spline_interpolate(
    epochs: NDARRAY_FLOAT64,
    values: NDARRAY_FLOAT64,
    knot_spacing: float,
    spline_k: int = 3,
    start_epoch_arg: Optional[float] = None,
    end_epoch_arg: Optional[float] = None,
) -> NDARRAY_FLOAT64:
    """
    Interpolates the given values at the specified epochs using a knotted spline.
    """
    if len(epochs) < spline_k + 1:
        raise ValueError(
            "Not enough epochs to create a spline with the given degree."
        )
    
    epochs = epochs - epochs[0]  # Normalize epochs to start from 0
    spline = create_knotted_spline(
        epochs=epochs,
        values=values,
        knot_spacing=knot_spacing,
        spline_k=spline_k,
        start_epoch_arg=start_epoch_arg,
        end_epoch_arg=end_epoch_arg,
    )
    return spline(epochs)  # type: ignore