"""
Author Brian Breitsch
Date: 2025-01-02
"""

import numba
import numpy as np


@numba.njit
def numba_array_lagrange_interpolation_float64(
    N: int,
    ydim: int,
    x: numba.float64[:],  # type: ignore
    y: numba.float64[:, :],  # type: ignore
    N_p: int,
    x_new: numba.float64[:],  # type: ignore
    y_new: numba.float64[:, :],  # type: ignore
    order: int,
    equality_tolerance: float = 1e-10,
    max_xdiff: float = 1e10,
    derivative: bool = False,
) -> int:
    """
    Numba helper to perform Lagrange interpolation of a 1D array.
    Uses barycentric form of Lagrange interpolation.
    See https://en.wikipedia.org/wiki/Lagrange_polynomial#Barycentric_form

    Parameters
    ----------
    x : numba.float64[:]
        The x-coordinates of the data points.
    y : numba.float64[:, :]
        The y-coordinates of the data points.
    x_new : numba.float64[:]
        The x-coordinates of the interpolated points.
    y_new : numba.float64[:, :]
        The interpolated y-coordinates of the x_new points.

    Returns
    -------
    error_flag : int

    """

    # Initially use first `order` elements
    ref_index = 0

    # Compute initial barycentric weights
    barycentric_weights = np.empty(order, dtype=np.float64)
    for j in range(order):
        barycentric_weights[j] = 1.0
        for m in range(order):
            if j != m:
                barycentric_weights[j] /= x[j] - x[m]

    weights_need_update: bool = False

    for output_index in range(N_p):

        # Check that we are using the closest `order` elements, starting from ref_index
        # If the closest elements are too far apart, output nan and update the reference index
        # if
        # if abs(x_new[output_index] - x[ref_index]) > max_xdiff or abs(x_new[output_index] - x[ref_index + order - 1]) > max_xdiff:
        #     xdiff_too_large = True
        if np.isnan(x_new[output_index]):
            y_new[output_index, :] = np.nan
            continue

        while abs(x_new[output_index] - x[ref_index + order]) < abs(
            x_new[output_index] - x[ref_index]
        ):
            ref_index += 1
            weights_need_update = True
            if ref_index + order == N:
                ref_index = N - order - 1
                break

        if (
            abs(x_new[output_index] - x[ref_index]) > max_xdiff
            or abs(x_new[output_index] - x[ref_index + order - 1]) > max_xdiff
        ):
            if x[ref_index + 1] < x_new[output_index]:
                ref_index += 1
                weights_need_update = True
            y_new[output_index, :] = np.nan
            continue

        if weights_need_update:
            for j in range(order):
                barycentric_weights[j] = 1.0
                for m in range(order):
                    x_delta = x[ref_index + j] - x[ref_index + m]
                    if j != m:
                        barycentric_weights[j] /= x_delta
        elif output_index > 0 and np.isnan(y_new[output_index - 1, 0]):
            # If the previous value was NaN and there was no weight update, then the next value will also be NaN
            y_new[output_index, :] = np.nan
            continue

        weights_need_update = False

        # If x_new[output_index] is equal to one of the x[ref_index + j], then y_new = y[ref_index + j]

        if derivative:
            # Derivative interpolation
            # First compute dlog(l(x)) = sum_{m=0}^k 1 / (x - x_m)
            # If we find x == x_j for some j, then L'(x) = L'j(x) y_j = y_j * (sum_{m \neq j} 1 / (x - x_m))
            product_x_delta = 1.0
            sum_x_delta_recip = 0.0
            x_j_equal_index = -1
            for j in range(order):
                x_delta = x_new[output_index] - x[ref_index + j]
                if abs(x_delta) < equality_tolerance:
                    x_j_equal_index = j
                    continue
                product_x_delta *= x_delta
                sum_x_delta_recip += 1.0 / x_delta

            value = np.zeros(ydim)
            if x_j_equal_index >= 0:
                # If x == x_j for some j, then we skipped x_j in
                # the loop above, and:
                for j in range(order):
                    x_delta = x_new[output_index] - x[ref_index + j]
                    if j != x_j_equal_index:
                        value[:] += product_x_delta * barycentric_weights[j] * y[ref_index + j, :] / x_delta
                    else:
                        value[:] += sum_x_delta_recip * y[ref_index + x_j_equal_index]
            else:
                # Otherwise, we now compute the derivative as L'(x) = sum_j L_j(x) * y_j * sum_{m neq j} 1 / (x - x_m)
                for j in range(order):
                    x_delta = x_new[output_index] - x[ref_index + j]
                    # Don't have to check for equality here, since we already did above
                    value += (
                        barycentric_weights[j]
                        * y[ref_index + j]
                        / x_delta
                        * (sum_x_delta_recip - 1.0 / x_delta)
                    )
                value[:] = product_x_delta * value
            y_new[output_index, :] = value
        else:
            # Normal interpolation
            # We can take a shortcut if x == x_j for some j
            value = np.zeros(ydim)
            product_x_delta = 1.0
            for j in range(order):
                x_delta = x_new[output_index] - x[ref_index + j]
                if abs(x_delta) < equality_tolerance:
                    value = y[ref_index + j]
                    product_x_delta = 1.0
                    break
                product_x_delta *= x_delta
                value[:] += barycentric_weights[j] * y[ref_index + j, :] / x_delta
            value[:] = product_x_delta * value
            y_new[output_index, :] = value[:]

    return 0


def compute_array_lagrange_interpolation(
    x_p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    equality_tolerance: float = 1e-10,
    max_xdiff: float = 1e10,
    derivative: bool = False,
) -> np.ndarray:
    """
    Efficiently compute Lagrange interpolation of a 1D array, assuming
    `x` and `x_p` are sorted in ascending order.

    Parameters
    ----------
    x_p : np.ndarray
        The x-coordinates of the interpolated points.
    x : np.ndarray
        The x-coordinates of the data points.
    y : np.ndarray
        The y-coordinates of the data points.
    order : int
        The order of the Lagrange interpolation.
    max_xdiff : float
        The maximum difference between x_p and x[ref_index] or x[ref_index + order - 1]
        before the interpolation is set to NaN.
    derivative : bool
        If True, compute and return the derivative of the interpolation polynomial.
        Note: the derivative estimation seems to be slightly less stable than the
        normal interpolation.  It has a more complicated formula when x is near x_j for some j.
        Use at your own risk.

    Returns
    -------
    y_p : np.ndarray
        The interpolated y-coordinates of the x_p points.

    """
    is_sorted = lambda a: np.all((a[:-1] <= a[1:]) | np.isnan(a[1:] - a[:-1]))
    if not is_sorted(x):
        raise ValueError("`x` must be sorted in ascending order.")
    if not is_sorted(x_p):
        raise ValueError("`x_p` must be sorted in ascending order.")
    if order < 2:
        raise ValueError("`order` must be at least 2.")
    if order > len(x):
        raise ValueError("`order` must be less than or equal to the length of `x`.")

    N = x.shape[0]
    N_p = x_p.shape[0]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim != 2:
        raise ValueError("`y` must have either 1 or 2 dimensions")
    _N, ydim = y.shape
    assert _N == N
    y_p = np.empty((N_p, ydim), dtype=np.float64)
    numba_array_lagrange_interpolation_float64(
        N, ydim, x, y, N_p, x_p, y_p, order, equality_tolerance, max_xdiff, derivative
    )
    if y_p.shape[1] == 1:
        y_p = y_p.squeeze()
    return y_p
