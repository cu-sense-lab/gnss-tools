import numpy as np
from scipy.signal import detrend

def nan_scipy_detrend(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Detrend a array along a specified axis, ignoring NaN values.
    Uses scipy's detrend function.
    """
    # Create a mask for non-NaN values
    mask = ~np.isnan(x)
    result = np.empty_like(x)
    # Detrend along the specified axis
    result[mask] = detrend(x[mask], axis=axis)
    # Fill NaN values with NaN
    result[~mask] = np.nan
    return result

def nan_endpoint_detrend(x: np.ndarray) -> np.ndarray:
    """
    Perform endpoint detrend on a 1D array, ignoring NaN values.
    """
    if x.ndim != 1:
        raise ValueError("Input array must be 1D.")
    # Create a mask for non-NaN values
    mask = ~np.isnan(x)
    if np.sum(mask) < 2:
        return x  # Not enough data to detrend
    # Get the first and last non-NaN values and their indices
    first_idx = np.where(mask)[0][0]
    last_idx = np.where(mask)[0][-1]
    first_val = x[first_idx]
    last_val = x[last_idx]
    # Create a linear trend
    slope = (last_val - first_val) / (last_idx - first_idx)
    trend = slope * (np.arange(len(x)) - first_idx) + first_val
    # Detrend the data
    y_detrended = x - trend
    return y_detrended

def nan_polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    """
    Fit a polynomial of degree deg to the data (x, y), ignoring NaN values.
    """
    # Create a mask for non-NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    # Fit the polynomial
    coeffs = np.polyfit(x[mask], y[mask], deg)
    return coeffs

def nan_polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial with coefficients coeffs at the points x, ignoring NaN values.
    """
    # Create a mask for non-NaN values
    mask = ~np.isnan(x)
    # Evaluate the polynomial
    result = np.empty_like(x)
    result[mask] = np.polyval(coeffs, x[mask])
    # Fill NaN values with NaN
    result[~mask] = np.nan
    return result

def nan_polydetrend(x: np.ndarray, y: np.ndarray, deg: int = 1) -> np.ndarray:
    """
    Perform polynomial detrend a array along a specified axis, ignoring NaN values.
    """
    # Fit the polynomial
    coeffs = nan_polyfit(x, y, deg)
    # Evaluate the polynomial
    trend = nan_polyval(coeffs, x)
    # Detrend the data
    y_detrended = y - trend
    return y_detrended

def nan_filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    """
    Apply a filter to the data x, ignoring NaN values.
    """
    import scipy.signal
    # Create a mask for non-NaN values
    mask = ~np.isnan(x)
    # Apply the filter
    result = np.empty_like(x)
    result[mask] = scipy.signal.filtfilt(b, a, x[mask])
    # Fill NaN values with NaN
    result[~mask] = np.nan
    return result

def nan_unwrap(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """
    Unwrap the phase of the data x, ignoring NaN values.
    """
    # Create a mask for non-NaN values
    mask = ~np.isnan(x)
    # Unwrap the phase
    result = np.empty_like(x)
    result[mask] = np.unwrap(x[mask], period=period)
    # Fill NaN values with NaN
    result[~mask] = np.nan
    return result

def interpolate_nans(x: np.ndarray, method: str = 'linear', axis: int = -1) -> np.ndarray:
    """
    Interpolate the nan values in x.
    Overwrites NaN values with interpolated values.
    """
    if x.ndim != 1:
        shape = x.shape
        x = np.moveaxis(x, axis, -1).reshape(-1, shape[axis])
        for i in range(x.shape[0]):
            x[i, :] = interpolate_nans(x[i, :], method=method, axis=-1)
        x = x.reshape(*shape[:-1], shape[axis])
        x = np.moveaxis(x, -1, axis)
        return x

    from scipy.interpolate import interp1d
    # Create a mask for non-NaN values
    mask = ~np.isnan(x)
    if np.sum(mask) < 2:
        return x  # Not enough data to interpolate
    # Create an interpolator
    interpolator = interp1d(np.where(mask)[0], x[mask], kind=method, bounds_error=False, fill_value="extrapolate")
    # Interpolate the data
    nan_indices = np.where(np.isnan(x))[0]
    x[nan_indices] = interpolator(nan_indices)
    return x