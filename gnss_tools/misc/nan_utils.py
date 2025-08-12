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