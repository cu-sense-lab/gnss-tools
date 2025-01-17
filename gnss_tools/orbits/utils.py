"""
Author Brian Breitsch
Date: 2025-01-02
"""

import numpy as np
from typing import Tuple

def solve_kepler(
        M: float,
        e: float,
        E0: float = 0.0,
        tol: float = 1e-12,
        timeout: int = 200) -> float:
    """Given mean anomaly `M` and orbital eccentricity `e`, uses Newton's method
    to iteratively solve Kepler's equations for the eccentric anomaly.
    Iteration stops when the magnitude between successive values of eccentric
    anomaly `E` is less than `tol`, or when `timeout` iterations have occurred in
    which case an exception is thrown.
    
    Input:
        `M` -- mean anomaly as scalar or array of shape (N,)
        `e` -- eccentricity (scalar)
        `E0` -- initial guess for `E` as scalar or array of shape (N,)
        `tol` -- desired tolerance for `abs(E[k+1]-E[k]) < tol`
        `timeout` -- number of iterations to attempt before throwing timeout error
    Output:
        `E` -- the eccentric anomaly as scalar or array of shape (N,)
    Raises:
        exception on timeout
    """
    E = E0
    g = lambda E: E - e * np.sin(E) - M
    g_prime = lambda E: 1 - e * np.cos(E)
    for i in range(timeout):
        E_new = E - g(E) / g_prime(E)
        if np.all(np.abs(E_new - E) < tol):
            return E_new
        E = E_new
    raise Exception('Max number of iterations exceeded ({0}) when solving for Eccentric Anomaly'.format(timeout))

    
def compute_true_anomaly(
        a: float,
        e: float,
        M0: float,
        DT: float,
        mu: float) -> float:
    """Solve for true anomaly given Keplerian parameters and time delta
    ----------------------------------------------------------------------------
    Inputs:
    --------
    a -- semi-major axis [m]
    e -- orbit eccentricity
    M0 -- mean anomaly [rad] at some reference time "t0"
    DT -- array of time deltas [s] relative to reference time "t0".  Cartesian
        coordinate vectors `p` and `v` will be computed for each time delta.
    mu -- gravitational parameter (units must correspond to `a`)
    
    Returns:
    --------
    `nu` -- true anomaly
    
    Reference:  https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    """
    # Compute orbital period and mean motion
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    n = 2 * np.pi / T
    # Compute mean anomaly (M) and eccentric anomaly (E) arrays
    M_arr = (M0 + DT * n) % (2 * np.pi)
    E_arr = np.array([solve_kepler(M, e, tol=1e-12) for M in M_arr])
    # Compute true anomaly (nu) and orbital radius from central body (rc)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E_arr / 2), np.sqrt(1 - e) * np.cos(E_arr / 2))
    return nu

    
def Kepler_to_Cartesian(
        a: float, 
        e: float, 
        i: float, 
        Omega: float, 
        omega: float, 
        nu: float, 
        mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for Cartesian position/velocity given Keplerian parameters
    ----------------------------------------------------------------------------
    Inputs:
    --------
    a -- semi-major axis [m]
    e -- eccentricity
    i -- inclination [rad]
    Omega -- longitude of ascending node (LAN) [rad]
    omega -- argument of periapsis [rad]
    nu -- true anomaly [rad]
    mu -- gravitational parameter (units must correspond to `a`)
    
    Returns:
    --------
    `p`, `v` -- Cartesian 3-vectors for orbital position [m] and velocity [m/s]
        in central body's inertial frame
    
    Reference:  https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    and
    "Fundamentals of Astrodynamics and Applications" -- Vallado (Fourth Edition, 2013)
    Algorithm 10: COE2RV, p. 118
    """
    p = a * (1 - e**2)
    rc = p / (1 + e * np.cos(nu))
    # Compute perifocal position and velocity vectors
    orx = rc * np.cos(nu)
    ory = rc * np.sin(nu)
    orvx = -np.sqrt(mu / p) * np.sin(nu)
    orvy = np.sqrt(mu / p) * (e + np.cos(nu))
    # Transform from perifocal to central body's inertial frame
    cxx = np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.cos(i) * np.sin(Omega)
    cxy = -np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.cos(i) * np.sin(Omega)
    cyx = np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(i) * np.cos(Omega)
    cyy = np.cos(omega) * np.cos(i) * np.cos(Omega) - np.sin(omega) * np.sin(Omega)
    czx = np.sin(omega) * np.sin(i)
    czy = np.cos(omega) * np.sin(i)
    r = np.array([
        cxx * orx + cxy * ory,
        cyx * orx + cyy * ory,
        czx * orx + czy * ory,
    ]).T
    v = np.array([
        cxx * orvx + cxy * orvy,
        cyx * orvx + cyy * orvy,
        czx * orvx + czy * orvy,
    ]).T
    return r, v