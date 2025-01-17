"""
Miscellaneous utilities for working with GPS ephemerides

Author: Brian Breitsch
Date: 2023-03-07
"""

from typing import Tuple
import numpy as np
from ..orbits.utils import solve_kepler
from ..time.gtime import GTime
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OrbitalParams_GPS_LNAV:
    """
    Orbital Parameters corresponding to GPS LNAV ephemeris
    i: float -- orbital inclination [rad]
    r: float -- orbital radius [meters]
    Omega: float -- longitude of ascending node [rad]
    Phi: float -- argument of latitude [rad]
    n: float -- mean motion [rad/s]
    E: float -- eccentric anomaly [rad]
    """
    i: float
    r: float
    Omega: float
    Phi: float
    n: float
    E: float


def compute_gps_orbital_params_from_ephemeris(
        TOW_GPS: GTime,  # GPS TOW for which to compute the orbital parameters
        a: float,  # semi-major axis [meters]
        e: float,  # orbital eccentricity
        i0: float,  # orbital inclination at reference epoch [rad]
        iDot: float,  # rate of inclincation perturbation [rad/s]
        omega: float,  # argument of perigee? [rad]
        M0: float,  # mean anomaly at reference epoch [rad]
        TOE: float,  # GPS TOW of ephemeris applicability
        Omega0: float,  # Longitude of ascending node [ECF rad]
        OmegaDot: float,  # rate of perturbation of Long. of ascending node [rad/s]
        deln: float,  # mean motion perturbation [rad/s]
        Cus: float,  # sine harmonic correction to arg. of latitude
        Cuc: float,  # cosine harmonic correction to arg. of latitude
        Crs: float,  # sine harmonic correction to orbital radius
        Crc: float,  # cosine harmonic correction to orbital radius
        Cis: float,  # sine harmonic correction to orbital inclination
        Cic: float,  # cosine harmonic correction to orbital inclination
        ) -> Tuple[float, float, float, float, float, float, float, float]:
    '''
    Given GPS ephemeris parameters for a particular SV and a GPS TOW,
    extracts the necessary Keplerian parameters, applies precise corrections,
    and returns the orbital parameters necessary to compute satellite position
    in ECF coordinates.

    The first three parameters (semi-major axis, eccentricity, inclination) are
    normal Keplerian elements.
    
    The fourth parameter is the longitude of ascending node `Omega`, which
    includes Earth rotation so that the an orbit estimated using these
    parameters will be in the Earth-centered fixed coordinate system (ECF).
    
    Note that, because ECF is a rotating coordinate frame, `Omega` will depend
    on time `t`, as will the true anomaly `nu`.
    
    The last parameter is the argument of latitude `Phi` which accounts for
    perigee plus true anomaly.
    ----------------------------------------------------------------------------
    Input:
        TOW_GPS: GTime -- time for which to compute orbital parameters
            NOTE: this must correspond to the same TOW as TOE in the ephemeris
        a: float -- semi-major axis [meters]
        e: float -- orbital eccentricity
        i0: float -- orbital inclination at reference epoch [rad]
        iDot: float -- rate of inclincation perturbation [rad/s]
        omega: float -- argument of perigee? [rad]
        M0: float -- mean anomaly at reference epoch [rad]
        TOE: float -- GPS TOW of ephemeris applicability
        Omega0: float -- Longitude of ascending node [ECF rad]
        OmegaDot: float -- rate of perturbation of Long. of ascending node [rad/s]
        deln: float -- mean motion perturbation [rad/s]
        Cus: float -- sine harmonic correction to arg. of latitude
        Cuc: float -- cosine harmonic correction to arg. of latitude
        Crs: float -- sine harmonic correction to orbital radius
        Crc: float -- cosine harmonic correction to orbital radius
        Cis: float -- sine harmonic correction to orbital inclination
        Cic: float -- cosine harmonic correction to orbital inclination
    Output:
        OrbitalParams_GPS_LNAV struct
    '''
    pi = 3.1415926535898  # TODO: why doesn't this show up in equations?  Is that a problem?
    mu_E = 3.986005e14
    Omega_E_dot = 7.2921151467e-5
    # next compute the time of week `TOW` corresponding to the GPST input `t`
    # and compute `Dt` -- the difference between `TOW` and the time of ephemeris
    Dt = TOW_GPS.add_integer_seconds(-TOE).to_float()
    # next, compute `Omega` -- the so-called longitude of ascending node, or the RAAN
    # in the ECF coordinate frame
    Omega_0 = Omega0
    Omega_dot = OmegaDot
    Omega = Omega_0 - TOW_GPS.to_float() * Omega_E_dot + Dt * Omega_dot  # longitude of ascending node
    # compute mean motion, mean anomaly, and eccentric anomaly
    n0 = np.sqrt(mu_E / a**3)       # intitial mean motion
    n = n0 + deln                   # corrected mean motion
    M = M0 + n * Dt                 # mean anomaly
    E = solve_kepler(M, e)          # eccentric anomaly
    # compute true anomaly
    nu = np.arctan2(np.sqrt(1. - e**2) * np.sin(E), np.cos(E) - e)  # true anomaly
    # compute argument of latitude, then compute the secord-order harmonic perturbation
    # correction terms for argument of latitude, radius, and orbital inclination
    Phi = nu + omega  # argument of latitude
    du = Cus * np.sin(2 * Phi) + Cuc * np.cos(2 * Phi)
    dr = Crs * np.sin(2 * Phi) + Crc * np.cos(2 * Phi)
    di = Cis * np.sin(2 * Phi) + Cic * np.cos(2 * Phi)
    # correct arg. of latitude, orbital radius, and inclination
    Phi = Phi + du  # argument of latitude (corrected)
    r = a * (1. - e * np.cos(E)) + dr  # orbital radius (corrected)
    i = i0 + di + iDot * Dt  # inclination (corrected)
    return OrbitalParams_GPS_LNAV(i, r, Omega, Phi, n, E)


def compute_ecf_position_from_orbital_parameters(
        i: float,
        r: float,
        Omega: float,
        Phi: float) -> Tuple[float, float, float]:
    '''
    Computes and returns the satellite position given a set of orbital parameters
    obtained from GPS ephemeris.
    ----------------------------------------------------------------------------
    Inputs:
        i - orbital inclination [rad]
        r - orbital radius [meters]
        Omega - longitude of ascending node [rad]
        Phi - argument of latitude [rad]
    '''
    x_orb, y_orb = r * np.cos(Phi), r * np.sin(Phi)  # compute x, y in orbital plane
    x_ecf = x_orb * np.cos(Omega) - y_orb * np.sin(Omega) * np.cos(i)  # transform from orbital system to ECF system
    y_ecf = x_orb * np.sin(Omega) + y_orb * np.cos(Omega) * np.cos(i)
    z_ecf = y_orb * np.sin(i)
    return (x_ecf, y_ecf, z_ecf)


def compute_ecf_velocity_from_orbital_parameters(
        a: float,
        e: float,
        i: float,
        Omega: float,
        n: float,
        E: float) -> Tuple[float, float, float]:
    '''
    Computes and returns the satellite velocity given an appropriate set of
    orbital parameters obtained from GPS ephemeris. (See e.g. `compute_gps_orbital_parameters_from_ephemeris`)
    ----------------------------------------------------------------------------
    Input:
        e - orbital eccentricity
        i - orbital inclination [rad]
        Omega - longitude of ascending node [rad]
        n - mean motion [rad/s]
        E - eccentric anomaly [rad]
    '''
    v_x_orb = n * a * np.sin(E) / (1. - e * np.cos(E))
    v_y_orb = -n * a * np.sqrt(1. - e**2) * np.cos(E) / (1. - e * np.cos(E))
    v_x_ecf = v_x_orb * np.cos(Omega) - v_y_orb * np.sin(Omega) * np.cos(i)  # transform from orbital system to ECF system
    v_y_ecf = v_x_orb * np.sin(Omega) + v_y_orb * np.cos(Omega) * np.cos(i)
    v_z_ecf = v_y_orb * np.sin(i)
    return (v_x_ecf, v_y_ecf, v_z_ecf)


def compute_gps_transmitter_clock_bias(
        TOW_GPS: GTime,
        TOC: int,
        af0: float,
        af1: float,
        af2: float) -> float:
    '''Given the ephemeris clock and orbital parameters, computes the satellite transmitter clock bias.
    Note the eccentric anomaly `E` is computed in (e.g.) `compute_gps_orbital_parameters_from_ephemeris`

    TODO: TOC is in `epoch` of RINEX Nav files: they need to be converted to TOW integer
    sat_time_correction_epoch = np.array([(e - GPS_EPOCH.replace(tzinfo=None)).total_seconds() for e in eph.epoch]])    
    dt = transmit_time_gpst - sat_time_correction_epoch
    ----------------------------------------------------------------------------
    Input:
        TOW_GPS: GTime -- GPS system TOW for which to compute sat clock correction
            NOTE: must be relative to same week number as TOC
        TOC: int -- "time of clock" [GPS TOW seconds]
        af0: float -- sat clock bias at TOC
        af1: float -- sat clock drift at TOC
        af2: float -- sat clock acceleration at TOC
    Returns:
        `clock_bias_seconds` -- the satellite clock bias in seconds
    '''
    dt = TOW_GPS.add_integer_seconds(-TOC).to_float()
    clock_bias_seconds = af0 + af1 * dt + af2 * dt**2
    return clock_bias_seconds


def compute_gps_transmitter_clock_bias_with_relativity_correction(
        TOW_GPS: GTime,
        TOC: int,
        af0: float,
        af1: float,
        af2: float,
        a: float,
        e: float,
        E: float) -> float:
    '''Given the ephemeris clock and orbital parameters, computes the satellite transmitter clock bias.
    Note the eccentric anomaly `E` is computed in (e.g.) `compute_gps_orbital_parameters_from_ephemeris`

    TODO: TOC is in `epoch` of RINEX Nav files:
    sat_time_correction_epoch = np.array([(e - GPS_EPOCH.replace(tzinfo=None)).total_seconds() for e in eph.epoch]])    
    dt = transmit_time_gpst - sat_time_correction_epoch
    ----------------------------------------------------------------------------
    Input:
        TOW_GPS: GTime -- GPS system TOW for which to compute sat clock correction
        TOC: int -- "time of clock" [GPS TOW seconds]
        af0: float -- sat clock bias at TOC
        af1: float -- sat clock drift at TOC
        af2: float -- sat clock acceleration at TOC
        a: float -- orbital semi-major axis [meters] (used for relativity correction)
        e: float -- orbital eccentricity (used for relativity correction)
        E: float -- orbital eccentric anomaly (used for relativity correction)
    '''
    dt = TOW_GPS.add_integer_seconds(-TOC).to_float()
    clock_bias = af0 + af1 * dt + af2 * dt**2 - 4.442807633e-10 * e * np.sqrt(a) * np.sin(E)
    return clock_bias
