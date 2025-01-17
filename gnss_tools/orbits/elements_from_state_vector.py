"""
Author Brian Breitsch
Date: 2025-01-02
"""

from math import atan2, floor, fmod, isinf, isnan
from typing import Tuple

import numpy as np
# from numpy import arccos as acos
from numpy import dot
from numpy.linalg import norm
from scipy.constants import pi

import numba

# def elements_from_state_vector(
#         r: np.ndarray,
#         v: np.ndarray,
#         mu: float
#     ) -> Tuple[float, float, float, float, float, float]:
#     h = np.cross(r, v)
#     n = np.cross([0, 0, 1], h)

#     ev = 1 / mu * ((norm(v) ** 2 - mu / norm(r)) * r - dot(r, v) * v)

#     E = norm(v) ** 2 / 2 - mu / norm(r)

#     a = -mu / (2 * E)
#     e = norm(ev)

#     SMALL_NUMBER = 1e-15

#     # Inclination is the angle between the angular
#     # momentum vector and its z component.
#     i = np.arccos(h[2] / norm(h))

#     if abs(i) < SMALL_NUMBER:
#         # For non-inclined orbits, raan is undefined;
#         # set to zero by convention
#         raan = 0
#         if abs(e) < SMALL_NUMBER:
#             # For circular orbits, place periapsis
#             # at ascending node by convention
#             arg_pe = 0
#         else:
#             # Argument of periapsis is the angle between
#             # eccentricity vector and its x component.
#             arg_pe = np.arccos(ev[0] / norm(ev))
#     else:
#         # Right ascension of ascending node is the angle
#         # between the node vector and its x component.
#         raan = np.arccos(n[0] / norm(n))
#         if n[1] < 0:
#             raan = 2 * pi - raan

#         # Argument of periapsis is angle between
#         # node and eccentricity vectors.
#         arg_pe = np.arccos(dot(n, ev) / (norm(n) * norm(ev)))

#     if abs(e) < SMALL_NUMBER:
#         if abs(i) < SMALL_NUMBER:
#             # True anomaly is angle between position
#             # vector and its x component.
#             f = np.arccos(r[0] / norm(r))
#             if v[0] > 0:
#                 f = 2 * pi - f
#         else:
#             # True anomaly is angle between node
#             # vector and position vector.
#             f = np.arccos(dot(n, r) / (norm(n) * norm(r)))
#             if dot(n, v) > 0:
#                 f = 2 * pi - f
#     else:
#         if ev[2] < 0:
#             arg_pe = 2 * pi - arg_pe

#         # True anomaly is angle between eccentricity
#         # vector and position vector.
#         f = np.arccos(dot(ev, r) / (norm(ev) * norm(r)))

#         if dot(r, v) < 0:
#             f = 2 * pi - f

#     # semi-maj axis, eccentricity, inclination, right-angle of ascending node, arg. perigee, true anomaly
#     return a, e, i, raan, arg_pe, f



def elements_from_state_vector(
        r_arr: np.ndarray,  # (3,) or (N, 3)
        v_arr: np.ndarray,  # (3,) or (N, 3)
        mu: float,
        SMALL_NUMBER: float = 1e-15
) -> np.ndarray:  # (6,) or (N, 6)
    r_arr = r_arr.reshape((-1, 3))
    v_arr = v_arr.reshape((-1, 3))
    N = r_arr.shape[0]
    assert(v_arr.shape[0] == N)
    elements = np.zeros((N, 6))
    for i, (r, v) in enumerate(zip(r_arr, v_arr)):
        elements[i] = numba_elements_from_state_vector(r, v, mu, SMALL_NUMBER)
    return elements

@numba.njit
def numba_elements_from_state_vector(
        r: numba.float64[:],
        v: numba.float64[:],
        mu: float,
        SMALL_NUMBER: float
    ) -> Tuple[float, float, float, float, float, float]:
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)

    ev = 1 / mu * ((norm(v) ** 2 - mu / norm(r)) * r - dot(r, v) * v)

    E = norm(v) ** 2 / 2 - mu / norm(r)

    a = -mu / (2 * E)
    e = norm(ev)


    # Inclination is the angle between the angular
    # momentum vector and its z component.
    i = np.arccos(h[2] / norm(h))

    if abs(i) < SMALL_NUMBER:
        # For non-inclined orbits, raan is undefined;
        # set to zero by convention
        raan = 0
        if abs(e) < SMALL_NUMBER:
            # For circular orbits, place periapsis
            # at ascending node by convention
            arg_pe = 0
        else:
            # Argument of periapsis is the angle between
            # eccentricity vector and its x component.
            arg_pe = np.arccos(ev[0] / norm(ev))
    else:
        # Right ascension of ascending node is the angle
        # between the node vector and its x component.
        raan = np.arccos(n[0] / norm(n))
        if n[1] < 0:
            raan = 2 * pi - raan

        # Argument of periapsis is angle between
        # node and eccentricity vectors.
        arg_pe = np.arccos(dot(n, ev) / (norm(n) * norm(ev)))

    if abs(e) < SMALL_NUMBER:
        if abs(i) < SMALL_NUMBER:
            # True anomaly is angle between position
            # vector and its x component.
            f = np.arccos(r[0] / norm(r))
            if v[0] > 0:
                f = 2 * pi - f
        else:
            # True anomaly is angle between node
            # vector and position vector.
            f = np.arccos(dot(n, r) / (norm(n) * norm(r)))
            if dot(n, v) > 0:
                f = 2 * pi - f
    else:
        if ev[2] < 0:
            arg_pe = 2 * pi - arg_pe

        # True anomaly is angle between eccentricity
        # vector and position vector.
        f = np.arccos(dot(ev, r) / (norm(ev) * norm(r)))

        if dot(r, v) < 0:
            f = 2 * pi - f

    # semi-maj axis, eccentricity, inclination, right-angle of ascending node, arg. perigee, true anomaly
    return (a, e, i, raan, arg_pe, f)





# def elements_from_state_vector_2(
#         r_vec: np.ndarray,
#         v_vec: np.ndarray,
#         mu: float
# ) -> np.ndarray:
#     r = np.linalg.norm(r_vec)
#     v = np.linalg.norm(v_vec)
#     v_r = np.dot(r_vec / r, v_vec)
#     v_p = np.sqrt(v**2 - v_r**2)
#     h_vec = np.cross(r_vec, v_vec)
#     h = np.linalg.norm(h_vec)
#     i = np.arccos(h_vec[2] / h)
#     K = np.array((0, 0, 1))
#     N_vec = np.cross(K, h_vec)
#     N = np.linalg.norm(N_vec)
#     Omega = 2 * np.pi - np.arccos(N_vec[0] / N)
#     K = np.array((0, 0, 1))
#     N_vec = np.cross(K, h_vec)
#     N = np.linalg.norm(N_vec)
#     Omega = 2 * np.pi - np.arccos(N_vec[0] / N)

#     # [section-5]
#     e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
#     e = np.linalg.norm(e_vec)
#     omega = 2 * np.pi - np.arccos(np.dot(N_vec, e_vec) / (N * e))
#     nu = np.arccos(np.dot(r_vec / r, e_vec / e))

#     return h, e, i, Omega, omega, nu


def compute_true_anomaly_from_state_vectors_for_small_ecc(
        r: np.ndarray,
        v: np.ndarray,
) -> np.ndarray:
    # Does not work for small inclination
    r = r.reshape((-1, 3))
    v = v.reshape((-1, 3))
    h = np.cross(r, v, axis=-1)
    K = np.array([0, 0, 1])[None, :]
    n = np.cross(K, h, axis=-1)
    norm_n = np.linalg.norm(n, axis=-1)
    norm_r = np.linalg.norm(r, axis=-1)
    # would check norm_n or inclination here to determine whether to use following formula
    f = np.arccos(np.sum(n * r, axis=-1) / (norm_n * norm_r))
    # or this:
    # f = np.arccos(r[0] / norm(r))
    # mask = np.sum(r * v, axis=-1) > 0
    mask = r[:, 2] < 0
    f[mask] = 2 * np.pi - f[mask]
    return f