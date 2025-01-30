"""
coordinate_utils.py

Author Brian Breitsch
Date: 2025-01-02
"""

from typing import Optional, Tuple
import numpy as np
from gnss_tools.time.gmst import gpst2gmst, gpst2gmst_vec
from numpy.typing import NDArray


def make_3_tuple_array(arr: np.ndarray):
    """
    Reshapes ndarray so that it has dimensions (N,3)
    """
    arr = np.array(arr) if isinstance(arr, list) or isinstance(arr, tuple) else arr
    assert arr.ndim <= 2 and arr.size >= 3
    if arr.shape[0] == 3:
        arr = arr.reshape((1, 3)) if arr.ndim == 1 else arr.T
    assert arr.shape[1] == 3
    return arr


def dms2deg(arr: np.ndarray):
    """Converts degrees, minutes, seconds to degrees.
    Parameters
    ----------
    arr : list, tuple, or ndarray of shape (N,3) (or of length 3)
    The minutes/seconds should be unsigned but the degrees may be signed

    Returns
    -------
    (N, 3) array of degrees (or scalar in N == 1)
    """
    arr = make_3_tuple_array(arr)
    return (
        np.sign(arr[:, 0]) * (abs(arr[:, 0]) + arr[:, 1] / 60.0 + arr[:, 2] / 3600.0)
    ).squeeze()


def ecf2geo(
    x_ref: NDArray[np.float64],
    timeout: Optional[int] = None,
    return_ROC: bool = False,
    tolerance: float = 1e-10,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Converts ecf coordinates to geodetic coordinates,

    Parameters
    ----------
    x_ref : an ndarray of N ecf coordinates with shape (N,3).
    timeout : iterations will end when timeout goes to zero.
    return_ROC : whether to return optional output `N` (radius of curvature)

    Returns
    -------
    output : (N,3) ndarray
        geodetic coordinates in degrees and meters (lon, lat, alt)
    `N` : (optional) the radius of curvature, passed in as output parameter

    Notes
    -----
    >>> from numpy import array, radians
    >>> geo = array([27.174167, 78.042222, 0])  # lat, lon, alt
    >>> ecf = geo2ecf(radians(geo))
    >>> new_geo = ecf2geo(ecf)
    array([[             nan],
           [  7.08019709e+01],
           [ -6.37805436e+06]])
    >>> # [1176.45, 5554.887, 2895.397] << should be this
    >>> ecf2geo(array([
        [27.174167, 78.042222, 0],
        [39.5075, -84.746667, 0]])).reshaps((3,2))
    array([[             nan,              nan],
           [  7.08019709e+01,  -6.50058423e+01],
           [ -6.37805436e+06,  -6.37804350e+06]])
    [1176.45, 5554.887, 2895.397]
    [451.176, -4906.978, 4035.946]
    """
    x_ref = make_3_tuple_array(x_ref)
    # we = 7292115*1**-11 # Earth angular velocity (rad/s)
    # c = 299792458    # Speed of light in vacuum (m/s)
    rf = 298.257223563  # Reciprocal flattening (1/f)
    a = 6378137.0  # Earth semi-major axis (m)
    b = a - a / rf  # Earth semi-minor axis derived from f = (a - b) / a
    if x_ref.shape == (3,):
        x_ref = x_ref.reshape((1, 3))
    x = x_ref[:, 0]
    y = x_ref[:, 1]
    z = x_ref[:, 2]

    # We must iteratively derive N
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    h = np.zeros(len(lat))
    # h[abs(lat) > tolerance] = z / np.sin(lat[abs(lat) > tolerance])
    # h = z / np.sin(lat) if numpy.any(abs(lat) > tolerance) else 0 * lat
    d_h = 1.0
    d_lat = 1.0
    while (d_h > tolerance) and (d_lat > tolerance):
        N = a**2 / (np.sqrt(a**2 * np.cos(lat) ** 2 + b**2 * np.sin(lat) ** 2))
        N1 = N * (b / a) ** 2

        temp_h = np.sqrt(x**2 + y**2) / np.cos(lat) - N
        temp_lat = np.arctan2(z / (N1 + h), np.sqrt(x**2 + y**2) / (N + h))
        d_h = np.max(np.absolute(h - temp_h))
        d_lat = np.max(np.absolute(lat - temp_lat))

        h = temp_h
        lat = temp_lat
        if timeout is not None:
            timeout -= 1
            if timeout <= 0:
                break

    lon = np.arctan2(y, x)

    lon = np.degrees(lon)
    lat = np.degrees(lat)

    geo = np.column_stack((lon, lat, h)).squeeze()
    if return_ROC:
        return geo, N
    return geo


def local_enu(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    Rl = np.array(
        [
            [
                -np.sin(lon),
                np.cos(lon),
                np.zeros((lat.shape[0],)) if isinstance(lat, np.ndarray) else 0,
            ],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )
    return Rl


def ecf2enu(
    x_ref: NDArray[np.float64],
    x_obj: NDArray[np.float64],
    timeout: Optional[int] = None,
):
    """Converts satellite ecf coordinates to user-relative ENU coordinates.

    Parameters
    ----------
    x_ref : ndarray of shape (3,)
        observer ecf coordinate
    x_obj : ndarray of shape(N,3)
        object ecf coordinates
    timeout : int
        timeout integer, passed to ecf2geo

    Returns
    -------
    output : ndarray of shape(N,3)
        The east-north-up coordinates
    """
    x_ref = make_3_tuple_array(x_ref)
    x_obj = make_3_tuple_array(x_obj)
    # get the lat and lon of the user position
    geo = ecf2geo(x_ref, timeout=timeout)
    geo = make_3_tuple_array(geo)
    lon = np.radians(geo[:, 0])
    lat = np.radians(geo[:, 1])
    N = geo.shape[0]

    # create the rotation matrix
    Rl = local_enu(lat, lon)
    dx = x_obj - x_ref
    return np.sum(Rl * dx.T[None, :, :], axis=1).T  # sum across columns
    # return Rl.dot(dx.T).T.squeeze()


def enu2sky(enu: NDArray[np.float64]):
    """Converts local East-North-Up coordinates to Sky coordinates (azimuth, elevation, radius)

    Parameters
    ----------
    enu : ndarray of shape(N,3)
        ENU coordinates

    Returns
    -------
    output : ndarray of shape(N,3)
        The sky coordinates
        (azimuth, elevation, radius)  in degrees and meters
    """
    enu = make_3_tuple_array(enu)
    e = enu[:, 0]
    n = enu[:, 1]
    u = enu[:, 2]
    az = np.arctan2(e, n)
    r = np.sqrt(e**2 + n**2 + u**2)
    el = np.arcsin(u / r)
    return np.column_stack((np.degrees(az), np.degrees(el), r)).squeeze()


def xyz2sky(xyz: NDArray[np.float64]):
    """Converts XYZ coordinates to azimuth, elevation, and radius
    The Z axis is defined to be the direction of the observer's zenith.
    Azimuth is measured counterclockwise from the X axis, which is
    defined to be the direction of the observer's meridian.

    Parameters
    ----------
    xyz : ndarray of shape(N,3)
        XYZ coordinates

    Returns
    -------
    output : ndarray of shape(N,3)
        The spherical coordinates
        (azimuth, elevation, radius) in degrees and meters
    """
    # xyz = make_3_tuple_array(xyz)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    az = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(z / r)
    return np.stack((np.degrees(az), np.degrees(el), r), axis=-1)


def sky2enu(sky: NDArray[np.float64]):
    """Converts local Sky coordinates back to local East-North-Up coordinates."""
    sky = make_3_tuple_array(sky)
    az, el, r = sky[:, 0], sky[:, 1], sky[:, 2]
    x = r * np.array([1, 0, 0])
    theta = np.radians(90 - az)
    phi = np.radians(el)
    e = r * np.cos(theta) * np.cos(phi)
    n = r * np.sin(theta) * np.cos(phi)
    u = r * np.sin(phi)
    return np.column_stack((e, n, u)).squeeze()


def ecf2sky(
    x_ref: NDArray[np.float64],
    x_obj: NDArray[np.float64],
    timeout: Optional[int] = None,
):
    """Converts user and satellite ecf coordinates to azimuth and elevation
    from user on Earth, by first computing their relative ENU coordinates.  See `enu2sky`.

    Parameters
    ----------
    x_ref : ndarray of shape (3,)
        observer coordinate
    xs : ndarray of shape(N,3)
        object coordinates
    timeout: timout integer--passed to ecf2enu

    Returns
    -------
    output : ndarray of shape(N,3)
        The objects' sky coordinatescoordinates
        (azimuth, elevation, radius)  in degrees and meters
    """
    enu = ecf2enu(x_ref, x_obj, timeout=timeout)
    return enu2sky(enu)


def local_radius_of_curvature(geo: NDArray[np.float64]):
    """Returns local radius of curvature given geodetic coordinates."""
    geo = make_3_tuple_array(geo)
    a = 6378137.0  # Earth semi-major axis (m)
    rf = 298.257223563  # Reciprocal flattening (1/f)
    b = a * (rf - 1) / rf  # Earth semi-minor axis derived from f = (a - b) / a
    lon = np.radians(geo[:, 0])
    lat = np.radians(geo[:, 1])
    h = geo[:, 2]

    N = a**2 / np.sqrt(a**2 * np.cos(lat) ** 2 + b**2 * np.sin(lat) ** 2)
    N1 = N * (b / a) ** 2
    return N1


def geo2ecf(geo):
    """Converts geodetic coordinates to ecf coordinates

    Parameters
    ----------
    geo : ndarray of shape (N,3)
        geodetic coordinates (lon, lat, alt) in degrees and meters above WGS84 ellipsoid

    Returns
    -------
    output : ndarray of shape(N,3)
        ecf coordinates

    Notes
    -----
    """
    geo = make_3_tuple_array(geo)
    a = 6378137.0  # Earth semi-major axis (m)
    rf = 298.257223563  # Reciprocal flattening (1/f)
    b = a * (rf - 1) / rf  # Earth semi-minor axis derived from f = (a - b) / a
    lon = np.radians(geo[:, 0])
    lat = np.radians(geo[:, 1])
    h = geo[:, 2]

    N = a**2 / np.sqrt(a**2 * np.cos(lat) ** 2 + b**2 * np.sin(lat) ** 2)
    N1 = N * (b / a) ** 2

    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N1 + h) * np.sin(lat)

    x_ref = np.column_stack((x, y, z))
    return x_ref.squeeze()


def geo2sky(geo_ref: NDArray[np.float64], geo_obj: NDArray[np.float64]):
    """Converts object geodetic coordinates to azimuth and elevation from
    reference geodetic coordinates on Earth by first computing their relative
    Sky coordinates.  See `enu2sky`.

    Parameters
    ----------
    geo_ref : ndarray of shape (3,)
        geodetic (lon, lat, alt) coordinates of observer
    geo_obj : ndarray of shape (N,3)
        geodetic (lon, lat, alt) coordinates of object

    Returns
    -------
    output : ndarray of shape(N,3)
        sky coordinates (azimuth, elevation, radius) in degrees and meters
    """
    geo_ref = make_3_tuple_array(geo_ref)
    geo_obj = make_3_tuple_array(geo_obj)
    x_ref = geo2ecf(geo_ref)
    x_obj = geo2ecf(geo_obj)
    return ecf2sky(x_ref, x_obj)


def eci2ecf(
    time_gpst_arr: np.ndarray,
    r_eci: np.ndarray,
    v_eci: Optional[np.ndarray] = None,
    gtime_dtype: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Converts ECI coordinates to ECF coordinates given the set of coordinates and coordinate times

    Parameters
    ----------
    time : array of shape N; the time coordinates
    r_eci : array of shape (N,3); the ECI position vectors
    v_eci : (optional) array of shape (N,3); the ECI velocity vectors

    Returns
    -------
    r_ecf : array of shape (N,3); the ECF coordinate vectors
    v_ecf : (only if `v_eci` is specified) array of shape (N,3); the ECF velocity vectors

    Notes
    -----
    Time of 2000 January 1, 12 UTC is (in GPS seconds) 630763213.0
    See: http://physics.stackexchange.com/questions/98466/radians-to-rotate-earth-to-match-eci-lat-lon-with-ecf-lat-lon
    """

    gmst = gpst2gmst_vec(time_gpst_arr, gtime_dtype=gtime_dtype)
    gmst_hour = gmst % 24
    theta = -2 * np.pi * gmst_hour / 24
    shp = theta.shape

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), np.zeros(shp)],
            [np.sin(theta), np.cos(theta), np.zeros(shp)],
            [np.zeros(shp), np.zeros(shp), np.ones(shp)],
        ]
    )
    r_ecf = np.sum(R * r_eci.T[None, :, :], axis=1).T

    if v_eci is not None:
        dtheta = 7.29211585275553e-005  # Earth rotation [rad/s]
        Rdot = (
            np.array(
                [
                    [-np.sin(theta), -np.cos(theta), np.zeros(shp)],
                    [np.cos(theta), -np.sin(theta), np.zeros(shp)],
                    [np.zeros(shp), np.zeros(shp), np.zeros(shp)],
                ]
            )
            * dtheta
        )

        v_ecf = (
            np.sum(R * v_eci.T[None, :, :], axis=1).T
            + np.sum(Rdot * r_eci.T[None, :, :], axis=1).T
        )

        return r_ecf, v_ecf

    else:
        return r_ecf


def ecf2eci(
    time_gpst_arr: np.ndarray,
    r_ecf: np.ndarray,
    v_ecf: Optional[np.ndarray] = None,
    gtime_dtype: bool = False,
) -> np.ndarray:
    """Converts ecf coordinates to eci coordinates given
    the time or times for said coordinates

    Parameters
    ----------
    time : array of shape N; the time coordinates
    r_ecf : array of shape (N,3); the ECF position coordinates
    v_ecf : (optional) array of shape (N,3); the ECF velocity

    Returns
    -------
    r_eci : array of shape (N,3); the ECI position coordinates
    v_eci : array of shape (N,3); the ECI velocity coordinates (only returned if `v_ecf` is not None)

    Notes
    -----
    See: http://physics.stackexchange.com/questions/98466/radians-to-rotate-earth-to-match-eci-lat-lon-with-ecf-lat-lon
    Time of 2000 January 1, 12 UTC is (in GPS seconds) 630763213.0
    """
    gmst = gpst2gmst_vec(time_gpst_arr, gtime_dtype=gtime_dtype)
    gmst_hour = gmst % 24
    theta = 2 * np.pi * gmst_hour / 24
    shp = theta.shape

    # NOTE: super important to use array and not asarray for matrices
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), np.zeros(shp)],
            [np.sin(theta), np.cos(theta), np.zeros(shp)],
            [np.zeros(shp), np.zeros(shp), np.ones(shp)],
        ]
    )
    r_eci = np.sum(R * r_ecf.T[None, :, :], axis=1).T  # sum across columns

    if v_ecf is not None:
        dtheta = 7.29211585275553e-005  # Earth rotation [rad/s]
        Rdot = (
            np.array(
                [
                    [-np.sin(theta), -np.cos(theta), np.zeros(shp)],
                    [np.cos(theta), -np.sin(theta), np.zeros(shp)],
                    [np.zeros(shp), np.zeros(shp), np.zeros(shp)],
                ]
            )
            * dtheta
        )

        v_eci = (
            np.sum(R * v_ecf.T[None, :, :], axis=1).T
            + np.sum(Rdot * r_ecf.T[None, :, :], axis=1).T
        )

        return r_eci, v_eci

    else:
        return r_eci


def rotate_pos_ecf(pos_ecf: np.ndarray, tau: float):
    """
    Rotates ECF position around the polar axis by `tau` times Earth's rotation rate

    Parameters
    ----------
    r_ecf : array of shape (3,); the ECF position coordinates
    tau : float; the amount of time to rotate by

    Returns
    -------
    r_rot : array of shape (3,); the rotated position coordinates

    """
    Omega_E_dot = 7.2921151467e-5
    theta = Omega_E_dot * tau
    if np.isscalar(theta):
        theta = np.array([theta])
    pos_ecf = np.array(pos_ecf).reshape((-1, 3))
    Z = np.zeros(len(theta))
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), Z],
            [np.sin(theta), np.cos(theta), Z],
            [Z, Z, 1 + Z],
        ]
    )
    pos_rot: np.ndarray = (R * pos_ecf.T[None, :, :]).sum(axis=1).T
    return pos_rot.squeeze()
