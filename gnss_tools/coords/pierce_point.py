
import numpy as np
from .utils import ecf2geo, local_enu, radius_of_curvature


def compute_earth_pierce_point(
        pos_0_ecf: np.ndarray,
        pos_1_ecf: np.ndarray,
        piercing_altitude: float,
) -> np.ndarray:
    """
    Compute the pierce point of a line through a shell at some altitude above the Earth's surface.
    
    Parameters
    ----------
    pos_0_ecf : np.ndarray
        The ECEF coordinates of the reference point.
    pos_1_ecf : np.ndarray
        The ECEF coordinates of the object point."
    piercing_altitude : float
        The altitude of the shell above the Earth's surface.
    
    Returns
    -------
    np.ndarray
        The ECEF coordinates of the pierce point.
    """
    pos_0_ecf = pos_0_ecf.reshape((-1, 3))
    pos_1_ecf = pos_1_ecf.reshape((-1, 3))
    N = pos_1_ecf.shape[0]
    if pos_0_ecf.shape[0] != 1:
        if pos_0_ecf.shape[0] != N:
            raise ValueError("Reference position should be a single point or have the same number of points as object positions")
        
    delta_pos_ecf = pos_1_ecf - pos_0_ecf
    delta_pos_ecf /= np.linalg.norm(delta_pos_ecf, axis=1, keepdims=True)

    pos_0_geo = ecf2geo(pos_0_ecf)
    lon = np.radians(pos_0_geo[:, 0])
    lat = np.radians(pos_0_geo[:, 1])

    # create the rotation matrix
    Rl = local_enu(lat, lon)
    enu = np.sum(Rl * delta_pos_ecf.T[None, :, :], axis=1).T
    e = enu[:, 0]
    n = enu[:, 1]
    u = enu[:, 2]
    r = np.sqrt(e**2 + n**2 + u**2)
    az = np.arctan2(e, n)
    el = np.arcsin(u / r)

    R = radius_of_curvature(lat, az)
    
    X = R * np.sin(el)
    Y = piercing_altitude**2 + 2 * R * piercing_altitude
    Z = np.sqrt(Y + X**2)
    delta_u = np.empty((N,))
    idx = el < 0
    delta_u[idx] = np.abs(-X[idx] - Z[idx])
    delta_u[~idx] = np.abs(-X[~idx] + Z[~idx])
        
    piercing_points_ecf = pos_0_ecf + delta_u[:, None] * delta_pos_ecf
    return piercing_points_ecf














# def compute_earth_piercing_point(
#         ref_pos_ecf: np.ndarray,  # (3,)
#         obj_pos_ecf: np.ndarray,  # (3,)
#         piercing_altitude: float,
#         max_iterations: int = 10,
#         tolerance: float = 1,
# ) -> np.ndarray:
#     """
#     ----------------------------------------------------------------------------
#     Given ECF positions of reference and object, computes the ECF position of
#     the object's piercing point through the Earth's atmosphere at the given altitude.

#     Inputs:
#     `ref_pos_ecf` -- ECF position of reference point
#     `obj_pos_ecf` -- ECF position of object
#     `piercing_altitude` -- altitude of piercing point

#     Returns:
#     `piercing_pos_ecf` -- ECF position of piercing point
#     """
#     ref_pos_ecf = ref_pos_ecf.reshape((-1, 3))
#     obj_pos_ecf = obj_pos_ecf.reshape((-1, 3))
#     if ref_pos_ecf.shape[0] == 1:
#         N = obj_pos_ecf.shape[0]
#         piercing_pos_ecf = np.zeros((N, 3))
#         for i in range(N):
#             piercing_pos_ecf[i] = _compute_earth_piercing_point_helper(
#                 ref_pos_ecf, obj_pos_ecf[i], piercing_altitude, max_iterations, tolerance
#             )
#         return piercing_pos_ecf
#     elif obj_pos_ecf.shape[0] == 1:
#         N = ref_pos_ecf.shape[0]
#         piercing_pos_ecf = np.zeros((N, 3))
#         for i in range(N):
#             piercing_pos_ecf[i] = _compute_earth_piercing_point_helper(
#                 ref_pos_ecf[i], obj_pos_ecf, piercing_altitude, max_iterations, tolerance
#             )
#         return piercing_pos_ecf
#     else:
#         raise ValueError("Either ref_pos_ecf or obj_pos_ecf must have shape (1, 3)")


# def _compute_earth_piercing_point_helper(
#         ref_pos_ecf: np.ndarray,  # (1, 3)
#         obj_pos_ecf: np.ndarray,  # (1, 3)
#         piercing_altitude: float,
#         max_iterations: int = 10,
#         tolerance: float = 1,
# ) -> np.ndarray:
#     pos_ecf = ref_pos_ecf.copy().reshape((1, 3))
#     obj_pos_ecf = obj_pos_ecf.reshape((1, 3))
#     assert(pos_ecf.shape[0] == 1)
#     assert(obj_pos_ecf.shape[0] == 1)
#     for i in range(max_iterations):
#         # compute unit vector from reference to object
#         unit_vec = obj_pos_ecf - pos_ecf
#         unit_vec /= np.linalg.norm(unit_vec, axis=1, keepdims=True)

#         # compute reference point's altitude and local radius of curvature
#         pos_geo, local_radius = coord_utils.ecf2geo(pos_ecf, return_ROC=True)
#         pos_altitude = pos_geo[:, 2]
#         # print(f" {pos_altitude}", end="")

#         # figure out projection of unit vector along "up" direction
#         rot_enu = coord_utils.local_enu(pos_geo[:, 1], pos_geo[:, 0])
#         # unit vec shape: (N, 3), rot_enu shape: (3, 3, N)
#         unit_vec_enu = (rot_enu[:, :, :] * unit_vec.T[None, :, :]).sum(axis=1).T
#         up_proj = unit_vec_enu[:, 2]

#         # compute signed difference between target altitude and reference altitude
#         altitude_delta = piercing_altitude - pos_altitude
#         if np.all(np.abs(altitude_delta)) < tolerance:
#             # print("X", end="")
#             break

#         # Generally, we want to move a distance of `altitude_delta` along the "up" direction
#         # which means moving roughly `altitude_delta / up_proj` distance along the unit vector
#         # However, we should also check if the ray will not intersect the sphere defined by
#         # the local radius of curvature.  In that case, we will return the tangent point on the
#         # sphere as the piercing point.

#         unit_delta = altitude_delta / up_proj
#         pos_ecf = pos_ecf + unit_vec * unit_delta

#         # TODO: check for sphere intersection; handle tangent point case and error codes
#     # print("")

#     return pos_ecf