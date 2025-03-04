import numpy as np
from typing import Tuple

# Quaternion utilitites
# Note: these functions assume normalized quaternion vectors with shape (..., 4)
def q_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    Assumes q1 and q2 have shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    q = np.zeros(w.shape + (4,))
    q[..., 0] = w
    q[..., 1] = x
    q[..., 2] = y
    q[..., 3] = z
    return q

def q_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Conjugate of a quaternion.
    Assumes q has shape (..., 4)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_conj = np.empty_like(q)
    q_conj[..., 0] = w
    q_conj[..., 1] = -x
    q_conj[..., 2] = -y
    q_conj[..., 3] = -z
    return q_conj

def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q.
    Assumes v has shape (..., 3) and q has shape (..., 4)
    """
    v_as_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return q_mult(q_mult(q, v_as_quat), q_conjugate(q))[..., 1:]


def axis_angle_to_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to quaternions.
    Assumes axis has shape (..., 3) and angle has shape (...)
    """
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)[..., np.newaxis]
    return np.stack((w, xyz), axis=-1)


def quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert quaternions to axis-angle representation.
    Assumes q has shape (..., 4)
    """
    w, xyz = q[..., 0], q[..., 1:]
    angle = 2 * np.arccos(w)
    norm_xyz = np.linalg.norm(xyz, axis=-1)
    ind = norm_xyz > 1e-12
    axis = xyz
    if not np.any(ind):
        return axis, angle
    axis[ind, :] = axis[ind, :] / norm_xyz[ind, None]
    return axis, angle
