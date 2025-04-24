from typing import Callable, Tuple, Optional
from matplotlib.cm import ScalarMappable
import numpy as np
import numba as nb
import gnss_tools.coords.icososphere as icososphere
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
import matplotlib.patches as patches
import scipy.spatial
from scipy.spatial.transform import Rotation

def rotation_to_z_axis(v: np.ndarray) -> Rotation:
    """
    Compute the rotation matrix R such that R @ v = [0, 0, 1].
    
    Parameters:
        v (array-like): A 3D unit vector.
    
    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    v = np.array(v, dtype=np.float64)
    v /= np.linalg.norm(v)  # Ensure it's a unit vector
    z_axis = np.array([0, 0, 1])
    
    if np.allclose(v, z_axis):
        return Rotation.from_rotvec(np.zeros(3))  # No rotation needed
    
    if np.allclose(v, -z_axis):
        return Rotation.from_rotvec(np.pi * np.array([1, 0, 0]))
    
    axis = np.cross(v, z_axis)
    angle = np.arccos(np.dot(v, z_axis))
    
    rotation = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
    return rotation


@nb.jit(nopython=True)
def numba_find_closest_vertex(
    num_vertices: int,
    vertices: np.ndarray,  # (num_vertices, num_dims)
    num_points: int,
    points: np.ndarray,  # (num_points, num_dims)
) -> np.ndarray:
    # assert(vertices.shape == (num_vertices, 3))
    # assert(points.shape == (num_points, 3))
    closest_vertices = np.zeros(num_points, dtype=np.int32)
    for i in range(num_points):
        min_dist = np.inf
        for j in range(num_vertices):
            diff = vertices[j] - points[i]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_vertices[i] = j
    return closest_vertices

def find_closest_vertex(
        vertices: np.ndarray,  # (num_vertices, num_dims)
        points: np.ndarray,  # (num_points, num_dims)
) -> np.ndarray:
    num_vertices = vertices.shape[0]
    num_points = points.shape[0]
    return numba_find_closest_vertex(num_vertices, vertices, num_points, points)


@nb.jit(nopython=True)
def numba_bin_mean_std(
    closest_vertex_indices: np.ndarray,  # (num_points,)
    values: np.ndarray,  # (num_points,)
    means: np.ndarray,  # (num_vertices,)
    stds: np.ndarray,  # (num_vertices,)
    counts: np.ndarray,  # (num_vertices,)
) -> None:
    num_points = len(values)
    num_vertices = len(means)
    for i in range(num_points):
        vertex_index = closest_vertex_indices[i]
        if not np.isnan(values[i]):
            means[vertex_index] += values[i]
            stds[vertex_index] += values[i] ** 2
            counts[vertex_index] += 1
    for i in range(num_vertices):
        if counts[i] > 0:
            means[i] /= counts[i]
            stds[i] = np.sqrt(stds[i] / counts[i] - means[i] ** 2)
        else:
            means[i] = np.nan
            stds[i] = np.nan


def old_spherical_mean_std(
        elevations: np.ndarray,  # deg  (num_points,)
        azimuths: np.ndarray,  # deg  (num_points,)
        values: np.ndarray,  # (num_points,)
        icososphere_num_subdivisions: int = 2,
        center_elevation: float = 90,
        center_azimuth: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        bin_vertices, bin_values, counts
    """
    ico_vertices, _ = icososphere.generate_icososphere(icososphere_num_subdivisions)

    # apply fixed rotation to ico vertices
    R = Rotation.from_euler('zyx', [3, 8, 31], degrees=True)
    ico_vertices = R.apply(ico_vertices)

    center_theta = np.radians(90 - center_azimuth)
    center_phi = np.radians(center_elevation)
    center_xyz = np.array([
        np.cos(center_theta) * np.cos(center_phi),
        np.sin(center_theta) * np.cos(center_phi),
        np.sin(center_phi)
    ])
    R = rotation_to_z_axis(center_xyz)

    num_vertices = ico_vertices.shape[0]
    num_points = len(values)
    points = np.zeros((num_points, 3))
    theta = np.radians(90 - azimuths)
    phi = np.radians(elevations)
    points[:, 0] = np.cos(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.cos(phi)
    points[:, 2] = np.sin(phi)
    points = R.apply(points)
    closest_vertices = find_closest_vertex(ico_vertices, points)

    means = np.zeros(num_vertices)
    stds = np.zeros(num_vertices)
    counts = np.zeros(num_vertices)
    numba_bin_mean_std(closest_vertices, values, means, stds, counts)
    # for i in range(num_vertices):
    #     indices = np.where(closest_vertices == i)[0]
    #     counts[i] = len(indices)
    #     if len(indices) > 0:
    #         means[i] = np.nanmean(values[indices])
    #         stds[i] = np.nanstd(values[indices])
    return ico_vertices, means, stds, counts


def plot_sky_voronoi_2d(
    ax: Axes,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    values: np.ndarray,
    nan_mask: Optional[np.ndarray] = None,
    clip_radius: float = 90,
    show_elevation_rings: bool = True,
    elevation_ring_spacing: int = 30,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    text_fontcolor: str = "k",
) -> ScalarMappable:
    
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax

    r = 90 - elevations
    theta = np.deg2rad(90 - azimuths)
    xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    vor = scipy.spatial.Voronoi(xy)
    
    clip_patch = patches.Circle((0, 0), radius=clip_radius, transform=ax.transData)
    cnorm = plt.Normalize(vmin=vmin, vmax=vmax)  # type: ignore
    cmap = plt.cm.get_cmap("viridis")
    vor_plot_patches = []
    for i in range(len(vor.points)):
        vor_point = vor.points[i]
        vor_region = vor.regions[vor.point_region[i]]
        
        if not -1 in vor_region:
            polygon = [vor.vertices[i] for i in vor_region]
            if nan_mask is not None and nan_mask[i]:
                color = (1, 1, 1, 0)
            else:
                color = cmap(cnorm(values[i]))
            vor_patch = ax.fill(*zip(*polygon), color=color)
            vor_plot_patches.extend(vor_patch)

    for patch in vor_plot_patches:
        patch.set_clip_path(clip_patch)

    if show_elevation_rings:
        for elev in np.arange(90, 90 - clip_radius - 1, -elevation_ring_spacing):
            r = 90 - elev
            circle = patches.Circle((0, 0), radius=r, fill=False, linestyle="--", alpha=.2)
            ax.add_patch(circle)
            ax.text(0, r, f"{elev}°", ha="center", va="center", fontsize=8, color=text_fontcolor)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.plot([0, 0], [-clip_radius, clip_radius], "k", linestyle="--", alpha=.2)
    ax.plot([-clip_radius, clip_radius], [0, 0], "k", linestyle="--", alpha=.2)
    ax.set_xlim(-clip_radius, clip_radius)
    ax.set_ylim(-clip_radius, clip_radius)
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])

    return sm




@nb.jit(nopython=True)
def numba_bin_values(
    bin_index: np.ndarray,  # (num_values,)
    values: np.ndarray,  # (num_values, num_dims)
    binned_values: np.ndarray,  # (num_bins, num_dims)
    counts: np.ndarray,  # (num_bins,)
) -> None:
    """
    Accumulate the values into the corresponding bins.
    """
    num_values, num_dims = values.shape
    num_bins, = counts.shape
    assert bin_index.shape == (num_values,)
    assert binned_values.shape == (num_bins, num_dims)

    for i in range(num_values):
        bindex = bin_index[i]
        for j in range(num_dims):
            if not np.isnan(values[i, j]):
                binned_values[bindex, j] += values[i, j]
        counts[bindex] += 1


# def spherical_mean_std(
#         elevations: np.ndarray,  # deg  (num_points,)
#         azimuths: np.ndarray,  # deg  (num_points,)
#         values: np.ndarray,  # (num_points, num_dims)
#         icososphere_num_subdivisions: int = 2,
#         icososphere_rotation: Optional[Rotation] = None,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Returns:
#         bin_vertices, bin_values, counts
#     """
#     ico_vertices, _ = icososphere.generate_icososphere(icososphere_num_subdivisions)
#     if icososphere_rotation is not None:
#         ico_vertices = icososphere_rotation.apply(ico_vertices)
    
#     num_vertices = ico_vertices.shape[0]
#     num_points, num_dims = values.shape
#     points = np.zeros((num_points, 3))
#     theta = np.deg2rad(360 - azimuths)
#     phi = np.deg2rad(elevations)
#     points[:, 0] = np.cos(theta) * np.cos(phi)
#     points[:, 1] = np.sin(theta) * np.cos(phi)
#     points[:, 2] = np.sin(phi)
#     closest_vertices = find_closest_vertex(ico_vertices, points)

#     binned_values = np.zeros((num_vertices, num_dims))
#     counts = np.zeros(num_vertices)
#     numba_bin_values(closest_vertices, values, binned_values, counts)
#     return ico_vertices, binned_values, counts




def bin_3d(
        points: np.ndarray,  # (num_points, 3)
        values: np.ndarray,  # (num_points, num_dims)
        bin_points: np.ndarray,  # (num_bins, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the closest bin point for each point and accumulate the values.

    Returns:
        bin_values, counts
    """
    num_points, num_dims = values.shape
    num_bins, _ = bin_points.shape
    assert points.shape == (num_points, 3)
    assert bin_points.shape == (num_bins, 3)

    closest_vertices = find_closest_vertex(bin_points, points)

    bin_values = np.zeros((num_bins, num_dims))
    counts = np.zeros(num_bins)
    numba_bin_values(closest_vertices, values, bin_values, counts)
    return bin_values, counts


def spherical_hist(
        elevations: np.ndarray,  # deg  (num_points,)
        azimuths: np.ndarray,  # deg  (num_points,)
        values: np.ndarray,  # (num_points, num_dims)
        bin_vertices: Optional[np.ndarray] = None,
        icososphere_num_subdivisions: int = 2,
        icososphere_rotation: Optional[Rotation] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        bin_vertices, bin_values, counts
    """
    if bin_vertices is None:
        bin_vertices, _ = icososphere.generate_icososphere(icososphere_num_subdivisions)
        if icososphere_rotation is not None:
            bin_vertices = icososphere_rotation.apply(bin_vertices)
    else:
        assert bin_vertices.shape[1] == 3
    if bin_vertices is None:
        raise ValueError("bin_vertices must be provided or generated.")
    
    num_vertices = bin_vertices.shape[0]
    num_points, num_dims = values.shape
    points = np.zeros((num_points, 3))
    theta = np.deg2rad(360 - azimuths)
    phi = np.deg2rad(elevations)
    points[:, 0] = np.cos(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.cos(phi)
    points[:, 2] = np.sin(phi)
    mask = ~np.isnan(points).any(axis=1)
    bin_values, counts = bin_3d(points[mask, :], values[mask, :], bin_vertices)
    return bin_vertices, bin_values, counts



    # points = np.zeros((num_points, 3))
    # theta = np.deg2rad(360 - azimuths)
    # phi = np.deg2rad(elevations)
    # points[:, 0] = np.cos(theta) * np.cos(phi)
    # points[:, 1] = np.sin(theta) * np.cos(phi)
    # points[:, 2] = np.sin(phi)
    # closest_vertices = find_closest_vertex(bin_points, points)

from scipy.interpolate import RBFInterpolator

def get_azimuth_elevation_icososphere_interpolator(
        ico_vertices: np.ndarray,  # (num_vertices, 3)
        bin_values: np.ndarray,  # (num_vertices, num_dims)
        shrink_alpha: float = 1e-8,
        mask_nans: bool = True,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a function that takes azimuths and elevations and returns the interpolated values.
    """
    num_vertices, num_dims = bin_values.shape
    assert ico_vertices.shape == (num_vertices, 3)

    if mask_nans:
        mask = ~np.isnan(bin_values).any(axis=1)
        interpolator_3d = RBFInterpolator(ico_vertices[mask], bin_values[mask])
    else:
        interpolator_3d = RBFInterpolator(ico_vertices, bin_values)

    def interpolator_azel(azimuths: np.ndarray, elevations: np.ndarray) -> np.ndarray:
        num_points = len(azimuths)
        points = np.zeros((num_points, 3))
        theta = np.deg2rad(360 - azimuths)
        phi = np.deg2rad(elevations)
        points[:, 0] = np.cos(theta) * np.cos(phi)
        points[:, 1] = np.sin(theta) * np.cos(phi)
        points[:, 2] = np.sin(phi)
        return interpolator_3d(points * (1 - shrink_alpha))
    
    return interpolator_azel