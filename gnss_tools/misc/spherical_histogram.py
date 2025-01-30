from typing import Tuple
import numpy as np
import numba as nb

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


def spherical_mean_std(
        elevations: np.ndarray,  # deg
        azimuths: np.ndarray,  # deg
        values: np.ndarray,
        icososphere_num_subdivisions: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        bin_vertices, means, stds
    """
    ico_vertices, _ = icososphere.generate_icososphere(icososphere_num_subdivisions)
    num_vertices = ico_vertices.shape[0]
    num_points = len(values)
    points = np.zeros((num_points, 3))
    theta = np.radians(90 - azimuths)
    phi = np.radians(elevations)
    points[:, 0] = np.cos(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.cos(phi)
    points[:, 2] = np.sin(phi)
    closest_vertices = find_closest_vertex(ico_vertices, points)

    means = np.nan * np.zeros(num_vertices)
    stds = np.nan * np.zeros(num_vertices)
    for i in range(num_vertices):
        indices = np.where(closest_vertices == i)[0]
        if len(indices) > 0:
            means[i] = np.mean(values[indices])
            stds[i] = np.std(values[indices])
    return ico_vertices, means, stds

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
import scipy.spatial

def plot_sky_voronoi_2d(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    values: np.ndarray,
    clip_radius: float = 90,
    show_elevation_rings: bool = True,
    elevation_ring_spacing: int = 30,
) -> Tuple[Figure, Axes]:
    

    r = 90 - elevations
    theta = np.deg2rad(90 - azimuths)
    xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    vor = scipy.spatial.Voronoi(xy)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    clip_patch = patches.Circle((0, 0), radius=clip_radius, transform=ax.transData)
    cnorm = plt.Normalize(vmin=-2, vmax=2)
    cmap = plt.cm.get_cmap("viridis")
    vor_plot_patches = []
    for i in range(len(vor.points)):
        vor_point = vor.points[i]
        vor_region = vor.regions[vor.point_region[i]]
        
        if not -1 in vor_region:
            polygon = [vor.vertices[i] for i in vor_region]
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
            ax.text(0, r, f"{elev}°", ha="center", va="center", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.plot([0, 0], [-clip_radius, clip_radius], "k", linestyle="--", alpha=.2)
    ax.plot([-clip_radius, clip_radius], [0, 0], "k", linestyle="--", alpha=.2)
    ax.set_xlim(-clip_radius, clip_radius)
    ax.set_ylim(-clip_radius, clip_radius)
    ax.set_aspect("equal")

    return fig, ax