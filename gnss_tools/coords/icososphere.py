from typing import Dict, Tuple
import numpy as np





def generate_icososphere(
        num_subdivisions: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:  # vertices (N, 3), faces (M, 3)
    """
    Stolen from: http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html

    Returns:
    vertices: (N, 3) array of vertex positions
    faces: (M, 3) array of face indices
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    icosohedron_vertices = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ])
    # Normalize vertices to lie on the unit sphere
    icosohedron_vertices /= np.linalg.norm(icosohedron_vertices, axis=1)[:, None]
    icosohedron_faces = np.asarray([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ])
    if num_subdivisions == 0:
        return icosohedron_vertices, icosohedron_faces

    # An icosohedron has 20 faces and 12 vertices and 30 edges (V + F = E + 2)
    # A subdivided icoshedron has F' faces and E' = 3F' / 2 edges
    # (since each face has 3 edges, but each edge is shared by 2 faces)
    # Each subdivision will add 4 new faces and 3 new edges, and previous edges are all split in half
    # Each subdivision adds 3 vertices per face, but each new vertex is shared by one previous face
    # So the number of vertices is V' = V + E
    # So F' = 4F
    #    E' = 2E + 3F = 4E
    #    V' = E' - F' + 2
    #       = 2E - F + 2
    #       = V + E
    # To compute the number of vertices for N subdivisions, we can evaluate these recurrence relations.
    def get_icososphere_VEF(num_subdivisions: int) -> Tuple[int, int, int]:
        V = 12
        E = 30
        F = 20
        for _ in range(num_subdivisions):
            F = 4 * F
            E = 4 * E
            V = E - F + 2
            # V, E, F = 2 * E - F + 2, 2 * E + 3 * F, 4 * F
        return V, E, F
    
    N_vertices, N_edges, N_faces = get_icososphere_VEF(num_subdivisions)
    vertices = np.zeros((N_vertices, 3))

    _curr_vertex_idx = 0
    def add_vertex(v: np.ndarray) -> int:
        nonlocal _curr_vertex_idx
        if _curr_vertex_idx >= vertices.shape[0]:
            raise ValueError(f"Tried to add too many vertices: {_curr_vertex_idx} >= {N_vertices}")
        vertices[_curr_vertex_idx] = v
        _curr_vertex_idx += 1
        return _curr_vertex_idx - 1
    for v in icosohedron_vertices:
        add_vertex(v)

    # (i, j) -> k  the index of the middle point of edge i-j
    middle_point_cache: Dict[Tuple[int, int], int] = {}
    
    current_faces = icosohedron_faces
    
    def get_middle_point(i: int, j: int) -> int:
        if i > j:
            i, j = j, i
        if (i, j) not in middle_point_cache:
            middle_point = (vertices[i] + vertices[j]) / 2
            middle_point /= np.linalg.norm(middle_point)
            middle_point_idx = add_vertex(middle_point)
            middle_point_cache[(i, j)] = middle_point_idx
        else:
            middle_point_idx = middle_point_cache[(i, j)]
        return middle_point_idx
                
    for _ in range(num_subdivisions):
        # Subdivide each face into 4 new faces
        new_faces = []
        for face in current_faces:
            i0, i1, i2 = face
            # Compute the middle points
            i01 = get_middle_point(i0, i1)
            i12 = get_middle_point(i1, i2)
            i20 = get_middle_point(i2, i0)
            # Add the new faces
            new_faces.extend([
                [i0, i01, i20],
                [i1, i12, i01],
                [i2, i20, i12],
                [i01, i12, i20],
            ])
        current_faces = np.asarray(new_faces)
    
    assert _curr_vertex_idx == N_vertices
    return vertices, current_faces