"""Differentiable geometry of polygon surface meshes.

This module provides geometric analysis functions for polygon surface meshes,
including face areas, normals, perimeters, adjacency, and neighbor information.

These utilities provide differentiable geometric computations for surface mesh
analysis, rendering, and gradient-based workflows.
"""

from math import pi

import numpy as np
import torch


def _unique_rows_torch(tensor, return_inverse=False, return_counts=False):
    """Device-aware row-wise unique for a 2D non-negative integer tensor.

    torch.unique(dim=0) is fast on CUDA (a massively parallel sort) but far
    slower than a numpy-based approach on CPU -- the same finding
    established for VoroPlex's core packing layer (see
    voroplex._mesh._packed._unique_rows). Measured at 20 million
    triangle edges (N=1,000,000, half-inner 3D surface mesh):
    torch.unique(dim=0) on CPU took 47.3s; this dispatch keeps that path
    on CUDA and uses numpy's fast 1D sort (via an opaque void-record view
    of each row) on CPU instead. The rows are grouped correctly on both, but
    the order of the unique rows differs: the CPU path sorts the raw bytes,
    which is not numeric order on little-endian machines. Callers should not
    rely on the order.
    """
    if tensor.device.type == "cuda":
        return torch.unique(
            tensor,
            dim=0,
            sorted=True,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

    arr = np.ascontiguousarray(tensor.cpu().numpy())
    void_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    flat = arr.view(void_dtype).reshape(-1)
    result = np.unique(
        flat, return_inverse=return_inverse, return_counts=return_counts
    )
    if return_inverse or return_counts:
        unique_flat, rest = result[0], result[1:]
    else:
        unique_flat, rest = result, ()
    unique = unique_flat.view(arr.dtype).reshape(-1, arr.shape[1])
    unique_t = torch.from_numpy(unique).to(tensor.device)
    if not rest:
        return unique_t
    extras = tuple(
        torch.from_numpy(r.astype(np.int64, copy=False)).to(tensor.device)
        for r in rest
    )
    return (unique_t,) + extras


def _triangle_centers(tri_vertices, tri_vertex_ids):
    """
    Compute triangle centroids from vertex coordinates and connectivity.

    Parameters
    ----------
    tri_vertices : torch.Tensor, shape (N_verts, 3)
        Vertex coordinates
    tri_vertex_ids : torch.Tensor, shape (N_trias, 3)
        Vertex indices per triangle

    Returns
    -------
    tri_centers : torch.Tensor, shape (N_trias, 3)
        Triangle centroids (average of 3 vertices)
    """
    # Gather vertex coordinates for each triangle
    v = tri_vertices[tri_vertex_ids]  # [N_trias, 3, 3]

    # Compute per-triangle centroid
    tri_centers = v.mean(dim=1)  # [N_trias, 3]

    return tri_centers


def _triangle_normals(v0, v1, v2):
    """
    Compute triangle normals using right-hand rule.

    Assumes counter-clockwise vertex ordering when viewed from outside.

    Parameters
    ----------
    v0, v1, v2 : torch.Tensor, shape (B, 3)
        Triangle vertices

    Returns
    -------
    normals : torch.Tensor, shape (B, 3)
        Normalized normal vectors
    """
    edge1 = v1 - v0  # (B, 3)
    edge2 = v2 - v0  # (B, 3)

    # Right-hand rule: edge1 × edge2
    normals = torch.linalg.cross(edge1, edge2, dim=1)  # (B, 3)
    norms = torch.norm(normals, p=2, dim=1, keepdim=True)  # (B, 1)

    # Unit normals, with zero for a triangle of zero area. (The test is exact,
    # so that it does not depend on the units of length.)
    normals = normals / torch.where(norms > 0, norms, torch.ones_like(norms))

    return normals


def _face_neighbor_pairs_and_unique_edges(tri_vertex_ids, tri_to_face_id, device):
    """
    Compute pairs of polygon faces that share edges (face adjacency).

    This function identifies which polygon faces are neighbors by finding
    shared edges in the triangulated representation.

    Parameters
    ----------
    tri_vertex_ids : torch.Tensor, shape (N_tria, 3)
        Vertex indices for each triangle. Last vertex may be face centroid.
    tri_to_face_id : torch.Tensor, shape (N_tria,)
        Mapping from triangle ID to polygon face ID
    device : torch.device
        Device for computation

    Returns
    -------
    face_neighbor_pairs : torch.Tensor, shape (N_pairs, 2)
        Neighboring polygon face IDs [f1, f2] where f1 < f2
    unique_edges : torch.Tensor, shape (N_unique_edges, 2)
        Undirected unique edges [v1, v2] where v1 < v2
    """
    # Use only first two vertices of each triangle as edges
    # (last vertex is centroid in fan triangulation)
    edges = tri_vertex_ids[:, :-1]  # [N_tria, 2]
    edges = torch.sort(edges, dim=1)[0]  # ensure v1 < v2

    unique_edges, inverse_indices, edge_counts = _unique_rows_torch(
        edges, return_inverse=True, return_counts=True
    )

    sorted_edges, sort_idx = torch.sort(inverse_indices)
    sorted_faces = tri_to_face_id[sort_idx]
    edge_starts = torch.cumsum(edge_counts, dim=0) - edge_counts
    max_faces = int(edge_counts.max()) if edge_counts.numel() else 0
    edge_faces = torch.full(
        (unique_edges.shape[0], max_faces),
        -1,
        dtype=torch.int64,
        device=device,
    )
    positions = (
        torch.arange(sorted_faces.numel(), device=device)
        - torch.repeat_interleave(edge_starts, edge_counts)
    )
    edge_faces[sorted_edges, positions] = sorted_faces

    pair_columns = torch.triu_indices(
        max_faces, max_faces, offset=1, device=device
    )
    if pair_columns.shape[1]:
        face_pairs = torch.stack(
            [
                edge_faces[:, pair_columns[0]],
                edge_faces[:, pair_columns[1]],
            ],
            dim=-1,
        ).reshape(-1, 2)
        valid_pairs = (
            (face_pairs[:, 0] >= 0)
            & (face_pairs[:, 1] >= 0)
            & (face_pairs[:, 0] != face_pairs[:, 1])
        )
        face_pairs = face_pairs[valid_pairs]
        face_pairs = torch.sort(face_pairs, dim=1).values
        face_pairs = _unique_rows_torch(face_pairs)
    else:
        face_pairs = torch.empty(
            (0, 2), dtype=torch.int64, device=device
        )

    return face_pairs, unique_edges


def _pairs_to_adjacency(pairs, n_items):
    """Convert unique undirected pairs to padded directed adjacency."""
    device = pairs.device
    if pairs.numel() == 0:
        return (
            torch.empty((n_items, 0), dtype=torch.int64, device=device),
            torch.zeros(n_items, dtype=torch.int64, device=device),
        )

    sources = torch.cat([pairs[:, 0], pairs[:, 1]])
    targets = torch.cat([pairs[:, 1], pairs[:, 0]])
    order = torch.argsort(sources * n_items + targets)
    sources = sources[order]
    targets = targets[order]
    counts = torch.bincount(sources, minlength=n_items)
    width = int(counts.max())
    starts = torch.cumsum(counts, dim=0) - counts
    columns = (
        torch.arange(sources.numel(), device=device)
        - torch.repeat_interleave(starts, counts)
    )
    adjacency = torch.full(
        (n_items, width), -1, dtype=torch.int64, device=device
    )
    adjacency[sources, columns] = targets
    return adjacency, counts


def _vertex_neighbor_info(
    edges, vertices, face_vertex_ids, face_mask, face_normals,
    n_vertices_per_face, face_neighbor_pairs, face_centers, face_areas
):
    """
    Compute vertex and face neighbor statistics.

    For each vertex: computes average position of neighboring vertices and
    average normal of incident faces.

    For each face: computes average centroid, area, and normal of neighboring faces.

    Parameters
    ----------
    edges : torch.Tensor, shape (N_edges, 2)
        Unique edges [v1, v2]
    vertices : torch.Tensor, shape (N, 3)
        Vertex coordinates
    face_vertex_ids : torch.Tensor, shape (F, max_verts)
        Face vertex indices (padded with -100)
    face_mask : torch.Tensor, shape (F, max_verts)
        Boolean mask for valid vertices
    face_normals : torch.Tensor, shape (F, 3)
        Face normal vectors
    n_vertices_per_face : torch.Tensor, shape (F,)
        Number of vertices per face
    face_neighbor_pairs : torch.Tensor, shape (N_pairs, 2)
        Neighboring face pairs
    face_centers : torch.Tensor, shape (F, 3)
        Face centroids
    face_areas : torch.Tensor, shape (F,)
        Face areas

    Returns
    -------
    dict
        Dictionary containing:
        - 'vertex_neighbor_avg_position': (N, 3) average neighbor positions
        - 'vertex_neighbor_avg_normal': (N, 3) average incident face normals
        - 'face_neighbor_avg_center': (F, 3) average neighbor centroids
        - 'face_neighbor_avg_area': (F,) average neighbor areas
        - 'face_neighbor_avg_normal': (F, 3) average neighbor normals
    """
    device = vertices.device
    N = vertices.shape[0]
    F = face_vertex_ids.shape[0]

    # Vertex neighbor statistics.
    vertex_neighbor_sum = torch.zeros((N, 3), device=device, dtype=vertices.dtype)
    vertex_neighbor_count = torch.zeros(N, device=device, dtype=torch.long)

    v1_indices = edges[:, 0]
    v2_indices = edges[:, 1]
    v1_positions = vertices[v1_indices]
    v2_positions = vertices[v2_indices]

    vertex_neighbor_sum.scatter_add_(0, v1_indices.unsqueeze(-1).expand(-1, 3), v2_positions)
    vertex_neighbor_count.scatter_add_(0, v1_indices, torch.ones_like(v1_indices))

    vertex_neighbor_sum.scatter_add_(0, v2_indices.unsqueeze(-1).expand(-1, 3), v1_positions)
    vertex_neighbor_count.scatter_add_(0, v2_indices, torch.ones_like(v2_indices))

    vertex_neighbor_avg_position = torch.zeros_like(vertices)
    has_neighbors = vertex_neighbor_count > 0
    vertex_neighbor_avg_position[has_neighbors] = (
        vertex_neighbor_sum[has_neighbors]
        / vertex_neighbor_count[has_neighbors].unsqueeze(-1).to(vertices.dtype)
    )

    vertex_neighbor_avg_normal = torch.zeros((N, 3), device=device, dtype=vertices.dtype)
    vertex_face_count = torch.zeros(N, device=device, dtype=torch.long)

    flat_vertex_ids = face_vertex_ids[face_mask]
    repeated_normals = face_normals.repeat_interleave(n_vertices_per_face, dim=0)

    vertex_neighbor_avg_normal.index_add_(0, flat_vertex_ids, repeated_normals)
    vertex_face_count.index_add_(
        0, flat_vertex_ids, torch.ones_like(flat_vertex_ids, dtype=torch.long))

    has_faces = vertex_face_count > 0
    vertex_neighbor_avg_normal[has_faces] = (
        vertex_neighbor_avg_normal[has_faces]
        / vertex_face_count[has_faces].unsqueeze(-1).to(vertices.dtype)
    )

    # Face neighbor statistics.
    nei_src = torch.cat([face_neighbor_pairs[:, 0], face_neighbor_pairs[:, 1]], dim=0)
    nei_dst = torch.cat([face_neighbor_pairs[:, 1], face_neighbor_pairs[:, 0]], dim=0)

    nei_face_centroids = face_centers[nei_dst]
    nei_face_areas = face_areas[nei_dst]
    nei_face_normals = face_normals[nei_dst]

    face_neighbor_avg_center = torch.zeros((F, 3), dtype=vertices.dtype, device=device)
    face_neighbor_avg_area = torch.zeros(F, dtype=vertices.dtype, device=device)
    face_neighbor_avg_normal = torch.zeros((F, 3), dtype=vertices.dtype, device=device)

    face_neighbor_avg_center.index_add_(0, nei_src, nei_face_centroids)
    face_neighbor_avg_area.index_add_(0, nei_src, nei_face_areas)
    face_neighbor_avg_normal.index_add_(0, nei_src, nei_face_normals)

    face_neighbor_counts = torch.bincount(nei_src, minlength=F).clamp(min=1).to(vertices.dtype)

    face_neighbor_avg_center /= face_neighbor_counts[:, None]
    face_neighbor_avg_area /= face_neighbor_counts
    face_neighbor_avg_normal /= face_neighbor_counts[:, None]

    return {
        "vertex_neighbor_avg_position": vertex_neighbor_avg_position,
        "vertex_neighbor_avg_normal": vertex_neighbor_avg_normal,
        "face_neighbor_avg_center": face_neighbor_avg_center,
        "face_neighbor_avg_area": face_neighbor_avg_area,
        "face_neighbor_avg_normal": face_neighbor_avg_normal,
    }


def _empty_surface_mesh_info(coords):
    """The output of compute_surface_geometry for a mesh with no faces."""
    dev, dt = coords.device, coords.dtype
    z = lambda *shape: torch.zeros(shape, device=dev, dtype=dt)
    i = lambda *shape: torch.zeros(shape, device=dev, dtype=torch.int64)
    n = coords.shape[0]
    return {
        "face_areas": z(0), "face_normals": z(0, 3), "face_centers": z(0, 3),
        "face_circularities": z(0), "face_perimeters": z(0),
        "tri_vertices": coords, "n_triangles": 0,
        "tri_vertex_ids": i(0, 3), "tri_to_face_id": i(0),
        "tri_centers": z(0, 3),
        "adjacency_tensor": i(0, 0), "n_adjacent": i(0),
        "face_neighbor_pairs": i(0, 2), "edges": i(0, 2),
        "vertex_neighbor_avg_position": z(n, 3), "vertex_neighbor_avg_normal": z(n, 3),
        "face_neighbor_avg_center": z(0, 3), "face_neighbor_avg_area": z(0),
        "face_neighbor_avg_normal": z(0, 3),
    }


def compute_surface_geometry(
    vertex_coords, face_vertex_ids, n_vertices_per_face, face_mask,
    adjacency_tensor=None, n_adjacent=None, face_neighbor_pairs=None, edges=None
):
    """
    Compute comprehensive geometric and connectivity information for polygon surface meshes.

    This function performs fan triangulation of polygon faces and computes:
    - Basic geometry: areas, normals, centers, circularities, perimeters
    - Triangle subdivision: for rendering and integration
    - Face adjacency: neighbor pairs and connectivity
    - Vertex neighbor statistics: for Laplacian smoothing and regularization
    - Face neighbor statistics: for geometric analysis and regularization

    The function uses efficient vectorized operations with PyTorch and maintains
    differentiability with respect to input coordinates.

    Parameters
    ----------
    vertex_coords : torch.Tensor, shape (N, 3)
        Vertex coordinates of all surface vertices
    face_vertex_ids : torch.Tensor, shape (F, max_verts)
        Face vertex indices (padded with -100 for faces with fewer vertices)
    n_vertices_per_face : torch.Tensor, shape (F,)
        Number of vertices per face
    face_mask : torch.Tensor, shape (F, max_verts)
        Boolean mask indicating valid vertices (True) vs padding (False)
    adjacency_tensor : torch.Tensor, shape (F, max_adj), optional
        Pre-computed adjacency tensor (will be computed if None)
    n_adjacent : torch.Tensor, shape (F,), optional
        Pre-computed neighbor counts (will be computed if None)
    face_neighbor_pairs : torch.Tensor, shape (N_pairs, 2), optional
        Pre-computed neighbor pairs (will be computed if None)
    edges : torch.Tensor, shape (N_edges, 2), optional
        Pre-computed unique edges (will be computed if None)

    Returns
    -------
    dict
        Dictionary containing:

        **Basic geometry:**
        - 'face_areas': (F,) tensor of face areas
        - 'face_normals': (F, 3) tensor of unit normals, from Newell's method;
          they point outward if the vertices are counter-clockwise as seen
          from outside, as in the output of ``surface_mesh``
        - 'face_centers': (F, 3) tensor of the mean of each face's vertices
          (not the area centroid)
        - 'face_circularities': (F,) tensor of isoperimetric ratios (4π·Area/Perimeter²)
        - 'face_perimeters': (F,) tensor of face perimeters

        **Triangle subdivision (fan triangulation):**
        - 'tri_vertices': (N+F, 3) tensor [original vertices; face centroids]
        - 'n_triangles': int, total number of triangles
        - 'tri_vertex_ids': (N_trias, 3) tensor of triangle connectivity [v0, v1, centroid]
        - 'tri_to_face_id': (N_trias,) tensor mapping triangles to parent faces
        - 'tri_centers': (N_trias, 3) tensor of triangle centroids

        **Adjacency and connectivity:**
        - 'adjacency_tensor': (F, max_adj) tensor with -1 padding
        - 'n_adjacent': (F,) tensor of neighbor counts per face
        - 'face_neighbor_pairs': (N_pairs, 2) tensor of neighboring face IDs
        - 'edges': (N_edges, 2) tensor of unique edges [v1, v2] where v1 < v2

        **Vertex neighbor information (for smoothing and regularization):**
        - 'vertex_neighbor_avg_position': (N, 3) tensor of average neighbor positions
        - 'vertex_neighbor_avg_normal': (N, 3) tensor of average incident face normals

        **Face neighbor information (for geometric analysis):**
        - 'face_neighbor_avg_center': (F, 3) tensor of average neighbor centroids
        - 'face_neighbor_avg_area': (F,) tensor of average neighbor areas
        - 'face_neighbor_avg_normal': (F, 3) tensor of average neighbor normals

    Notes
    -----
    - Face vertices should be ordered counter-clockwise when viewed from
      outside, as in the output of ``surface_mesh``
    - Adjacency inputs that are given are reused; any that are None are
      computed
    - The floating-point outputs are differentiable with respect to
      vertex_coords. For a topology result, these are differentiable
      functions of the generator positions; for a geometry result, they are
      not
    - Uses fan triangulation: each polygon face is split into triangles [v_i, v_{i+1}, centroid]

    Examples
    --------
    >>> # Get surface mesh from VoroPlex
    >>> result = voroplex.compute_voronoi_3d(points, limits)
    >>> surf = voroplex.surface_mesh(result, points)
    >>>
    >>> # Compute geometric properties
    >>> geom = compute_surface_geometry(
    ...     surf['vertex_coords'],
    ...     surf['facet_vertex_ids'],
    ...     surf['n_vertices_per_facet'],
    ...     surf['facet_mask']
    ... )
    >>>
    >>> print(f"Face areas: {geom['face_areas']}")
    >>> print(f"Face normals: {geom['face_normals']}")
    >>> print(f"Number of triangles: {geom['n_triangles']}")
    """
    device = vertex_coords.device
    C = vertex_coords
    N = C.shape[0]
    F = face_vertex_ids.shape[0]
    face_mask = torch.as_tensor(face_mask, dtype=torch.bool, device=device)
    n_vertices_per_face = torch.as_tensor(
        n_vertices_per_face, dtype=torch.int64, device=device
    )
    if F == 0:
        return _empty_surface_mesh_info(C)
    if not torch.equal(face_mask.sum(dim=1), n_vertices_per_face):
        raise ValueError("face_mask does not match n_vertices_per_face")
    max_verts = face_vertex_ids.shape[1]

    # Padded face connectivity.
    face_vertex_ids_safe = face_vertex_ids.clone()
    face_vertex_ids_safe = face_vertex_ids_safe.masked_fill(~face_mask, 0)

    face_vertices = C[face_vertex_ids_safe] * face_mask[..., None]  # [F, max_verts, 3]
    face_centers = face_vertices.sum(dim=1) / n_vertices_per_face.unsqueeze(-1).to(C.dtype)

    face_vertex_ids = face_vertex_ids_safe.masked_fill(~face_mask, -100)

    # Fan triangles: [v0, v1, face centroid].
    v0_ID = face_vertex_ids
    v1_ID = torch.roll(v0_ID, shifts=-1, dims=1)

    # Close each polygon at its last valid vertex.
    last_values = v1_ID[:, -1].unsqueeze(1)
    invalid_mask = (v1_ID == -100)
    v1_ID = torch.where(invalid_mask, last_values.expand_as(v1_ID), v1_ID)

    tri_vertices = torch.cat((C, face_centers), dim=0)
    centroid_IDs = N + torch.arange(F, device=device)
    centroid_IDs = centroid_IDs[:, None].expand(F, max_verts)

    tri_vertex_ids = torch.stack([v0_ID, v1_ID, centroid_IDs], dim=-1)  # (F, max_verts, 3)
    tri_vertex_ids = tri_vertex_ids[face_mask]  # (n_triangles, 3)
    n_triangles = tri_vertex_ids.shape[0]

    face_ID_value = torch.arange(F, device=device).unsqueeze(1)
    tri_to_face_id = torch.zeros((F, max_verts), dtype=torch.int64, device=device)
    tri_to_face_id = torch.where(
        face_mask,
        face_ID_value.expand_as(face_mask),
        tri_to_face_id
    )
    tri_to_face_id = tri_to_face_id[face_mask]

    tri_centers = _triangle_centers(tri_vertices, tri_vertex_ids)

    TriaV0 = tri_vertices[tri_vertex_ids[:, 0]]
    TriaV1 = tri_vertices[tri_vertex_ids[:, 1]]
    TriaV2 = tri_vertices[tri_vertex_ids[:, 2]]

    ab = TriaV1 - TriaV0
    ac = TriaV2 - TriaV0
    cross_prod = torch.cross(ab, ac, dim=1)
    tri_areas = 0.5 * torch.norm(cross_prod, dim=1)

    face_areas = torch.zeros(F, device=device, dtype=C.dtype)
    face_areas.index_add_(0, tri_to_face_id, tri_areas)

    # Newell's normal: the sum of the fan-triangle cross products is twice the
    # vector area of the face, which is reliable even if some edges are very
    # short. With the vertices counter-clockwise as seen from outside, it
    # points outward.
    vector_areas = torch.zeros((F, 3), device=device, dtype=C.dtype)
    vector_areas.index_add_(0, tri_to_face_id, cross_prod)
    vector_norms = torch.linalg.vector_norm(vector_areas, dim=1, keepdim=True)
    face_normals = vector_areas / torch.where(
        vector_norms > 0, vector_norms, torch.ones_like(vector_norms)
    )

    rolled_vertices = torch.roll(face_vertices, shifts=-1, dims=1)
    last_valid_indices = n_vertices_per_face - 1
    face_range = torch.arange(F, device=device)
    rolled_vertices[face_range, last_valid_indices] = face_vertices[face_range, 0]

    edge_vectors = rolled_vertices - face_vertices
    edge_lengths = torch.norm(edge_vectors, dim=2) * face_mask.to(C.dtype)
    face_perimeters = torch.sum(edge_lengths, dim=1)

    # Isoperimetric ratio, 4 pi A / P^2, which is scale-free. It is set to
    # zero for a face of zero perimeter.
    perimeter_safe = torch.where(
        face_perimeters > 0, face_perimeters, torch.ones_like(face_perimeters)
    )
    face_circularities = torch.where(
        face_perimeters > 0,
        4 * pi * face_areas / perimeter_safe ** 2,
        torch.zeros_like(face_areas),
    )
    face_circularities = torch.clamp(face_circularities, min=0.0, max=1.0)

    if face_neighbor_pairs is None or edges is None:
        computed_pairs, computed_edges = _face_neighbor_pairs_and_unique_edges(
            tri_vertex_ids, tri_to_face_id, device
        )
        if face_neighbor_pairs is None:
            face_neighbor_pairs = computed_pairs
        if edges is None:
            edges = computed_edges

    if adjacency_tensor is None or n_adjacent is None:
        computed_adjacency, computed_counts = _pairs_to_adjacency(
            face_neighbor_pairs, F
        )
        if adjacency_tensor is None:
            adjacency_tensor = computed_adjacency
        if n_adjacent is None:
            n_adjacent = computed_counts

    vertex_nei_info_dict = _vertex_neighbor_info(
        edges, C, face_vertex_ids, face_mask, face_normals,
        n_vertices_per_face, face_neighbor_pairs, face_centers, face_areas
    )

    result_dict = {
        # Basic geometry
        "face_areas": face_areas,
        "face_normals": face_normals,
        "face_centers": face_centers,
        "face_circularities": face_circularities,
        "face_perimeters": face_perimeters,

        # Triangle subdivision
        "tri_vertices": tri_vertices,
        "n_triangles": n_triangles,
        "tri_vertex_ids": tri_vertex_ids,
        "tri_to_face_id": tri_to_face_id,
        "tri_centers": tri_centers,

        # Adjacency and connectivity
        "adjacency_tensor": adjacency_tensor,
        "n_adjacent": n_adjacent,
        "face_neighbor_pairs": face_neighbor_pairs,
        "edges": edges,
    }
    result_dict.update(vertex_nei_info_dict)

    return result_dict
