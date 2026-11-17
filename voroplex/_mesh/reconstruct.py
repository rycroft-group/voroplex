"""3D reconstruction and connectivity for packed Voronoi results."""

import numpy as np
import torch

from ._packed import (
    cell_vertex_layout,
    global_facets,
    global_vertex_map,
    packed_global_facets,
    reconstruct_topology_vertices,
    require_result,
    selected_facet_indices,
    unique_1d,
    unique_geometry_coordinates,
)


def _empty_surface_mesh(device):
    return {
        "facet_vertex_ids": torch.empty(
            (0, 0), dtype=torch.int64, device=device
        ),
        "vertex_coords": torch.empty(
            (0, 3), dtype=torch.float64, device=device
        ),
        "n_vertices_per_facet": torch.empty(
            0, dtype=torch.int64, device=device
        ),
        "facet_mask": torch.empty(
            (0, 0), dtype=torch.bool, device=device
        ),
        "facet_neighbors": torch.empty(
            0, dtype=torch.int64, device=device
        ),
    }


def _reconstruct_vertices_with_map(result, points, device, radii=None):
    require_result(result, dimension=3, representation="topology")
    return reconstruct_topology_vertices(
        result,
        points,
        device,
        return_local_to_global=True,
        radii=radii,
    )


def unique_vertices_3d(result, points, device="cpu", radii=None):
    """Reconstruct differentiable unique vertices from packed topology.

    For a radical tessellation, ``radii`` (a tensor of shape (N,)) gives the
    generator radii, and gradients then flow back to it too. If omitted, the
    radii stored in the result are used, as constants.
    """
    require_result(result, dimension=3, representation="topology")
    coordinates, _ = reconstruct_topology_vertices(
        result,
        points,
        device,
        return_local_to_global=False,
        radii=radii,
    )
    return coordinates


def padded_cell_vertices_3d(result, points, device="cpu", radii=None):
    """Return differentiable cell-local coordinates as a padded tensor.

    For a radical tessellation, ``radii`` (a tensor of shape (N,)) gives the
    generator radii, and gradients then flow back to it too. If omitted, the
    radii stored in the result are used, as constants.
    """
    require_result(result, dimension=3, representation="topology")
    device = torch.device(device)
    vertices, local_to_global = _reconstruct_vertices_with_map(
        result, points, device, radii
    )
    device = vertices.device
    layout = cell_vertex_layout(result, device)
    # Clone before returning: `layout` is cached per (result, device) and
    # reused by every future call -- handing out the cached tensor directly
    # would let an in-place op on the caller's side (e.g. `counts += 1`)
    # permanently corrupt it for this result, the same risk already
    # guarded against for `_reconstruction_plan`'s lhs/rhs templates.
    counts = layout["counts"].clone()
    output = torch.zeros(
        (result.n_cells, layout["max_vertices"], 3),
        dtype=torch.float64,
        device=device,
    )
    if local_to_global.numel() == 0:
        return output, counts

    local_coordinates = vertices[local_to_global]
    output[layout["cell_ids"], layout["local_ids"]] = local_coordinates
    return output, counts


def unique_facets_3d(result, facet_type="all", device="cpu"):
    """Return deduplicated packed facets with global vertex IDs.

    `device` must match whatever device a paired `unique_vertices_3d`
    call for the same `result` uses, or vertex indices from the two
    will silently refer to different unique-vertex orderings.
    """
    require_result(result, dimension=3)
    return global_facets(result, facet_type, deduplicate=True, device=device)


def cell_polytopes_3d(result, device="cpu"):
    """Return complete cell→face→global-vertex connectivity.

    `device` must match whatever device a paired `unique_vertices_3d`
    call for the same `result` uses, or vertex indices from the two
    will silently refer to different unique-vertex orderings.
    """
    require_result(result, dimension=3)
    faces, _ = global_facets(
        result,
        "all",
        deduplicate=False,
        device=device,
    )
    offsets = result.cell_facet_offsets.tolist()
    cells = [
        faces[start:stop]
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    return cells, list(range(result.n_cells))


def surface_mesh_3d(result, points=None, device="cpu"):
    """The 3D implementation of surface_mesh.

    The mesh contains the faces between inner and outer cells, and the faces
    of inner cells on the boundary of the domain. Each face is listed once, with
    its vertices counter-clockwise as seen from outside the inner cell, so
    that the right-hand rule gives the normal pointing out of the inner
    region.

    Parameters
    ----------
    result : VoronoiResult
        A 3D result computed with ``n_inner``. For the geometry and per-cell
        representations, the coordinates are those computed by Voro++. For the
        topology representation, they are recomputed from ``points``, and are
        differentiable with respect to them. Periodic domains, and results
        computed with ``neighbors=False``, are not supported.
    points : torch.Tensor or array_like, shape (N, 3), optional
        Generator positions, required for the topology representation.
    device : str or torch.device
        Device for the returned tensors.

    Returns
    -------
    dict
        ``facet_vertex_ids`` (F, max_vertices), padded with -100;
        ``vertex_coords`` (V, 3); ``n_vertices_per_facet`` (F,);
        ``facet_mask`` (F, max_vertices); and ``facet_neighbors`` (F,),
        the outer cell or boundary code on the other side of each face.
    """
    require_result(result, dimension=3)
    device = torch.device(device)
    facet_selection = selected_facet_indices(result, "surface")
    if facet_selection[0].size == 0:
        return _empty_surface_mesh(device)

    vertex_map = global_vertex_map(result, device)
    _, local_to_global = vertex_map
    if result.representation == "topology":
        if points is None:
            raise ValueError("points are required for topology reconstruction")
        coordinates, _ = reconstruct_topology_vertices(
            result,
            points,
            device,
            vertex_map=vertex_map,
            return_local_to_global=False,
        )
    else:
        coordinates = unique_geometry_coordinates(
            result, local_to_global, device
        )

    facet_vertices, facet_offsets, info = packed_global_facets(
        result,
        "surface",
        deduplicate=False,
        local_to_global=local_to_global,
        facet_selection=facet_selection,
        device=device,
    )

    used, compact_vertices = unique_1d(facet_vertices, device)
    counts_array = np.diff(facet_offsets)
    counts = torch.as_tensor(
        counts_array, dtype=torch.int64, device=device
    )
    width = int(counts_array.max())
    face_ids_array = np.full(
        (counts_array.size, width), -100, dtype=np.int64
    )
    rows = np.repeat(
        np.arange(counts_array.size, dtype=np.int64), counts_array
    )
    columns = (
        np.arange(facet_vertices.size, dtype=np.int64)
        - np.repeat(facet_offsets[:-1], counts_array)
    )
    # Voro++ lists the vertices of each face clockwise, as seen from outside
    # the cell. Reverse them, so that they are counter-clockwise as seen from
    # outside the inner cell, and the right-hand rule gives the outward normal.
    columns = np.repeat(counts_array, counts_array) - 1 - columns
    face_ids_array[rows, columns] = compact_vertices
    face_ids = torch.as_tensor(
        face_ids_array, dtype=torch.int64, device=device
    )
    mask = face_ids != -100

    return {
        "facet_vertex_ids": face_ids,
        "vertex_coords": coordinates[
            torch.as_tensor(used, dtype=torch.int64, device=device)
        ],
        "n_vertices_per_facet": counts,
        "facet_mask": mask,
        "facet_neighbors": torch.as_tensor(
            info[:, 1], dtype=torch.int64, device=device
        ),
    }
