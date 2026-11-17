"""2D reconstruction and connectivity for packed Voronoi results."""

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
        "facet_vertex_ids": torch.empty((0, 2), dtype=torch.int64, device=device),
        "vertex_coords": torch.empty((0, 2), dtype=torch.float64, device=device),
        "n_vertices_per_facet": torch.empty(0, dtype=torch.int64, device=device),
        "facet_mask": torch.empty((0, 2), dtype=torch.bool, device=device),
        "facet_neighbors": torch.empty(0, dtype=torch.int64, device=device),
    }


def _reconstruct_vertices_with_map_2d(result, points, device, radii=None):
    require_result(result, dimension=2, representation="topology")
    return reconstruct_topology_vertices(
        result,
        points,
        device,
        return_local_to_global=True,
        radii=radii,
    )


def unique_vertices_2d(result, points, device="cpu", radii=None):
    """Reconstruct differentiable unique 2D vertices from packed topology.

    For a radical tessellation, ``radii`` (a tensor of shape (N,)) gives the
    generator radii, and gradients then flow back to it too. If omitted, the
    radii stored in the result are used, as constants.
    """
    require_result(result, dimension=2, representation="topology")
    coordinates, _ = reconstruct_topology_vertices(
        result,
        points,
        device,
        return_local_to_global=False,
        radii=radii,
    )
    return coordinates


def padded_cell_vertices_2d(result, points, device="cpu", radii=None):
    """Return differentiable cell-local coordinates as a padded tensor.

    For a radical tessellation, ``radii`` (a tensor of shape (N,)) gives the
    generator radii, and gradients then flow back to it too. If omitted, the
    radii stored in the result are used, as constants.
    """
    require_result(result, dimension=2, representation="topology")
    device = torch.device(device)
    vertices, local_to_global = _reconstruct_vertices_with_map_2d(
        result, points, device, radii
    )
    device = vertices.device
    layout = cell_vertex_layout(result, device)
    # Clone before returning: see the identical note in
    # compute_padded_cell_vertices (reconstruct.py) -- `layout` is cached per
    # (result, device), so returning it directly would let a caller's
    # in-place op permanently corrupt future calls for this result.
    counts = layout["counts"].clone()
    output = torch.zeros(
        (result.n_cells, layout["max_vertices"], 2),
        dtype=torch.float64,
        device=device,
    )
    if local_to_global.numel() == 0:
        return output, counts

    local_coordinates = vertices[local_to_global]
    output[layout["cell_ids"], layout["local_ids"]] = local_coordinates
    return output, counts


def unique_facets_2d(result, facet_type="all", device="cpu"):
    """Return deduplicated packed edges with global vertex IDs.

    `device` must match whatever device a paired `unique_vertices_2d`
    call for the same `result` uses, or vertex indices from the two
    will silently refer to different unique-vertex orderings.
    """
    require_result(result, dimension=2)
    return global_facets(result, facet_type, deduplicate=True, device=device)


def cell_polytopes_2d(result, device="cpu"):
    """Return one ordered global-vertex cycle for each computed cell.

    `device` must match whatever device a paired `unique_vertices_2d`
    call for the same `result` uses, or vertex indices from the two
    will silently refer to different unique-vertex orderings.
    """
    require_result(result, dimension=2)
    edge_vertices, edge_offsets, _ = packed_global_facets(
        result,
        "all",
        deduplicate=False,
        device=device,
    )
    if not np.all(np.diff(edge_offsets) == 2):
        raise ValueError("2D Voronoi edges must contain two vertices")
    first_vertices = edge_vertices[edge_offsets[:-1]]
    offsets = result.cell_facet_offsets.tolist()
    polygons = [
        first_vertices[start:stop].tolist()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    return polygons, list(range(result.n_cells))


def surface_mesh_2d(result, points=None, device="cpu"):
    """The 2D implementation of surface_mesh."""
    require_result(result, dimension=2)
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

    edge_vertices, edge_offsets, info = packed_global_facets(
        result,
        "surface",
        deduplicate=False,
        local_to_global=local_to_global,
        facet_selection=facet_selection,
        device=device,
    )

    if not np.all(np.diff(edge_offsets) == 2):
        raise ValueError("2D Voronoi edges must contain two vertices")
    used, inverse = unique_1d(edge_vertices, device)
    n_edges = edge_offsets.size - 1
    return {
        "facet_vertex_ids": torch.as_tensor(
            inverse.reshape((-1, 2)), dtype=torch.int64, device=device
        ),
        "vertex_coords": coordinates[
            torch.as_tensor(used, dtype=torch.int64, device=device)
        ],
        "n_vertices_per_facet": torch.full(
            (n_edges,), 2, dtype=torch.int64, device=device
        ),
        "facet_mask": torch.ones((n_edges, 2), dtype=torch.bool, device=device),
        "facet_neighbors": torch.as_tensor(
            info[:, 1], dtype=torch.int64, device=device
        ),
    }
