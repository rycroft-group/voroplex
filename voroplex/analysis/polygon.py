"""Differentiable geometry helpers for 2D polygonal Voronoi cells."""

import numpy as np
import torch


def _unique_rows_torch(tensor, return_inverse=False, return_counts=False):
    """Device-aware row-wise unique for a 2D non-negative integer tensor.

    torch.unique(dim=0) is fast on CUDA (a massively parallel sort) but far
    slower than a numpy-based approach on CPU -- the same finding
    established for VoroPlex's core packing layer (see
    voroplex._mesh._packed._unique_rows). This dispatch keeps that path
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


def _as_tensor(value, *, dtype=None, device=None):
    if isinstance(value, torch.Tensor):
        options = {}
        if dtype is not None:
            options["dtype"] = dtype
        if device is not None:
            options["device"] = device
        return value.to(**options) if options else value
    return torch.as_tensor(value, dtype=dtype, device=device)


def _pairs_to_adjacency(pairs, n_cells):
    if pairs.numel() == 0:
        return (
            torch.empty(
                (n_cells, 0), dtype=torch.int64, device=pairs.device
            ),
            torch.zeros(
                n_cells, dtype=torch.int64, device=pairs.device
            ),
        )

    sources = torch.cat([pairs[:, 0], pairs[:, 1]])
    targets = torch.cat([pairs[:, 1], pairs[:, 0]])
    order = torch.argsort(sources * n_cells + targets)
    sources = sources[order]
    targets = targets[order]
    counts = torch.bincount(sources, minlength=n_cells)
    width = int(counts.max())
    starts = torch.cumsum(counts, dim=0) - counts
    columns = (
        torch.arange(sources.numel(), device=pairs.device)
        - torch.repeat_interleave(starts, counts)
    )
    adjacency = torch.full(
        (n_cells, width), -1, dtype=torch.int64, device=pairs.device
    )
    adjacency[sources, columns] = targets
    return adjacency, counts


def _build_adjacency(cell_vertex_ids, cell_mask):
    device = cell_vertex_ids.device
    n_cells, width = cell_vertex_ids.shape
    safe_ids = cell_vertex_ids.masked_fill(~cell_mask, 0)
    positions = torch.arange(width, device=device).unsqueeze(0)
    counts = cell_mask.sum(dim=1)
    next_positions = torch.where(
        positions + 1 < counts[:, None],
        positions + 1,
        torch.zeros_like(positions),
    )
    next_ids = torch.gather(safe_ids, 1, next_positions)
    edges = torch.stack(
        [safe_ids[cell_mask], next_ids[cell_mask]], dim=1
    )
    edges = torch.sort(edges, dim=1).values
    owners = torch.arange(n_cells, device=device)[:, None].expand_as(
        cell_vertex_ids
    )[cell_mask]

    unique_edges, inverse, edge_counts = _unique_rows_torch(
        edges, return_inverse=True, return_counts=True
    )
    sorted_edges, order = torch.sort(inverse)
    sorted_owners = owners[order]
    edge_starts = torch.cumsum(edge_counts, dim=0) - edge_counts
    max_owners = int(edge_counts.max()) if edge_counts.numel() else 0
    edge_owners = torch.full(
        (unique_edges.shape[0], max_owners),
        -1,
        dtype=torch.int64,
        device=device,
    )
    owner_positions = (
        torch.arange(sorted_owners.numel(), device=device)
        - torch.repeat_interleave(edge_starts, edge_counts)
    )
    edge_owners[sorted_edges, owner_positions] = sorted_owners

    pair_columns = torch.triu_indices(
        max_owners, max_owners, offset=1, device=device
    )
    if pair_columns.shape[1]:
        pairs = torch.stack(
            [
                edge_owners[:, pair_columns[0]],
                edge_owners[:, pair_columns[1]],
            ],
            dim=-1,
        ).reshape(-1, 2)
        valid_pairs = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 0] != pairs[:, 1])
        )
        pairs = pairs[valid_pairs]
        pairs = torch.sort(pairs, dim=1).values
        pairs = _unique_rows_torch(pairs)
    else:
        pairs = torch.empty((0, 2), dtype=torch.int64, device=device)

    adjacency, n_adjacent = _pairs_to_adjacency(pairs, n_cells)
    return pairs, unique_edges, adjacency, n_adjacent


def compute_polygon_geometry(
    vertex_coords,
    cell_vertex_ids,
    n_vertices_per_cell,
    cell_mask,
):
    """
    Compute geometry and adjacency for padded 2D polygon cells.

    Parameters
    ----------
    vertex_coords : torch.Tensor, shape (N_vertices, 2)
        Global vertex coordinate table.
    cell_vertex_ids : torch.Tensor, shape (N_cells, max_vertices)
        Vertex IDs for each polygon cell, padded with any value where
        ``cell_mask`` is False.
    n_vertices_per_cell : torch.Tensor, shape (N_cells,)
        Number of valid vertices per cell.
    cell_mask : torch.Tensor, shape (N_cells, max_vertices)
        Boolean mask for valid entries in ``cell_vertex_ids``.

    Returns
    -------
    dict
        Differentiable geometry, with respect to ``vertex_coords``:
        ``cell_areas`` (N_cells,), ``cell_perimeters`` (N_cells,),
        ``cell_centroids`` (N_cells, 2), the area centroids, and
        ``edge_lengths`` (N_cells, max_vertices), zero in padded entries.
        Integer connectivity: ``cell_neighbor_pairs``, ``unique_edges``,
        ``adjacency_tensor``, padded with -1, and ``n_adjacent``. Also
        ``cell_mask``.
    """
    coords = _as_tensor(vertex_coords)
    device = coords.device
    dtype = coords.dtype
    ids = _as_tensor(cell_vertex_ids, dtype=torch.int64, device=device)
    num_verts = _as_tensor(n_vertices_per_cell, dtype=torch.int64, device=device)
    mask = _as_tensor(cell_mask, dtype=torch.bool, device=device)

    n_cells, max_verts = ids.shape
    safe_ids = ids.masked_fill(~mask, 0)
    # Coordinates relative to the first vertex of each cell, so that the
    # shoelace formula does not lose precision for small cells far from the
    # origin.
    origin = coords[safe_ids[:, 0]].unsqueeze(1)
    gathered = coords[safe_ids] - origin
    positions = torch.arange(max_verts, device=device).unsqueeze(0)
    next_positions = torch.where(
        positions + 1 < num_verts[:, None],
        positions + 1,
        torch.zeros_like(positions),
    )
    next_ids = torch.gather(safe_ids, 1, next_positions)
    next_vertices = coords[next_ids] - origin

    cross = (
        gathered[..., 0] * next_vertices[..., 1]
        - next_vertices[..., 0] * gathered[..., 1]
    ) * mask.to(dtype)
    signed_areas = 0.5 * cross.sum(dim=1)
    cell_areas = signed_areas.abs()

    edge_lengths = torch.linalg.norm(
        next_vertices - gathered, dim=2
    ) * mask.to(dtype)
    cell_perimeters = edge_lengths.sum(dim=1)

    # The centroid formula divides by the area, so a cell whose area is
    # negligible compared with its perimeter squared uses the vertex mean.
    denominator = 6.0 * signed_areas
    regular = denominator.abs() > 1.0e-12 * cell_perimeters ** 2
    safe_denominator = torch.where(
        regular, denominator, torch.ones_like(denominator)
    )
    centroid_numerator = torch.stack(
        [
            ((gathered[..., 0] + next_vertices[..., 0]) * cross).sum(
                dim=1
            ),
            ((gathered[..., 1] + next_vertices[..., 1]) * cross).sum(
                dim=1
            ),
        ],
        dim=1,
    )
    polygon_centroids = centroid_numerator / safe_denominator[:, None] + origin[:, 0]
    mean_centroids = (
        (gathered * mask[..., None]).sum(dim=1)
        / num_verts.clamp(min=1).to(dtype)[:, None]
    ) + origin[:, 0]
    cell_centroids = torch.where(
        regular[:, None], polygon_centroids, mean_centroids
    )

    cell_neighbor_pairs, unique_edges, adjacency, n_adjacent = _build_adjacency(
        ids, mask
    )

    return {
        "cell_areas": cell_areas,
        "cell_perimeters": cell_perimeters,
        "cell_centroids": cell_centroids,
        "edge_lengths": edge_lengths,
        "cell_neighbor_pairs": cell_neighbor_pairs,
        "unique_edges": unique_edges,
        "adjacency_tensor": adjacency,
        "n_adjacent": n_adjacent,
        "cell_mask": mask,
    }
