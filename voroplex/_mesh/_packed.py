"""Shared adapters for packed Voronoi results."""

import numpy as np
import torch


_SIGNATURE_PAD = np.iinfo(np.int32).min
# Relative tolerance for a singular square vertex system (see
# _solve_vertex_systems).
_SINGULAR_TOLERANCE = 1.0e-12


def require_result(result, dimension=None, representation=None):
    if not hasattr(result, "representation"):
        raise TypeError("expected a VoronoiResult")
    if result.representation is None:
        raise ValueError(
            "this operation needs the cells, but the result was computed with "
            "representation=None, which only stores the requested properties"
        )
    if dimension is not None and result.dimension != dimension:
        raise ValueError("expected a {}D VoronoiResult".format(dimension))
    if representation is not None and result.representation != representation:
        raise ValueError(
            "this operation requires representation={!r}".format(representation)
        )


def _unique_rows(rows, return_index=False, return_inverse=False):
    """Fast replacement for np.unique(rows, axis=0, ...).

    NumPy's axis=0 path compares rows element-by-element and is dramatically
    slower than its 1D path for large arrays. Viewing each row as a single
    opaque fixed-width record lets NumPy dedupe with the fast 1D sort
    instead (byte-wise ordering differs from numeric ordering when values
    are negative, but that's irrelevant here: only grouping identical rows
    and reconstructing them via `unique[inverse]` matters, not row order).
    """
    contiguous = np.ascontiguousarray(rows)
    void_dtype = np.dtype(
        (np.void, contiguous.dtype.itemsize * contiguous.shape[1])
    )
    flat = contiguous.view(void_dtype).reshape(-1)
    result = np.unique(
        flat, return_index=return_index, return_inverse=return_inverse
    )
    if return_index or return_inverse:
        unique_flat = result[0]
        rest = result[1:]
    else:
        unique_flat = result
        rest = ()
    unique = unique_flat.view(contiguous.dtype).reshape(-1, contiguous.shape[1])
    return (unique,) + rest if rest else unique


def _unique_rows_first_occurrence(rows, device):
    """Return (unique_rows, first_occurrence_index) for a small-width 2D
    integer array, dispatching like `_unique_rows`/`global_vertex_map`: on
    CUDA via torch.unique(dim=0) (a scatter-reduce recovers the first
    original index per group, matching numpy's return_index exactly --
    verified against the numpy reference), on CPU via `_unique_rows`
    (torch.unique(dim=0) benchmarked slower than numpy on CPU, same
    finding as global_vertex_map).
    """
    device = torch.device(device)
    if device.index is None and device.type == "cuda":
        device = torch.device(device.type, torch.cuda.current_device())
    if device.type != "cuda":
        return _unique_rows(rows, return_index=True)

    n = rows.shape[0]
    tensor = torch.tensor(rows, dtype=torch.int64, device=device)
    unique_tensor, inverse = torch.unique(tensor, dim=0, return_inverse=True)
    positions = torch.arange(n, dtype=torch.int64, device=device)
    first = torch.full(
        (unique_tensor.shape[0],), n, dtype=torch.int64, device=device
    )
    first = first.scatter_reduce(
        0, inverse, positions, reduce="amin", include_self=True
    )
    return unique_tensor.cpu().numpy(), first.cpu().numpy()


def unique_1d(values, device="cpu"):
    """Device-aware replacement for np.unique(values, return_inverse=True)
    on a 1D integer array. Unlike the axis=0 row-dedup case, numpy's 1D
    unique is already the fast path on CPU (no row-vs-1D asymmetry to
    work around) -- but it's still a single-core sort, and torch.unique
    on CUDA is ~100x faster at real problem sizes (benchmarked: 2.17s
    numpy vs 0.02s torch/CUDA at 12M elements), so the same device
    dispatch is worthwhile here too.
    """
    device = torch.device(device)
    if device.index is None and device.type == "cuda":
        device = torch.device(device.type, torch.cuda.current_device())
    if device.type != "cuda":
        return np.unique(values, return_inverse=True)
    tensor = torch.tensor(values, dtype=torch.int64, device=device)
    unique_tensor, inverse_tensor = torch.unique(tensor, return_inverse=True)
    return (
        unique_tensor.cpu().numpy(),
        inverse_tensor.cpu().numpy().astype(np.int64, copy=False),
    )


def _ragged_signature_matrix(offsets, values):
    """Pack sorted ragged signatures into a padded integer matrix."""
    counts = np.diff(offsets)
    n_rows = counts.size
    width = int(counts.max()) if n_rows else 0
    matrix = np.full((n_rows, width), _SIGNATURE_PAD, dtype=np.int32)
    if values.size:
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
        positions = (
            np.arange(values.size, dtype=np.int64)
            - np.repeat(offsets[:-1], counts)
        )
        matrix[rows, positions] = values
    return matrix


def _ragged_signature_matrix_torch(offsets, values, device):
    """GPU counterpart of `_ragged_signature_matrix`.

    Transfers only the raw ragged CSR arrays (offsets/values, minimal
    size) and does the repeat + fancy-index scatter that pads them into a
    dense matrix directly on the GPU, instead of building the (larger,
    since padding wastes space for uneven rows) padded matrix on
    single-threaded CPU first and transferring that. Measured at
    N=1,000,000 (3D): this step alone was 2.22s of the 2.90s
    compute_unique_vertices(device="cuda") call -- 77%, dwarfing both the
    actual GPU dedup compute and the CPU<->GPU transfer combined.
    """
    # torch.tensor (not as_tensor): offsets/values are read-only VoronoiResult
    # arrays, and as_tensor's zero-copy CPU view would warn about that even
    # though the array ends up copied to `device` regardless.
    offsets_t = torch.tensor(offsets, dtype=torch.int64, device=device)
    counts = offsets_t[1:] - offsets_t[:-1]
    n_rows = counts.shape[0]
    width = int(counts.max().item()) if n_rows else 0
    matrix = torch.full(
        (n_rows, width), _SIGNATURE_PAD, dtype=torch.int32, device=device
    )
    if values.size:
        values_t = torch.tensor(values, dtype=torch.int32, device=device)
        rows = torch.repeat_interleave(
            torch.arange(n_rows, dtype=torch.int64, device=device), counts
        )
        positions = torch.arange(
            values_t.numel(), dtype=torch.int64, device=device
        ) - torch.repeat_interleave(offsets_t[:-1], counts)
        matrix[rows, positions] = values_t
    return matrix


def _geometry_signature_matrix(result):
    """Build local-vertex signatures from packed geometry connectivity."""
    n_local_vertices = int(result.cell_vertex_offsets[-1])
    local_owners = np.repeat(
        np.arange(result.n_cells, dtype=np.int32),
        np.diff(result.cell_vertex_offsets),
    )

    facet_owners_array = facet_owners(result)
    facet_ids = np.repeat(
        np.arange(result.facet_neighbors.size, dtype=np.int64),
        np.diff(result.facet_vertex_offsets),
    )
    reference_owners = facet_owners_array[facet_ids]
    local_vertex_ids = (
        result.cell_vertex_offsets[reference_owners]
        + result.facet_vertices
    )

    signature_rows = np.concatenate(
        [
            np.arange(n_local_vertices, dtype=np.int64),
            local_vertex_ids,
        ]
    )
    signature_values = np.concatenate(
        [
            local_owners,
            result.facet_neighbors[facet_ids],
        ]
    )
    order = np.lexsort((signature_values, signature_rows))
    signature_rows = signature_rows[order]
    signature_values = signature_values[order]

    keep = np.ones(signature_rows.size, dtype=bool)
    keep[1:] = (
        (signature_rows[1:] != signature_rows[:-1])
        | (signature_values[1:] != signature_values[:-1])
    )
    signature_rows = signature_rows[keep]
    signature_values = signature_values[keep]

    counts = np.bincount(
        signature_rows, minlength=n_local_vertices
    ).astype(np.int64, copy=False)
    offsets = np.empty(n_local_vertices + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return _ragged_signature_matrix(offsets, signature_values)


def local_vertex_signature_matrix(result, device="cpu"):
    """Return one padded sorted signature row per local vertex.

    For `representation="topology"`, dispatches on `device`: CUDA builds
    the padded matrix directly on the GPU from the raw ragged arrays (see
    `_ragged_signature_matrix_torch`) and returns a tensor; CPU returns a
    NumPy array as before. The geometry-mode path (`_geometry_signature_matrix`,
    a lexsort-based rebuild from raw facet connectivity, not a simple CSR
    pad) is not yet GPU-dispatched and always returns NumPy.
    """
    if np.any(result.periodic):
        # In a periodic domain, two different vertices can have the same
        # generator IDs (from different periodic images), so vertices cannot
        # be matched between cells by their signatures.
        raise NotImplementedError(
            "meshes with shared vertices are not available for periodic "
            "domains, since the vertex signatures do not record periodic "
            "images; use the per-cell vertices instead"
        )
    if result.has_signatures:
        device = torch.device(device)
        if device.index is None and device.type == "cuda":
            device = torch.device(device.type, torch.cuda.current_device())
        if device.type == "cuda":
            return _ragged_signature_matrix_torch(
                result.vertex_signature_offsets,
                result.vertex_signature_ids,
                device,
            )
        return _ragged_signature_matrix(
            result.vertex_signature_offsets,
            result.vertex_signature_ids,
        )

    return _geometry_signature_matrix(result)


def global_vertex_map(result, device="cpu"):
    """Return unique padded signatures and local-to-global vertex IDs.

    Deduplication runs on `device`: on CUDA it uses torch.unique(dim=0),
    which is ~200x faster than any CPU path at real problem sizes (a
    single-core CPU sort of tens of millions of rows vs. CUDA's massively
    parallel sort); on CPU it uses a numpy 1D-sort trick, since
    torch.unique(dim=0) is not competitive with numpy on CPU (both were
    benchmarked; torch was slower than even numpy's naive axis=0 path).
    Cached per-device on `result`, since the two code paths are used by
    calls requesting different target devices for the same result.
    """
    device = torch.device(device)
    if device.index is None and device.type == "cuda":
        # Normalize the bare "cuda" spelling to its concrete index so it
        # shares a cache entry (and vertex ordering) with "cuda:N" calls
        # for the same physical GPU, rather than silently diverging.
        device = torch.device(device.type, torch.cuda.current_device())
    cache = getattr(result, "_global_vertex_map_cache", None)
    if cache is None:
        cache = {}
        result._global_vertex_map_cache = cache
    device_key = (device.type, device.index)
    cached = cache.get(device_key)
    if cached is not None:
        return cached

    signatures = local_vertex_signature_matrix(result, device)
    if signatures.shape[0] == 0:
        if isinstance(signatures, torch.Tensor):
            signatures = signatures.cpu().numpy()
        inverse = np.empty(0, dtype=np.int64)
        cached = signatures, inverse
    elif device.type == "cuda":
        # `signatures` is already a GPU tensor when local_vertex_signature_matrix
        # dispatched to CUDA (topology mode); geometry mode still returns
        # NumPy there, so transfer it here instead.
        tensor = (
            signatures if isinstance(signatures, torch.Tensor)
            else torch.from_numpy(signatures).to(device)
        )
        unique_tensor, inverse_tensor = torch.unique(
            tensor, dim=0, return_inverse=True
        )
        unique = unique_tensor.cpu().numpy()
        inverse = inverse_tensor.cpu().numpy().astype(np.int64, copy=False)
        cached = unique, inverse

        # Stash the tensors already computed on GPU so downstream torch-path
        # consumers (_signatures_tensor, _local_to_global_tensor) can reuse
        # them directly instead of re-uploading the data we just pulled back
        # to CPU -- eliminates a redundant D2H-then-H2D round trip (measured
        # at N=1,000,000, 3D: ~0.21s of a fresh compute_unique_vertices(cuda)
        # call).
        sig_cache = getattr(result, "_signatures_tensor_cache", None)
        if sig_cache is None:
            sig_cache = {}
            result._signatures_tensor_cache = sig_cache
        sig_cache[(device.type, device.index, id(unique))] = unique_tensor

        ltg_cache = getattr(result, "_local_to_global_tensor_cache", None)
        if ltg_cache is None:
            ltg_cache = {}
            result._local_to_global_tensor_cache = ltg_cache
        ltg_cache[(device.type, device.index, id(inverse))] = (
            inverse_tensor
            if inverse_tensor.dtype == torch.int64
            else inverse_tensor.to(torch.int64)
        )
    else:
        unique, inverse = _unique_rows(signatures, return_inverse=True)
        cached = unique, inverse.astype(np.int64, copy=False)

    for array in cached:
        array.setflags(write=False)
    cache[device_key] = cached
    return cached


def cell_vertex_layout(result, device):
    """Return cached tensors that place packed vertices into padded cells."""
    device = torch.device(device)
    if device.index is None and device.type == "cuda":
        device = torch.device(device.type, torch.cuda.current_device())
    device_key = (device.type, device.index)
    cache = getattr(result, "_cell_vertex_layout_cache", None)
    if cache is None:
        cache = {}
        result._cell_vertex_layout_cache = cache
    cached = cache.get(device_key)
    if cached is not None:
        return cached

    counts_array = np.diff(result.cell_vertex_offsets)
    total = int(result.cell_vertex_offsets[-1])
    cell_ids_array = np.repeat(
        np.arange(result.n_cells, dtype=np.int64), counts_array
    )
    local_ids_array = (
        np.arange(total, dtype=np.int64)
        - np.repeat(result.cell_vertex_offsets[:-1], counts_array)
    )
    cached = {
        "counts": torch.as_tensor(
            counts_array, dtype=torch.int64, device=device
        ),
        "cell_ids": torch.as_tensor(
            cell_ids_array, dtype=torch.int64, device=device
        ),
        "local_ids": torch.as_tensor(
            local_ids_array, dtype=torch.int64, device=device
        ),
        "max_vertices": int(counts_array.max()) if counts_array.size else 0,
    }
    cache[device_key] = cached
    return cached


def _reconstruction_plan(
    result,
    signatures,
    inverse,
    device,
    include_inverse,
):
    """Return cached device tensors for topology-system assembly."""
    device = torch.device(device)
    if device.index is None and device.type == "cuda":
        device = torch.device(device.type, torch.cuda.current_device())
    device_key = (device.type, device.index)
    cache = getattr(result, "_reconstruction_plan_cache", None)
    if cache is None:
        cache = {}
        result._reconstruction_plan_cache = cache
    cached = cache.get(device_key)
    if (
        cached is not None
        and cached["signatures"] is signatures
        and cached["inverse_array"] is inverse
    ):
        if include_inverse and cached["inverse"] is None:
            cached["inverse"] = torch.tensor(
                np.array(inverse, copy=True),
                dtype=torch.int64,
                device=device,
            )
        return cached

    builder = (
        _build_reconstruction_plan_torch
        if device.type == "cuda"
        else _build_reconstruction_plan_numpy
    )
    cached = builder(result, signatures, inverse, device, include_inverse)
    cache[device_key] = cached
    return cached


def _build_reconstruction_plan_numpy(
    result, signatures, inverse, device, include_inverse
):
    """CPU builder: numpy masks/nonzero/fancy-indexing throughout.

    Benchmarked against a torch CPU equivalent: numpy wins on CPU (this
    is the same lesson as the vertex-dedup fix -- torch's elementwise/
    nonzero/indexing ops are not tuned for single-core CPU execution the
    way numpy's are).
    """
    valid = signatures != _SIGNATURE_PAD
    if np.any(
        valid[:, :-1]
        & valid[:, 1:]
        & (signatures[:, :-1] > signatures[:, 1:])
    ):
        raise ValueError("vertex signatures must be sorted")
    walls = valid & (signatures < 0)
    generators = valid & (signatures >= 0)
    wall_counts = walls.sum(axis=1, dtype=np.int64)
    generator_counts = generators.sum(axis=1, dtype=np.int64)
    equation_counts = wall_counts + generator_counts - 1

    if np.any(generator_counts == 0):
        raise ValueError("a vertex signature has no generator IDs")
    if np.any(equation_counts < result.dimension):
        raise ValueError("a vertex signature supplies too few constraints")
    generator_ids = signatures[generators]
    if np.any(generator_ids >= result.n_points):
        raise ValueError("a vertex signature references an unknown generator")
    wall_codes = -signatures[walls] - 1
    if np.any(wall_codes >= 2 * result.dimension):
        raise ValueError("a vertex signature contains an unknown wall code")

    n_vertices = signatures.shape[0]
    max_equations = int(equation_counts.max())
    base_ids = signatures[
        np.arange(n_vertices, dtype=np.int64), wall_counts
    ].astype(np.int64, copy=False)

    wall_rows, wall_columns = np.nonzero(walls)
    wall_indices = (
        -signatures[wall_rows, wall_columns] - 1
    ).astype(np.int64, copy=False)
    wall_axes = wall_indices // 2
    lhs_template = np.zeros(
        (n_vertices, max_equations, result.dimension), dtype=np.float64
    )
    rhs_template = np.zeros(
        (n_vertices, max_equations), dtype=np.float64
    )
    lhs_template[wall_rows, wall_columns, wall_axes] = 1.0
    rhs_template[wall_rows, wall_columns] = result.limits.reshape(-1)[
        wall_indices
    ]

    columns = np.arange(signatures.shape[1], dtype=np.int64)
    non_base_generators = generators & (
        columns[None, :] > wall_counts[:, None]
    )
    generator_rows, generator_columns = np.nonzero(non_base_generators)
    equation_rows = generator_columns - 1
    other_ids = signatures[
        generator_rows, generator_columns
    ].astype(np.int64, copy=False)
    reference_ids = base_ids[generator_rows]
    system_groups = tuple(
        (
            int(n_equations),
            torch.as_tensor(
                np.flatnonzero(equation_counts == n_equations),
                dtype=torch.int64,
                device=device,
            ),
        )
        for n_equations in np.unique(equation_counts)
    )

    return {
        "signatures": signatures,
        "inverse_array": inverse,
        "equation_counts": equation_counts,
        "lhs_template": torch.as_tensor(
            lhs_template, dtype=torch.float64, device=device
        ),
        "rhs_template": torch.as_tensor(
            rhs_template, dtype=torch.float64, device=device
        ),
        "generator_rows": torch.as_tensor(
            generator_rows, dtype=torch.int64, device=device
        ),
        "equation_rows": torch.as_tensor(
            equation_rows, dtype=torch.int64, device=device
        ),
        "other_ids": torch.as_tensor(
            other_ids, dtype=torch.int64, device=device
        ),
        "reference_ids": torch.as_tensor(
            reference_ids, dtype=torch.int64, device=device
        ),
        "base_ids": torch.as_tensor(base_ids, dtype=torch.int64, device=device),
        "wall_rows": torch.as_tensor(wall_rows, dtype=torch.int64, device=device),
        "wall_columns": torch.as_tensor(
            wall_columns, dtype=torch.int64, device=device
        ),
        "wall_axes": torch.as_tensor(wall_axes, dtype=torch.int64, device=device),
        "system_groups": system_groups,
        "inverse": (
            torch.tensor(
                np.array(inverse, copy=True),
                dtype=torch.int64,
                device=device,
            )
            if include_inverse
            else None
        ),
    }


def _signatures_tensor(result, signatures, device):
    """Return a cached device tensor for the deduplicated `signatures` array.

    Mirrors `_local_to_global_tensor`: `signatures` (the `unique` array
    returned by `global_vertex_map`) is stable for the lifetime of
    `result`, so keying on its identity lets a CUDA-computed dedup reuse
    the tensor it already produced instead of re-uploading the NumPy copy
    that was pulled back to CPU only to satisfy the CPU path's return
    type.
    """
    cache = getattr(result, "_signatures_tensor_cache", None)
    if cache is None:
        cache = {}
        result._signatures_tensor_cache = cache
    key = (device.type, device.index, id(signatures))
    cached = cache.get(key)
    if cached is not None:
        return cached
    cached = torch.tensor(signatures, dtype=torch.int64, device=device)
    cache[key] = cached
    return cached


def _build_reconstruction_plan_torch(
    result, signatures, inverse, device, include_inverse
):
    """CUDA builder: identical logic to the numpy version, but running the
    masks/reductions/nonzero/fancy-indexing entirely on the GPU via torch.

    This step is elementwise- and reduction-heavy over the (already
    deduplicated, much smaller than the raw per-cell signature matrix)
    unique-vertex array -- exactly the kind of work that stays CPU-bound
    if left in numpy even when the caller wants CUDA (measured: ~1.2s of
    numpy computation at N=1,000,000 that isn't the dedup step and isn't
    the final CPU->GPU transfer). Only `signatures`/`limits` (already
    small: signatures is the deduplicated array, limits is 2*dimension
    floats) cross the CPU->GPU boundary; everything else is derived
    on-device.
    """
    sig = _signatures_tensor(result, signatures, device)
    valid = sig != int(_SIGNATURE_PAD)
    if torch.any(
        valid[:, :-1]
        & valid[:, 1:]
        & (sig[:, :-1] > sig[:, 1:])
    ):
        raise ValueError("vertex signatures must be sorted")
    walls = valid & (sig < 0)
    generators = valid & (sig >= 0)
    wall_counts = walls.sum(dim=1, dtype=torch.int64)
    generator_counts = generators.sum(dim=1, dtype=torch.int64)
    equation_counts = wall_counts + generator_counts - 1

    if torch.any(generator_counts == 0):
        raise ValueError("a vertex signature has no generator IDs")
    if torch.any(equation_counts < result.dimension):
        raise ValueError("a vertex signature supplies too few constraints")
    if torch.any(sig[generators] >= result.n_points):
        raise ValueError("a vertex signature references an unknown generator")
    wall_codes = -sig[walls] - 1
    if torch.any(wall_codes >= 2 * result.dimension):
        raise ValueError("a vertex signature contains an unknown wall code")

    n_vertices = sig.shape[0]
    max_equations = int(equation_counts.max().item())
    base_ids = sig[
        torch.arange(n_vertices, dtype=torch.int64, device=device),
        wall_counts,
    ]

    wall_rows, wall_columns = torch.nonzero(walls, as_tuple=True)
    wall_indices = -sig[wall_rows, wall_columns] - 1
    wall_axes = wall_indices // 2
    lhs_template = torch.zeros(
        (n_vertices, max_equations, result.dimension),
        dtype=torch.float64,
        device=device,
    )
    rhs_template = torch.zeros(
        (n_vertices, max_equations), dtype=torch.float64, device=device
    )
    lhs_template[wall_rows, wall_columns, wall_axes] = 1.0
    limits_flat = torch.tensor(
        result.limits.reshape(-1), dtype=torch.float64, device=device
    )
    rhs_template[wall_rows, wall_columns] = limits_flat[wall_indices]

    columns = torch.arange(sig.shape[1], dtype=torch.int64, device=device)
    non_base_generators = generators & (
        columns[None, :] > wall_counts[:, None]
    )
    generator_rows, generator_columns = torch.nonzero(
        non_base_generators, as_tuple=True
    )
    equation_rows = generator_columns - 1
    other_ids = sig[generator_rows, generator_columns]
    reference_ids = base_ids[generator_rows]
    system_groups = tuple(
        (
            int(n_equations.item()),
            torch.nonzero(equation_counts == n_equations, as_tuple=True)[0],
        )
        for n_equations in torch.unique(equation_counts)
    )

    return {
        "signatures": signatures,
        "inverse_array": inverse,
        "equation_counts": equation_counts,
        "lhs_template": lhs_template,
        "rhs_template": rhs_template,
        "generator_rows": generator_rows,
        "equation_rows": equation_rows,
        "other_ids": other_ids,
        "reference_ids": reference_ids,
        "base_ids": base_ids.to(torch.int64),
        "wall_rows": wall_rows,
        "wall_columns": wall_columns,
        "wall_axes": wall_axes.to(torch.int64),
        "system_groups": system_groups,
        "inverse": (
            _local_to_global_tensor(result, inverse, device)
            if include_inverse
            else None
        ),
    }


def _solve_vertex_systems(lhs, rhs, system_groups):
    """Solve the vertex systems, grouped by their number of equations."""
    dimension = lhs.shape[2]
    coordinates = torch.empty(
        (lhs.shape[0], dimension),
        dtype=lhs.dtype,
        device=lhs.device,
    )

    for n_equations, indices in system_groups:
        matrices = lhs[indices, :n_equations]
        vectors = rhs[indices, :n_equations]

        if n_equations == dimension:
            # A genuine Voronoi vertex always gives a nonsingular system, so a
            # singular one means that the signatures do not describe this
            # tessellation. The test is scale-free: by Hadamard's inequality,
            # |det A| <= the product of the row norms, with equality for
            # orthogonal rows.
            determinants = torch.linalg.det(matrices).abs()
            row_norms = torch.linalg.vector_norm(matrices, dim=2).prod(dim=1)
            singular = determinants <= _SINGULAR_TOLERANCE * row_norms
            if bool(singular.any()):
                raise RuntimeError(
                    f"{int(singular.sum())} vertex systems are singular, so the "
                    "vertex signatures do not match the generator positions"
                )
            coordinates[indices] = torch.linalg.solve(matrices, vectors)
        else:
            # More than d constraints (a degenerate vertex): the system is
            # overdetermined but consistent, and least squares gives the
            # vertex.
            coordinates[indices] = torch.einsum(
                "bij,bj->bi",
                torch.linalg.pinv(matrices),
                vectors,
            )
    return coordinates


def reconstruct_topology_vertices(
    result,
    points,
    device,
    vertex_map=None,
    return_local_to_global=True,
    radii=None,
):
    """Reconstruct unique coordinates from ragged topology signatures.

    For a radical tessellation, radii (a tensor of shape (N,)) gives the
    generator radii, and gradients flow back to it. If omitted, the radii
    stored in the result are used, as constants.
    """
    require_result(result, representation="topology")
    device = torch.device(device)
    if isinstance(points, torch.Tensor):
        points_tensor = points.to(dtype=torch.float64, device=device)
    else:
        points_tensor = torch.tensor(points, dtype=torch.float64, device=device)
    device = points_tensor.device
    if points_tensor.ndim != 2 or points_tensor.shape[1] != result.dimension:
        raise ValueError(
            "points must have shape (n, {})".format(result.dimension)
        )
    if points_tensor.shape[0] != result.n_points:
        raise ValueError(
            "points must contain the same {} generators used for the "
            "Voronoi computation".format(result.n_points)
        )
    if radii is not None:
        if not result.is_radical:
            raise ValueError(
                "radii were given, but the tessellation was computed without "
                "radii"
            )
        radii_tensor = torch.as_tensor(radii).to(dtype=torch.float64, device=device)
        if radii_tensor.shape != (result.n_points,):
            raise ValueError(
                "radii must have shape ({},)".format(result.n_points)
            )
    elif result.is_radical:
        radii_tensor = torch.tensor(result.radii, dtype=torch.float64,
                                       device=device)
    else:
        radii_tensor = None

    if vertex_map is None:
        vertex_map = global_vertex_map(result, device)
    signatures, inverse = vertex_map
    if signatures.shape[0] == 0:
        empty = torch.empty(
            (0, result.dimension), dtype=torch.float64, device=device
        )
        local_to_global = (
            torch.empty(0, dtype=torch.int64, device=device)
            if return_local_to_global
            else None
        )
        return empty, local_to_global

    plan = _reconstruction_plan(
        result,
        signatures,
        inverse,
        device,
        include_inverse=return_local_to_global,
    )
    # The systems are solved for the position of each vertex relative to its
    # base generator p_1, w = v - p_1. A generator row is then
    # (p_j - p_1) . w = |p_j - p_1|^2 / 2, and a wall row is
    # w_axis = wall - p_1[axis]. Writing the rows in absolute coordinates,
    # (p_j - p_1) . v = (|p_j|^2 - |p_1|^2) / 2, would lose precision through
    # cancellation for small cells far from the origin. In a radical
    # tessellation, a generator row has the extra term -(r_j^2 - r_1^2) / 2.
    lhs = plan["lhs_template"].clone()
    rhs = plan["rhs_template"].clone()
    base_points = points_tensor[plan["base_ids"]]
    differences = (
        points_tensor[plan["other_ids"]] - points_tensor[plan["reference_ids"]]
    )
    lhs[plan["generator_rows"], plan["equation_rows"]] = differences
    generator_rhs = torch.sum(differences.square(), dim=1)
    if radii_tensor is not None:
        generator_rhs = generator_rhs - (
            radii_tensor[plan["other_ids"]].square()
            - radii_tensor[plan["reference_ids"]].square()
        )
    rhs[plan["generator_rows"], plan["equation_rows"]] = 0.5 * generator_rhs
    if plan["wall_rows"].numel():
        rhs[plan["wall_rows"], plan["wall_columns"]] = (
            rhs[plan["wall_rows"], plan["wall_columns"]]
            - base_points[plan["wall_rows"], plan["wall_axes"]]
        )

    coordinates = base_points + _solve_vertex_systems(
        lhs, rhs, plan["system_groups"]
    )
    inverse_tensor = (
        plan["inverse"]
        if return_local_to_global
        else None
    )
    return coordinates, inverse_tensor


def facet_owners(result):
    """Return the owning cell ID for every packed facet.

    Cached on `result`: this is a pure function of the static
    `cell_facet_offsets` array, but is called on every
    `selected_facet_indices` call (in turn called by every
    `packed_global_facets`/`global_facets` call) -- without caching, a
    workflow that extracts facets repeatedly for the same result (e.g.
    once per CVT recompute step) redundantly rebuilds this O(n_facets)
    array from scratch every time.
    """
    cached = getattr(result, "_facet_owners_cache", None)
    if cached is not None:
        return cached
    cached = np.repeat(
        np.arange(result.n_cells, dtype=np.int32),
        np.diff(result.cell_facet_offsets),
    )
    cached.setflags(write=False)
    result._facet_owners_cache = cached
    return cached


def selected_facet_indices(result, facet_type):
    """Return (facet_indices, owners) for the requested facet subset.

    `facet_indices` is `None` for `facet_type == "all"` instead of the
    materialized `arange(n_facets)` -- callers that would otherwise gather
    through that identity permutation (e.g. `owners[facet_indices]`) can
    use the unindexed array directly, avoiding a full-array copy that
    does nothing (measured: ~0.09s wasted at N=1,000,000, 15.4 million
    facets, for exactly this no-op gather).
    """
    if facet_type not in ("surface", "inner", "all"):
        raise ValueError(
            "facet type must be 'surface', 'inner', or 'all', got {!r}".format(
                facet_type
            )
        )
    if result.n_inner == 0 and facet_type != "all":
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )
    owners = facet_owners(result)
    neighbors = result.facet_neighbors
    if facet_type == "all":
        return None, owners
    surface = (
        (owners < result.n_inner)
        & ((neighbors >= result.n_inner) | (neighbors < 0))
    )
    inner = (
        (owners < result.n_inner)
        & (neighbors >= 0)
        & (neighbors < result.n_inner)
    )
    mask = surface if facet_type == "surface" else inner
    return np.flatnonzero(mask), owners


def _build_facet_csr_numpy(result, facet_indices, selected_owners, local_to_global):
    """CPU builder: CSR-index assembly (repeat/cumsum/fancy-indexing) for
    packed_global_facets, in numpy."""
    counts = (
        result.facet_vertex_offsets[facet_indices + 1]
        - result.facet_vertex_offsets[facet_indices]
    )
    offsets = np.empty(facet_indices.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    facet_rows = np.repeat(
        np.arange(facet_indices.size, dtype=np.int64), counts
    )
    positions = (
        np.arange(offsets[-1], dtype=np.int64)
        - np.repeat(offsets[:-1], counts)
    )
    source_references = (
        np.repeat(result.facet_vertex_offsets[facet_indices], counts)
        + positions
    )
    reference_owners = selected_owners[facet_rows]
    local_occurrences = (
        result.cell_vertex_offsets[reference_owners]
        + result.facet_vertices[source_references]
    )
    global_vertices = local_to_global[local_occurrences]
    return offsets, global_vertices


def _static_result_tensors(result, device):
    """Return cached device tensors for the result arrays that never change
    across calls (facet_vertex_offsets, cell_vertex_offsets, facet_vertices).

    `_build_facet_csr_torch` runs once per `packed_global_facets` call, and
    that function is routinely called many times for the same result (once
    per mesh-extraction call, once per CVT recompute step, ...). Without
    this cache, every call re-transfers these arrays from host to device --
    tens to hundreds of MB at N=1,000,000 -- even though their contents are
    fixed for the lifetime of `result`.
    """
    cache = getattr(result, "_static_tensor_cache", None)
    if cache is None:
        cache = {}
        result._static_tensor_cache = cache
    device_key = (device.type, device.index)
    cached = cache.get(device_key)
    if cached is not None:
        return cached
    cached = {
        "facet_vertex_offsets": torch.tensor(
            result.facet_vertex_offsets, dtype=torch.int64, device=device
        ),
        "cell_vertex_offsets": torch.tensor(
            result.cell_vertex_offsets, dtype=torch.int64, device=device
        ),
        "facet_vertices": torch.tensor(
            result.facet_vertices, dtype=torch.int64, device=device
        ),
    }
    cache[device_key] = cached
    return cached


def _local_to_global_tensor(result, local_to_global, device):
    """Return a cached device tensor for `local_to_global`.

    `local_to_global` itself is already a stable, cached array returned
    by `global_vertex_map` for a given (result, device) pair -- every
    caller in this module obtains it directly or indirectly from that
    cache, so its identity is stable for the lifetime of `result`. Keying
    the tensor cache on that identity (rather than re-transferring it,
    often 100+ MB at N=1,000,000) avoids redundant host-to-device copies
    across repeated facet-extraction calls.
    """
    cache = getattr(result, "_local_to_global_tensor_cache", None)
    if cache is None:
        cache = {}
        result._local_to_global_tensor_cache = cache
    key = (device.type, device.index, id(local_to_global))
    cached = cache.get(key)
    if cached is not None:
        return cached
    cached = torch.tensor(local_to_global, dtype=torch.int64, device=device)
    cache[key] = cached
    return cached


def _build_facet_csr_torch(
    result, facet_indices, selected_owners, local_to_global, device
):
    """CUDA builder: same CSR-index assembly as `_build_facet_csr_numpy`,
    entirely in torch on the GPU (repeat_interleave/cumsum/gather instead
    of numpy repeat/cumsum/fancy-indexing) -- benchmarked ~18x faster at
    real problem sizes, same lesson as the other dedup/assembly ports in
    this module.
    """
    static = _static_result_tensors(result, device)
    fi = torch.as_tensor(facet_indices, dtype=torch.int64, device=device)
    fvo = static["facet_vertex_offsets"]
    counts = fvo[fi + 1] - fvo[fi]
    total = int(counts.sum().item())
    offsets = torch.zeros(fi.numel() + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    facet_rows = torch.repeat_interleave(
        torch.arange(fi.numel(), dtype=torch.int64, device=device), counts
    )
    positions = torch.arange(
        total, dtype=torch.int64, device=device
    ) - torch.repeat_interleave(offsets[:-1], counts)
    source_references = (
        torch.repeat_interleave(fvo[fi], counts) + positions
    )
    so = torch.tensor(selected_owners, dtype=torch.int64, device=device)
    reference_owners = so[facet_rows]
    cvo = static["cell_vertex_offsets"]
    fverts = static["facet_vertices"]
    local_occurrences = cvo[reference_owners] + fverts[source_references]
    ltg = _local_to_global_tensor(result, local_to_global, device)
    global_vertices = ltg[local_occurrences]
    return offsets.cpu().numpy(), global_vertices.cpu().numpy()


def packed_global_facets(
    result,
    facet_type,
    deduplicate=True,
    local_to_global=None,
    facet_selection=None,
    device="cpu",
):
    """Return packed global-vertex facets and owner/neighbor pairs.

    `device` must match whatever device any paired `compute_unique_vertices`
    call for the same `result` uses: global_vertex_map's unique-vertex
    ordering differs between the CPU and CUDA dedup paths (same set,
    different order), so mixing devices between this function and vertex
    reconstruction for the same result silently misaligns vertex indices.
    """
    if facet_selection is None:
        facet_selection = selected_facet_indices(result, facet_type)
    facet_indices, owners = facet_selection
    if facet_indices is None:
        # "all" selection: `owners`/`facet_neighbors` are already in the
        # right order, so use them directly instead of gathering through
        # an identity-permutation index array (see selected_facet_indices).
        if owners.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.empty((0, 2), dtype=np.int32),
            )
        selected_owners = owners
        selected_neighbors = result.facet_neighbors
        facet_indices = np.arange(result.facet_neighbors.size, dtype=np.int64)
    else:
        if facet_indices.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.empty((0, 2), dtype=np.int32),
            )
        selected_owners = owners[facet_indices]
        selected_neighbors = result.facet_neighbors[facet_indices]
    if local_to_global is None:
        _, local_to_global = global_vertex_map(result, device)

    if deduplicate:
        internal = selected_neighbors >= 0
        internal_positions = np.flatnonzero(internal)
        if internal_positions.size:
            pairs = np.column_stack(
                [
                    np.minimum(
                        selected_owners[internal],
                        selected_neighbors[internal],
                    ),
                    np.maximum(
                        selected_owners[internal],
                        selected_neighbors[internal],
                    ),
                ]
            )
            _, first = _unique_rows_first_occurrence(pairs, device)
            keep = ~internal
            keep[internal_positions[first]] = True
            facet_indices = facet_indices[keep]
            selected_owners = selected_owners[keep]
            selected_neighbors = selected_neighbors[keep]

    device_obj = torch.device(device)
    if device_obj.index is None and device_obj.type == "cuda":
        device_obj = torch.device(
            device_obj.type, torch.cuda.current_device()
        )
    if device_obj.type == "cuda":
        offsets_np, global_vertices_np = _build_facet_csr_torch(
            result, facet_indices, selected_owners, local_to_global, device_obj
        )
    else:
        offsets_np, global_vertices_np = _build_facet_csr_numpy(
            result, facet_indices, selected_owners, local_to_global
        )
    offsets = offsets_np
    global_vertices = global_vertices_np
    info = np.column_stack(
        [selected_owners, selected_neighbors]
    ).astype(np.int32, copy=False)
    return global_vertices, offsets, info


def global_facets(
    result,
    facet_type,
    deduplicate=True,
    local_to_global=None,
    facet_selection=None,
    device="cpu",
):
    """Return selected facets as global vertex IDs plus owner/neighbor pairs.

    See packed_global_facets for the `device` consistency requirement.
    """
    vertices, offsets, info = packed_global_facets(
        result,
        facet_type,
        deduplicate=deduplicate,
        local_to_global=local_to_global,
        facet_selection=facet_selection,
        device=device,
    )
    facets = [
        vertices[offsets[row]:offsets[row + 1]].tolist()
        for row in range(info.shape[0])
    ]
    return facets, info


def unique_geometry_coordinates(result, inverse, device):
    """Choose the first cell-local coordinate for every global vertex ID."""
    if inverse.size == 0:
        return torch.empty(
            (0, result.dimension), dtype=torch.float64, device=device
        )
    _, first = np.unique(inverse, return_index=True)
    return torch.as_tensor(
        result.vertices[first], dtype=torch.float64, device=device
    )
