"""Per-cell differentiable Voronoi vertices.

Each Voronoi cell keeps its own copy of each of its vertices. The coordinates
are taken directly from Voro++, so no vertex reconstruction is needed in the
forward pass, and no global deduplication of vertices is needed at all.

Gradients with respect to the generator positions are computed analytically in
the backward pass. A vertex v defined by generators p_1, ..., p_k (and possibly
by domain walls) satisfies the linear system A v = b, where the row for
generator p_j (j > 1) is

    2 (p_j - p_1) . v = |p_j|^2 - |p_1|^2,

and the row for a wall is e_axis . v = (wall position). Differentiating gives
A dv = r, with r_j = 2 (p_j - v) . dp_j - 2 (p_1 - v) . dp_1 for a generator row,
and r = 0 for a wall row. So for an incoming gradient g = dE/dv, the
contribution to dE/dp is found from y = A^{-T} g:

    dE/dp_j += 2 y_j (p_j - v),        dE/dp_1 -= 2 (sum_j y_j) (p_1 - v).

If a vertex has more than d + 1 defining generators and walls (a degenerate
vertex), the system is overdetermined but consistent, and y is the
least-squares (pseudo-inverse) solution y = A (A^T A)^{-1} g. In the usual,
square case, y = A^{-T} g is computed with explicit 2x2 or 3x3 formulas.

In a radical (power) tessellation, with generator radii r, the row for
generator p_j is

    2 (p_j - p_1) . v = |p_j|^2 - |p_1|^2 - (r_j^2 - r_1^2).

The matrix is the same, so the gradient with respect to the positions is
unchanged, and the gradient with respect to the radii is

    dE/dr_j -= 2 y_j r_j,        dE/dr_1 += 2 (sum_j y_j) r_1.

In a periodic domain, each generator that defines a vertex is taken to be its
periodic image nearest to the vertex. This is correct because all of the
defining generators are at the same power distance |v - p|^2 - r^2 from the
vertex, and no generator is closer, and the periodic images of one generator
all have the same radius; it requires the box to be large compared with the
cells, which is checked.

The backward pass is a single parallel loop over the vertex copies in C++
(per_cell_backward in vpp.cpp). For tensors on a GPU, the vertex gradient is
copied to the CPU for this step and the result copied back. Setting
VOROPLEX_PER_CELL_BACKWARD=torch instead uses batched tensor operations on the
GPU.
"""
import os

import numpy as np
import torch

from voroplex._lib import voroplusplus_

# Backward pass for tensors on a GPU: "cpu" (default) runs the C++ kernel on
# the CPU; "torch" uses batched tensor operations on the GPU instead.
_BACKWARD = os.environ.get("VOROPLEX_PER_CELL_BACKWARD", "cpu")

from ._packed import require_result


def _require_per_cell(result):
    require_result(result)
    if result.representation != "per_cell":
        raise ValueError(
            "this operation requires a result computed with "
            "representation='per_cell'"
        )


def _signature_plan(result):
    """Build (and cache) the integer description of each vertex's system.

    Works directly on the flat signature arrays. Each local vertex (one copy
    of a vertex in one cell) has a sorted signature: walls (negative IDs)
    first, then generators. The first generator is the base p_1, and every
    other entry gives one equation row: a generator row or a wall row.

    Square vertices (exactly d rows, the usual case) are described by dense
    (n_square, d) arrays; the rare overdetermined vertices by padded arrays.
    """
    cached = getattr(result, "_per_cell_plan_cache", None)
    if cached is not None:
        return cached

    d = result.dimension
    offs = np.asarray(result.vertex_signature_offsets, dtype=np.int64)
    ids = np.asarray(result.vertex_signature_ids, dtype=np.int64)
    n_vertices = offs.size - 1
    lens = np.diff(offs)
    v_of_e = np.repeat(np.arange(n_vertices, dtype=np.int64), lens)
    pos = np.arange(ids.size, dtype=np.int64) - offs[v_of_e]
    is_wall = ids < 0
    wall_count = np.bincount(v_of_e, weights=is_wall, minlength=n_vertices).astype(np.int64)
    gen_count = lens - wall_count
    if np.any(gen_count == 0):
        raise ValueError("a vertex signature has no generator IDs")
    n_eq = lens - 1
    if np.any(n_eq < d):
        _check_backward(0, int(np.sum(n_eq < d)))

    base_pos = offs[:-1] + wall_count
    base_id = ids[base_pos]
    # Row index of each non-base entry: walls keep their position; generators
    # after the base move down by one.
    is_row = pos != wall_count[v_of_e]
    row = pos - (pos > wall_count[v_of_e])
    e_row = np.flatnonzero(is_row)
    rv, rr, rid, rwall = v_of_e[e_row], row[e_row], ids[e_row], is_wall[e_row]
    gen_of_row = np.where(rwall, -1, rid)
    axis_of_row = np.where(rwall, (-rid - 1) // 2, -1)

    plan = dict(base_id=base_id)
    square = np.flatnonzero(n_eq == d)
    sq_index = np.full(n_vertices, -1, dtype=np.int64)
    sq_index[square] = np.arange(square.size)
    sel = sq_index[rv] >= 0
    sq_gen = np.full((square.size, d), -1, dtype=np.int64)
    sq_axis = np.full((square.size, d), -1, dtype=np.int64)
    sq_gen[sq_index[rv[sel]], rr[sel]] = gen_of_row[sel]
    sq_axis[sq_index[rv[sel]], rr[sel]] = axis_of_row[sel]
    plan.update(square=square, sq_gen=sq_gen, sq_axis=sq_axis)

    over = np.flatnonzero(n_eq > d)
    width = int(n_eq[over].max()) if over.size else 1
    ov_index = np.full(n_vertices, -1, dtype=np.int64)
    ov_index[over] = np.arange(over.size)
    sel = ov_index[rv] >= 0
    ov_gen = np.full((over.size, width), -1, dtype=np.int64)
    ov_axis = np.full((over.size, width), -1, dtype=np.int64)
    ov_gen[ov_index[rv[sel]], rr[sel]] = gen_of_row[sel]
    ov_axis[ov_index[rv[sel]], rr[sel]] = axis_of_row[sel]
    plan.update(over=over, ov_gen=ov_gen, ov_axis=ov_axis)

    result._per_cell_plan_cache = plan
    return plan


def _plan_tensors(result, device):
    """The plan as tensors on the given device, cached per result."""
    cache = getattr(result, "_per_cell_plan_tensors_cache", None)
    if cache is None:
        cache = {}
        result._per_cell_plan_tensors_cache = cache
    key = (torch.device(device).type, torch.device(device).index)
    if key not in cache:
        cache[key] = {k: torch.as_tensor(v, device=device)
                      for k, v in _signature_plan(result).items()}
    return cache[key]


def _box_lengths(result):
    """Box length along each periodic axis, and 0 along the others."""
    lengths = result.limits[:, 1] - result.limits[:, 0]
    return np.where(result.periodic, lengths, 0.0)


def _nearest_image(p, v, box):
    """The periodic image of each point p that is nearest to v."""
    if box is None:
        return p
    periodic = box > 0
    safe = torch.where(periodic, box, torch.ones_like(box))
    shift = torch.round((p - v) / safe) * safe
    return p - torch.where(periodic, shift, torch.zeros_like(shift))


def _check_backward(n_singular, n_mismatched):
    if n_mismatched:
        raise RuntimeError(
            f"{n_mismatched} vertices are not at the same distance from all of "
            "the generators that define them. Either the points (or radii) "
            "passed to cell_vertices differ from those used to compute the "
            "tessellation (for example, because they were converted to a "
            "different precision), or, in a periodic domain, the box is too "
            "small compared with the cells."
        )
    if n_singular:
        raise RuntimeError(
            f"{n_singular} vertex systems were singular in the backward pass"
        )


def _check_equidistant(pg, is_gen, p_base, v, s_gen=0.0, s_base=0.0):
    """Check that the generators are at the same power distance from v.

    s_gen and s_base are the squared radii (zero without radii).
    """
    d_base = ((p_base - v) ** 2).sum(dim=1, keepdim=True)
    d_gen = ((pg - v.unsqueeze(1)) ** 2).sum(dim=2)
    diff = (d_gen - s_gen) - (d_base - s_base)
    scale = torch.maximum(d_gen + s_gen, d_base + s_base)
    bad = is_gen & (diff.abs() > 1e-6 * scale)
    _check_backward(0, int(bad.any(dim=1).sum()))


def _rows(points, p_base, gen, axis, v=None, box=None):
    """Equation rows for a batch of vertices: 2(p_j - p_1) or a unit vector.

    In a periodic domain, each generator p_j is replaced by its periodic image
    nearest to the vertex v (p_base should already be the nearest image).
    """
    d = points.shape[1]
    is_gen = gen >= 0
    pg = points[gen.clamp(min=0)]                                   # (n, r, d)
    if box is not None:
        pg = _nearest_image(pg, v.unsqueeze(1), box)
    unit = torch.nn.functional.one_hot(axis.clamp(min=0), d).to(points.dtype)
    unit = unit * (axis >= 0).unsqueeze(-1)
    A = torch.where(is_gen.unsqueeze(-1), 2.0 * (pg - p_base.unsqueeze(1)), unit)
    return A, pg, is_gen


def _scatter_add(target, index, values):
    """target[index] += values (rows), using bincount on the CPU."""
    if target.ndim == 1:
        target.index_add_(0, index, values)
    elif target.device.type == "cpu":
        n = target.shape[0]
        for k in range(target.shape[1]):
            target[:, k] += torch.bincount(index, weights=values[:, k], minlength=n)
    else:
        target.index_add_(0, index, values)


def _solve_transpose_square(A, g):
    """y = A^{-T} g for a batch of 2x2 or 3x3 matrices, by explicit formulas."""
    if A.shape[1] == 2:
        a, b, c, e = A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1]
        det = a * e - b * c
        y0 = (e * g[:, 0] - c * g[:, 1]) / det
        y1 = (-b * g[:, 0] + a * g[:, 1]) / det
        return torch.stack([y0, y1], dim=1)
    r0, r1, r2 = A[:, 0], A[:, 1], A[:, 2]
    c0 = torch.cross(r1, r2, dim=1)
    det = (r0 * c0).sum(dim=1)
    y = torch.stack([
        (c0 * g).sum(dim=1),
        (torch.cross(r2, r0, dim=1) * g).sum(dim=1),
        (torch.cross(r0, r1, dim=1) * g).sum(dim=1),
    ], dim=1)
    return y / det.unsqueeze(1)


class _CellVertices(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, radii, result):
        # The result's arrays are read-only, so copy them into a new tensor.
        vertices = torch.tensor(
            result.vertices, dtype=points.dtype, device=points.device
        )
        ctx.result = result
        ctx.has_radii = radii is not None
        if radii is None and result.is_radical:
            # The radii are constants, taken from the result.
            radii = torch.tensor(result.radii, dtype=points.dtype,
                                    device=points.device)
        ctx.save_for_backward(points.detach(), vertices,
                              None if radii is None else radii.detach())
        return vertices

    @staticmethod
    def backward(ctx, grad_vertices):
        # The backward pass is computed outside PyTorch, so it cannot itself be
        # differentiated. With create_graph=True, PyTorch would silently leave
        # out its contribution to second derivatives, so this is an error.
        # (once_differentiable does not catch this, because its error node has
        # no path back to the points.)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "cell_vertices does not support second derivatives; compute "
                "the gradient without create_graph=True"
            )
        points, vertices, radii = ctx.saved_tensors
        result = ctx.result
        want_radii = ctx.has_radii and ctx.needs_input_grad[1]
        if points.device.type == "cpu" or _BACKWARD != "torch":
            # One parallel pass over the vertex copies in C++. For tensors on
            # a GPU, the vertex gradient is copied to the CPU and the result
            # copied back: this is much faster than the batched tensor version
            # below, and the vertex coordinates are already on the CPU.
            grad, grad_r, n_singular, n_mismatched = voroplusplus_.per_cell_backward_cpu(
                points.detach().cpu().numpy(), result.vertices,
                grad_vertices.detach().cpu().contiguous().numpy(),
                result.vertex_signature_offsets, result.vertex_signature_ids,
                _box_lengths(result), result.threads,
                radii=None if radii is None else radii.cpu().numpy(),
                radii_grad=want_radii,
            )
            _check_backward(n_singular, n_mismatched)
            if grad_r is not None:
                grad_r = torch.from_numpy(grad_r).to(points.device, points.dtype)
            return (torch.from_numpy(grad).to(points.device, points.dtype),
                    grad_r, None)

        # The same calculation with batched tensor operations on the device.
        pt = _plan_tensors(result, points.device)
        base_id = pt["base_id"]
        grad_points = torch.zeros_like(points)
        grad_radii = torch.zeros_like(radii) if want_radii else None
        box_np = _box_lengths(result)
        box = (torch.as_tensor(box_np, dtype=points.dtype, device=points.device)
               if box_np.any() else None)

        for kind in ("sq", "ov"):
            idx = pt["square"] if kind == "sq" else pt["over"]
            if idx.numel() == 0:
                continue
            gen, axis = pt[kind + "_gen"], pt[kind + "_axis"]
            v, g = vertices[idx], grad_vertices[idx]
            p_base = points[base_id[idx]]
            p_base = _nearest_image(p_base, v, box)
            A, pg, is_gen = _rows(points, p_base, gen, axis, v, box)
            if radii is not None:
                r_gen = radii[gen.clamp(min=0)] * is_gen
                r_base = radii[base_id[idx]]
                _check_equidistant(pg, is_gen, p_base, v, r_gen ** 2,
                                   r_base.unsqueeze(1) ** 2)
            elif box is not None:
                _check_equidistant(pg, is_gen, p_base, v)
            if kind == "sq":
                # y = A^{-T} g, by explicit 2x2 or 3x3 formulas.
                y = _solve_transpose_square(A, g)
            else:
                # Least-squares (pseudo-inverse) solution y = A (A^T A)^{-1} g.
                M = torch.einsum("vrd,vre->vde", A, A)
                y = torch.einsum("vrd,vd->vr", A, torch.linalg.solve(M, g))
            y = y * is_gen
            # Generator rows: dE/dp_j += 2 y_j (p_j - v).
            contrib = 2.0 * y.unsqueeze(-1) * (pg - v.unsqueeze(1))
            m = is_gen.reshape(-1)
            _scatter_add(grad_points, gen.reshape(-1)[m],
                         contrib.reshape(-1, points.shape[1])[m])
            # Base generator: dE/dp_1 -= 2 (sum_j y_j) (p_1 - v).
            _scatter_add(grad_points, base_id[idx],
                         -2.0 * y.sum(dim=1, keepdim=True) * (p_base - v))
            if grad_radii is not None:
                # dE/dr_j -= 2 y_j r_j, and dE/dr_1 += 2 (sum_j y_j) r_1.
                _scatter_add(grad_radii, gen.reshape(-1)[m],
                             (-2.0 * y * r_gen).reshape(-1)[m])
                _scatter_add(grad_radii, base_id[idx],
                             2.0 * y.sum(dim=1) * r_base)
        return grad_points, grad_radii, None


def cell_vertices(result, points, radii=None):
    """Return each cell's own copy of its vertices, as a differentiable tensor.

    Parameters
    ----------
    result : VoronoiResult
        Computed with ``representation="per_cell"`` from the same generator
        positions as ``points``.
    points : torch.Tensor, shape (N, d)
        Generator positions. Gradients of any quantity computed from the
        returned vertices flow back to this tensor.
    radii : torch.Tensor, shape (N,), optional
        For a radical tessellation, the generator radii, with the same values
        as those used to compute it. Gradients then flow back to this tensor
        too. If omitted, the radii stored in the result are used, as
        constants. Not allowed for a standard Voronoi tessellation.

    Returns
    -------
    torch.Tensor, shape (n_local_vertices, d)
        The vertices of all cells, cell by cell. The vertices of cell ``i`` are
        rows ``result.cell_vertex_offsets[i]`` to
        ``result.cell_vertex_offsets[i + 1] - 1``. A vertex shared by several
        cells appears once for each of them.
    """
    _require_per_cell(result)
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim != 2 or points.shape != (result.n_points, result.dimension):
        raise ValueError(
            "points must have shape ({}, {})".format(result.n_points, result.dimension)
        )
    if radii is not None:
        if not result.is_radical:
            raise ValueError(
                "radii were given, but the tessellation was computed without "
                "radii"
            )
        if not isinstance(radii, torch.Tensor):
            raise TypeError("radii must be a torch.Tensor")
        if radii.shape != (result.n_points,):
            raise ValueError(
                "radii must have shape ({},)".format(result.n_points)
            )
        if radii.dtype != points.dtype or radii.device != points.device:
            raise ValueError(
                "radii must have the same dtype and device as points"
            )
    return _CellVertices.apply(points, radii, result)


def cell_fan_simplices(result, device="cpu"):
    """Split every cell into simplices that share the cell's generator.

    In 2D, each edge (v0, v1) of a cell gives one triangle with corners
    (generator, v0, v1). In 3D, each face of a cell is split into triangles
    from its first vertex, and each triangle gives one tetrahedron. Integrals
    over the cells can then be computed as sums over these simplices. The
    simplices are built in parallel in C++.

    Every simplex is positively oriented: with q_k the corners relative to
    the generator, its signed area ``(q1 x q2) / 2`` or signed volume
    ``q1 . (q2 x q3) / 6`` is positive when the generator lies inside the
    cell. Use these signed measures, not their absolute values. Then the sum
    over a cell's simplices of any integral is the integral over the cell even
    when the generator lies outside its own cell, which can happen in a
    radical tessellation: the simplices that lie outside the cell have
    negative measure, and cancel. A cell that is empty has no simplices.

    Parameters
    ----------
    result : VoronoiResult
        Computed with ``representation="per_cell"``.
    device : str or torch.device
        Device for the returned tensors.

    Returns
    -------
    owners : torch.LongTensor, shape (n_simplices,)
        The cell (generator ID) that each simplex belongs to. The generator,
        ``points[owners]``, is the shared corner of the simplex.
    simplex_vertices : torch.LongTensor, shape (n_simplices, d)
        The other corners of each simplex, as row indices into the tensor
        returned by :func:`cell_vertices`.
    """
    _require_per_cell(result)
    owners, simplices = voroplusplus_.per_cell_fan_simplices_cpu(
        result.dimension, result.cell_vertex_offsets, result.cell_facet_offsets,
        result.facet_vertex_offsets, result.facet_vertices, result.threads,
    )
    return (
        torch.from_numpy(owners).to(device),
        torch.from_numpy(simplices).to(device),
    )


def _cell_fan_simplices_numpy(result, device="cpu"):
    """Reference NumPy version of :func:`cell_fan_simplices`.

    Kept for testing; it gives exactly the same arrays, in the same order.

    In 2D, each edge of a cell gives one triangle (generator, v0, v1). In 3D,
    each face of a cell is split into triangles from its first vertex, and
    each triangle gives one tetrahedron (generator, v0, v_{k+2}, v_{k+1}).

    Returns
    -------
    owners : torch.LongTensor, shape (n_simplices,)
        The cell (generator ID) that each simplex belongs to.
    simplex_vertices : torch.LongTensor, shape (n_simplices, d)
        Row indices into the tensor returned by :func:`cell_vertices`.
    """
    _require_per_cell(result)
    cell_vertex_offsets = result.cell_vertex_offsets
    cell_facet_offsets = result.cell_facet_offsets
    facet_vertex_offsets = result.facet_vertex_offsets
    facet_vertices = result.facet_vertices.astype(np.int64)

    n_cells = cell_facet_offsets.size - 1
    facets_per_cell = np.diff(cell_facet_offsets)
    facet_owner = np.repeat(np.arange(n_cells, dtype=np.int64), facets_per_cell)
    # Convert cell-local vertex indices to rows of the cell_vertices tensor.
    refs_per_facet = np.diff(facet_vertex_offsets)
    ref_owner = np.repeat(facet_owner, refs_per_facet)
    global_refs = facet_vertices + cell_vertex_offsets[ref_owner]

    if result.dimension == 2:
        if np.any(refs_per_facet != 2):
            raise ValueError("2D facets must have exactly two vertices")
        starts = facet_vertex_offsets[:-1]
        simplex_vertices = np.stack(
            [global_refs[starts], global_refs[starts + 1]], axis=1
        )
        owners = facet_owner
    else:
        n_tri = np.maximum(refs_per_facet - 2, 0)
        face_of_tri = np.repeat(np.arange(n_tri.size), n_tri)
        tri_start = np.cumsum(n_tri) - n_tri
        k = np.arange(n_tri.sum()) - np.repeat(tri_start, n_tri)
        base = facet_vertex_offsets[face_of_tri]
        simplex_vertices = np.stack(
            [global_refs[base], global_refs[base + 2 + k], global_refs[base + 1 + k]],
            axis=1,
        )
        owners = facet_owner[face_of_tri]
    return (
        torch.as_tensor(owners, dtype=torch.long, device=device),
        torch.as_tensor(simplex_vertices, dtype=torch.long, device=device),
    )
