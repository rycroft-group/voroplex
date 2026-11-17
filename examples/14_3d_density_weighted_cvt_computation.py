"""
3D Density-Field CVT Optimization with VoroPlex + PyTorch
========================================================

Optimizes 3D generators for a density field rho(x), which is the weight in the
centroidal Voronoi tessellation (CVT) energy

    E = sum_i integral_{cell_i} rho(x) ||x - p_i||^2 dx.

At each iteration, VoroPlex computes the tessellation with the per-cell
representation. voroplex.cell_vertices gives the vertices of each cell as a
PyTorch tensor whose gradient flows back to the generators. Each cell is
split into tetrahedra that have its generator as one corner, the center of one
of its faces as another, and one edge of that face as the other two, and the
energy is integrated over each tetrahedron by quadrature. The whole calculation
is vectorized, so it also runs on a GPU.

Use --density=name to give rho directly, or --sizing=name to give a target
relative spacing h, which is converted to rho = h^(-5): for a 3D CVT, the cell
length scale is proportional to rho^(-1/5), so cells of size h need
rho = h^(-(d+2)). Add a field to the dictionaries below to use your own.

Run from the project root:
    python examples/14_3d_density_weighted_cvt_computation.py

Optional arguments:
    python examples/14_3d_density_weighted_cvt_computation.py \
        [num_threads] [num_points] [num_iters] [lr]
    python examples/14_3d_density_weighted_cvt_computation.py --density=gaussian
    python examples/14_3d_density_weighted_cvt_computation.py --sizing=diagonal
    python examples/14_3d_density_weighted_cvt_computation.py \
        --sizing=edge_cluster --edge-h-min=0.5 --edge-h-max=1.5
    python examples/14_3d_density_weighted_cvt_computation.py --lr=0.002 --lr-min=0.0002
    python examples/14_3d_density_weighted_cvt_computation.py --converge-tol=1e-4 --converge-window=30
    python examples/14_3d_density_weighted_cvt_computation.py --quad-points=4 --quad-refine=1
    python examples/14_3d_density_weighted_cvt_computation.py --device=auto   # or cpu, cuda, mps
    python examples/14_3d_density_weighted_cvt_computation.py --clip-axis=z --clip-value=0.5
    python examples/14_3d_density_weighted_cvt_computation.py --clip-axis=diagonal --clip-value=1.5
    python examples/14_3d_density_weighted_cvt_computation.py --no-vtu
    python examples/14_3d_density_weighted_cvt_computation.py --init-from=results/run_params.json

The three 3D examples in Section 5.1 of the VoroPlex paper (N = 1500, at most
1600 iterations, stopping when the energy has converged):
    python examples/14_3d_density_weighted_cvt_computation.py \
        --density=uniform --num-points=1500 --num-iters=1600 \
        --lr=0.002 --lr-min=0.0002
    python examples/14_3d_density_weighted_cvt_computation.py \
        --sizing=diagonal --num-points=1500 --num-iters=1600 \
        --lr=0.0008 --lr-min=0.00005
    python examples/14_3d_density_weighted_cvt_computation.py \
        --sizing=edge_cluster --edge-h-min=0.6 --edge-h-max=1.2 \
        --num-points=1500 --num-iters=1600 --lr=0.002 --lr-min=0.0002

Quadrature has two independent settings:
    --quad-points={1,4}  the rule on each tetrahedron: a 4-point rule that is
                         exact for quadratics, and so exact for a uniform
                         density (the default), or the 1-point midpoint rule.
                         The integrand |x - p|^2 is quadratic, so the midpoint
                         rule is inaccurate on these tetrahedra, which have the
                         generator as a corner: for a uniform density, its
                         gradient is wrong by tens of percent.
    --quad-refine=N      first split each tetrahedron into 8^N congruent pieces,
                         and apply the rule to each piece. This improves the
                         accuracy for densities that vary within a cell. Each
                         step multiplies the cost by 8.

Optional regularization terms, all off by default:
    --vol-reg=w          penalizes differences between the masses of the cells
    --face-area-reg=w    penalizes differences between the areas of the faces
                         on the boundary, scaled by the local spacing h^2
    --face-circ-reg=w    makes the faces on the boundary rounder

Available fields:
    density or sizing: uniform, gaussian, two_blobs, z_heavy, diagonal,
                       linear_xyz_123, linear_xyz_123_gradual, shell
    sizing only:       edge_cluster, edge_cluster_sharp, edge_cluster_analytic
                       (with --density, these are converted to rho = h^(-5))

    edge_cluster is a linear ramp from --edge-h-min on the edges of the cube to
    --edge-h-max at the largest distance from them, sqrt(2)/2. The exact
    distance to the nearest edge is precomputed on a 65^3 grid and sampled with
    differentiable trilinear interpolation. edge_cluster_sharp is an earlier,
    rapidly saturating exponential profile, and edge_cluster_analytic is a
    smooth closed form of it.
"""

import functools
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import voroplex


DEFAULT_FIELD = "uniform"
SEED = 42

POSITIONAL_ARGS = [arg for arg in sys.argv[1:] if not arg.startswith("--")]


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------

def parse_int_arg(index, default):
    try:
        return int(POSITIONAL_ARGS[index])
    except (IndexError, ValueError):
        return default


def parse_float_arg(index, default):
    try:
        return float(POSITIONAL_ARGS[index])
    except (IndexError, ValueError):
        return default


def parse_option(name, default, kind=str):
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            try:
                return kind(arg[len(prefix):])
            except ValueError:
                raise SystemExit(f"Invalid value for --{name}: {arg[len(prefix):]}")
    return default


def parse_field():
    """Return (field name, True if it is a sizing field)."""
    density = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--density=")]
    sizing = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--sizing=")]
    if density and sizing:
        raise SystemExit("Choose either --density or --sizing, not both")
    if sizing:
        return sizing[-1], True
    return (density or [DEFAULT_FIELD])[-1], False


def resolve_device(name):
    """Resolve auto as CUDA > MPS > CPU; reject unavailable explicit backends."""
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if name == "auto":
        name = next(device for device in ("cuda", "mps", "cpu") if available[device])
    if name not in available:
        raise SystemExit("--device must be one of: auto, cpu, cuda, mps")
    if not available[name]:
        raise SystemExit(f"--device={name} requested but this backend is unavailable")
    return torch.device(name)


def build_clip_spec(limits):
    """The optional selection of cells for the ParaView output."""
    axis = parse_option("clip-axis", None)
    if axis is None:
        return None
    axis = axis.lower()
    axis_to_index = {"x": 0, "y": 1, "z": 2, "diagonal": None}
    if axis not in axis_to_index:
        raise SystemExit("--clip-axis must be one of: x, y, z, diagonal")
    side = parse_option("clip-side", "lower").lower()
    if side not in ("lower", "upper"):
        raise SystemExit("--clip-side must be either 'lower' or 'upper'")
    index = axis_to_index[axis]
    if axis == "diagonal":
        lo, hi = sum(b[0] for b in limits), sum(b[1] for b in limits)
    else:
        lo, hi = limits[index]
    value = parse_option("clip-value", 0.5 * (lo + hi), float)
    return {"axis": axis, "axis_index": index, "value": value, "side": side,
            "range": [lo, hi]}


def clip_mask_for_points(points, clip_spec):
    if clip_spec is None:
        return np.ones(len(points), dtype=bool)
    points = np.asarray(points)
    coord = (points.sum(axis=1) if clip_spec["axis"] == "diagonal"
             else points[:, clip_spec["axis_index"]])
    if clip_spec["side"] == "lower":
        return coord <= clip_spec["value"]
    return coord >= clip_spec["value"]


def describe_clip_spec(clip_spec):
    if clip_spec is None:
        return "none"
    op = "<=" if clip_spec["side"] == "lower" else ">="
    coordinate = "x+y+z" if clip_spec["axis"] == "diagonal" else clip_spec["axis"]
    return f'{coordinate} {op} {clip_spec["value"]} (selected by cell center)'


# ---------------------------------------------------------------------------
# Fields. Each takes points of shape (..., 3) and returns values of shape (...).
# ---------------------------------------------------------------------------

def _gaussian(xyz, center, sigma):
    c = torch.tensor(center, dtype=xyz.dtype, device=xyz.device)
    return torch.exp(-((xyz - c) ** 2).sum(dim=-1) / (2.0 * sigma ** 2))


def field_uniform(xyz):
    return torch.ones_like(xyz[..., 0])


def field_gaussian(xyz):
    return 0.12 + 4.0 * _gaussian(xyz, (0.5, 0.5, 0.5), 0.17)


def field_two_blobs(xyz):
    return (0.10 + 3.0 * _gaussian(xyz, (0.30, 0.72, 0.62), 0.14)
            + 2.2 * _gaussian(xyz, (0.76, 0.34, 0.38), 0.18))


def field_z_heavy(xyz):
    return 0.15 + 2.8 * xyz[..., 2] ** 2


def field_diagonal(xyz):
    """1 + x + y + z: with --sizing, the linear field of the VoroPlex paper."""
    return 1.0 + xyz[..., 0] + xyz[..., 1] + xyz[..., 2]


def field_linear_xyz_123(xyz):
    return 1.0 + xyz[..., 0] + 2.0 * xyz[..., 1] + 3.0 * xyz[..., 2]


def field_shell(xyz):
    radius = torch.linalg.norm(xyz - 0.5, dim=-1)
    return 0.12 + 3.2 * torch.exp(-((radius - 0.30) ** 2) / (2.0 * 0.06 ** 2))


LINEAR_GRADUAL_H_MIN = parse_option("linear-h-min", 0.6, float)
LINEAR_GRADUAL_H_MAX = parse_option("linear-h-max", 1.8, float)


def sizing_linear_xyz_123_gradual(xyz):
    """A bounded ramp along x + 2y + 3z, from --linear-h-min to --linear-h-max."""
    t = (xyz[..., 0] + 2.0 * xyz[..., 1] + 3.0 * xyz[..., 2]) / 6.0
    return LINEAR_GRADUAL_H_MIN + (LINEAR_GRADUAL_H_MAX - LINEAR_GRADUAL_H_MIN) * t


def density_linear_xyz_123_gradual(xyz):
    """A direct density ramp with a 3:1 ratio along x + 2y + 3z."""
    return 0.5 + (xyz[..., 0] + 2.0 * xyz[..., 1] + 3.0 * xyz[..., 2]) / 6.0


def _soft_min(*values, tau=0.02):
    """A smooth approximation of min(values), -tau * logsumexp(-values / tau)."""
    return -tau * torch.logsumexp(-torch.stack(values, dim=0) / tau, dim=0)


def sizing_edge_cluster_analytic(xyz):
    """A smooth sizing field that is small near the 12 edges of the cube.

    The hard minima in the distance to the nearest edge are replaced by soft
    minima, and a small constant inside each square root keeps the gradient
    finite on the edges themselves.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dx = _soft_min(x, 1.0 - x)
    dy = _soft_min(y, 1.0 - y)
    dz = _soft_min(z, 1.0 - z)
    d = _soft_min(torch.sqrt(dy ** 2 + dz ** 2 + 1e-12),
                  torch.sqrt(dx ** 2 + dz ** 2 + 1e-12),
                  torch.sqrt(dx ** 2 + dy ** 2 + 1e-12))
    return 0.10 + 3.0 * (1.0 - torch.exp(-d ** 2 / (2.0 * 0.10 ** 2)))


# An odd resolution includes the center of the cube, where the distance to the
# edges is largest.
EDGE_GRID_RESOLUTION = 65
EDGE_H_MIN = parse_option("edge-h-min", 0.5, float)
EDGE_H_MAX = parse_option("edge-h-max", 1.5, float)
EDGE_MAX_DISTANCE = 0.5 * 2.0 ** 0.5


@functools.lru_cache(maxsize=4)
def _edge_distance_grid(resolution):
    """The exact distance to the nearest edge of the cube on a grid;
    grid[ix, iy, iz] is the value at (ix, iy, iz) / (resolution - 1)."""
    g = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    xx, yy, zz = torch.meshgrid(g, g, g, indexing="ij")
    d = []
    for a, b in ((yy, zz), (xx, zz), (xx, yy)):       # edges along x, y, and z
        for a0 in (0.0, 1.0):
            for b0 in (0.0, 1.0):
                d.append(torch.sqrt((a - a0) ** 2 + (b - b0) ** 2))
    return torch.stack(d, dim=0).min(dim=0).values


def _trilinear_sample(grid, x, y, z):
    """Differentiable trilinear interpolation in a grid covering [0, 1]^3."""
    r = grid.shape[0]
    g = [c.clamp(0.0, 1.0) * (r - 1) for c in (x, y, z)]
    i0 = [c.floor().long().clamp(0, r - 2) for c in g]
    t = [c - i.to(c.dtype) for c, i in zip(g, i0)]
    out = 0.0
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                w = ((t[0] if cx else 1 - t[0]) * (t[1] if cy else 1 - t[1])
                     * (t[2] if cz else 1 - t[2]))
                out = out + w * grid[i0[0] + cx, i0[1] + cy, i0[2] + cz]
    return out


def sizing_edge_cluster(xyz):
    """h rises linearly from --edge-h-min on the edges of the cube to
    --edge-h-max at the largest distance from them."""
    grid = _edge_distance_grid(EDGE_GRID_RESOLUTION).to(device=xyz.device, dtype=xyz.dtype)
    d = _trilinear_sample(grid, xyz[..., 0], xyz[..., 1], xyz[..., 2])
    return EDGE_H_MIN + (EDGE_H_MAX - EDGE_H_MIN) * (d / EDGE_MAX_DISTANCE).clamp(0.0, 1.0)


def sizing_edge_cluster_sharp(xyz):
    """The earlier, rapidly saturating exponential profile."""
    grid = _edge_distance_grid(EDGE_GRID_RESOLUTION).to(device=xyz.device, dtype=xyz.dtype)
    d = _trilinear_sample(grid, xyz[..., 0], xyz[..., 1], xyz[..., 2])
    return 0.10 + 3.0 * (1.0 - torch.exp(-d ** 2 / (2.0 * 0.10 ** 2)))


COMMON_FIELDS = {
    "uniform": field_uniform,
    "gaussian": field_gaussian,
    "two_blobs": field_two_blobs,
    "z_heavy": field_z_heavy,
    "diagonal": field_diagonal,
    "linear_xyz_123": field_linear_xyz_123,
    "shell": field_shell,
}
SIZING_FIELDS = dict(COMMON_FIELDS, **{
    "linear_xyz_123_gradual": sizing_linear_xyz_123_gradual,
    "edge_cluster": sizing_edge_cluster,
    "edge_cluster_gradual": sizing_edge_cluster,
    "edge_cluster_sharp": sizing_edge_cluster_sharp,
    "edge_cluster_analytic": sizing_edge_cluster_analytic,
})
DENSITY_FIELDS = dict(COMMON_FIELDS, **{
    "linear_xyz_123_gradual": density_linear_xyz_123_gradual,
})
# The edge fields are defined by their spacing, so --density uses them as sizing.
EDGE_FIELDS = ("edge_cluster", "edge_cluster_gradual", "edge_cluster_sharp",
               "edge_cluster_analytic")


def density_function(name, is_sizing):
    as_sizing = is_sizing or name in EDGE_FIELDS
    fields = SIZING_FIELDS if as_sizing else DENSITY_FIELDS
    try:
        field = fields[name]
    except KeyError as exc:
        choices = ", ".join(sorted(fields))
        raise SystemExit(f"Unknown field '{name}'. Choose one of: {choices}") from exc
    if not as_sizing:
        return field
    return lambda xyz: field(xyz).clamp_min(1e-12).pow(-5.0)


# ---------------------------------------------------------------------------
# Quadrature on tetrahedra
# ---------------------------------------------------------------------------

# The symmetric 4-point rule that is exact for quadratics: each point has
# weight BETA on one corner and ALPHA on the other three.
_QUAD4_ALPHA = (5.0 - 5.0 ** 0.5) / 20.0
_QUAD4_BETA = 1.0 - 3.0 * _QUAD4_ALPHA

# The eight tetrahedra of the midpoint subdivision of a tetrahedron, as
# barycentric combinations of its corners. They all have 1/8 of its volume.
_OCTASECTION = torch.tensor(
    [
        [[1, 0, 0, 0], [0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5]],
        [[0, 1, 0, 0], [0.5, 0.5, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0, 0.5]],
        [[0, 0, 1, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0.5, 0.5]],
        [[0, 0, 0, 1], [0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]],
        [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5]],
        [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5]],
        [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0.5, 0, 0.5], [0, 0.5, 0.5, 0]],
        [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0.5, 0.5, 0], [0.5, 0, 0.5, 0]],
    ],
    dtype=torch.float64,
)


@functools.lru_cache(maxsize=8)
def _subdivision(level):
    """(8^level, 4, 4) barycentric matrices of the pieces after `level`
    midpoint subdivisions, relative to the corners of the original tetrahedron."""
    current = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    for _ in range(level):
        current = torch.einsum("scv,kvd->kscd", _OCTASECTION, current).reshape(-1, 4, 4)
    return current


def quadrature_points(corners, quad_points, quad_refine):
    """The quadrature points of each tetrahedron, and the fraction of its
    volume that each one carries. corners has shape (T, 4, 3); returns points
    of shape (T, Q, 3) and weights of shape (Q,), summing to one."""
    if quad_points == 1:
        rule = torch.full((1, 4), 0.25, dtype=torch.float64)
    else:
        rule = torch.full((4, 4), _QUAD4_ALPHA, dtype=torch.float64)
        rule.fill_diagonal_(_QUAD4_BETA)
    # Barycentric coordinates of every point of every piece, relative to the
    # corners of the original tetrahedron.
    bary = torch.einsum("qc,mcd->mqd", rule, _subdivision(quad_refine)).reshape(-1, 4)
    bary = bary.to(device=corners.device, dtype=corners.dtype)
    weights = torch.full((bary.shape[0],), 1.0 / bary.shape[0],
                         dtype=corners.dtype, device=corners.device)
    return torch.einsum("qc,tcd->tqd", bary, corners), weights


# ---------------------------------------------------------------------------
# The density-weighted CVT energy, from the per-cell vertices
# ---------------------------------------------------------------------------

def face_center_tetrahedra(result, points_t, vertices):
    """Split every cell into tetrahedra (p_i, c_f, v_{k+1}, v_k), one for each
    edge (v_k, v_{k+1}) of each face f, where p_i is the generator and c_f is
    the average of the vertices of the face.

    Unlike voroplex.cell_fan_simplices, which splits each face into triangles
    from its first vertex, these tetrahedra do not depend on how Voro++
    numbers the vertices of a cell. The numbering can change between runs
    with several threads, since the points are inserted in parallel, and the
    quadrature of a varying density depends on the tetrahedra, so this keeps
    the results reproducible. Voro++ lists the vertices of each face
    clockwise as seen from outside the cell, so the order (v_{k+1}, v_k) gives
    a positive signed volume. Returns the owner of each tetrahedron and its
    corners, of shape (T, 4, 3).
    """
    device = points_t.device
    fvo = result.facet_vertex_offsets
    k = np.diff(fvo)
    face_owner = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    face = np.repeat(np.arange(len(k)), k)
    pos = np.arange(len(result.facet_vertices)) - np.repeat(fvo[:-1], k)
    base = result.cell_vertex_offsets[face_owner[face]]
    this = base + result.facet_vertices
    nxt = base + result.facet_vertices[fvo[face] + (pos + 1) % k[face]]
    t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
    face_t = t(face)
    counts = torch.as_tensor(k, dtype=vertices.dtype, device=device)
    centers = vertices.new_zeros((len(k), 3)).index_add(0, face_t, vertices[t(this)])
    centers = centers / counts[:, None]
    owners = t(face_owner[face])
    corners = torch.stack([points_t[owners], centers[face_t], vertices[t(nxt)],
                           vertices[t(this)]], dim=1)
    return owners, corners


def cell_terms(result, points_t, vertices, density_fn, quad_points, quad_refine):
    """The energy, mass, and density-weighted centroid of every cell, from the
    tetrahedra of face_center_tetrahedra."""
    owners, corners = face_center_tetrahedra(result, points_t, vertices)
    p = corners[:, 0]
    q = corners[:, 1:] - p[:, None]
    volume = (q[:, 0] * torch.linalg.cross(q[:, 1], q[:, 2], dim=1)).sum(1) / 6.0
    samples, fractions = quadrature_points(corners, quad_points, quad_refine)
    weights = volume[:, None] * fractions * density_fn(samples).clamp_min(1e-12)
    energy = (weights * (samples - p[:, None]).square().sum(-1)).sum(-1)
    moment = (weights[..., None] * samples).sum(1)
    n = points_t.shape[0]
    energies = points_t.new_zeros(n).index_add(0, owners, energy)
    masses = points_t.new_zeros(n).index_add(0, owners, weights.sum(-1))
    moments = points_t.new_zeros((n, 3)).index_add(0, owners, moment)
    return energies, masses, moments / masses.clamp_min(1e-300)[:, None]


def boundary_face_terms(result, vertices, density_fn):
    """Area and circularity penalties for the faces on the boundary of the domain.

    For a sizing field, equal face areas would remove the intended variation
    in size, so the area penalty equalizes area / h^2, with h = rho^(-1/5) at
    the center of the face. Returns (area loss, circularity loss, coefficient
    of variation of the areas, number of faces).
    """
    device = vertices.device
    nb = result.facet_neighbors
    fvo = result.facet_vertex_offsets
    lengths = np.diff(fvo)
    owner = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    on_boundary = np.flatnonzero((nb < 0) & (lengths >= 3))
    if on_boundary.size == 0:
        zero = vertices.new_zeros(())
        return zero, zero, float("nan"), 0
    # Each vertex of each face on the boundary, and the next vertex around the face.
    k = lengths[on_boundary]
    face = np.repeat(np.arange(on_boundary.size), k)
    start = np.repeat(fvo[on_boundary], k)
    pos = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    base = np.repeat(result.cell_vertex_offsets[owner[on_boundary]], k)
    this = base + result.facet_vertices[start + pos]
    nxt = base + result.facet_vertices[start + (pos + 1) % np.repeat(k, k)]
    first = base + result.facet_vertices[start]
    t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
    v, w, v0 = vertices[t(this)], vertices[t(nxt)], vertices[t(first)]
    face_t = t(face)
    n_faces = on_boundary.size
    # Area from the cross products of the triangles that share the first vertex.
    cross = vertices.new_zeros((n_faces, 3)).index_add(
        0, face_t, torch.linalg.cross(v - v0, w - v0, dim=1))
    areas = 0.5 * torch.linalg.norm(cross, dim=1)
    perimeters = vertices.new_zeros(n_faces).index_add(
        0, face_t, torch.linalg.norm(w - v, dim=1))
    counts = torch.as_tensor(k, dtype=vertices.dtype, device=device)
    centers = vertices.new_zeros((n_faces, 3)).index_add(0, face_t, v) / counts[:, None]
    circularity = (4.0 * torch.pi * areas / (perimeters ** 2 + 1e-12)).clamp(0.0, 1.0)
    h = density_fn(centers).clamp_min(1e-12).pow(-1.0 / 5.0)
    measure = areas / h ** 2
    area_loss = ((measure - measure.mean()) / measure.mean().clamp_min(1e-12)).square().mean()
    circularity_loss = (1.0 - circularity).square().mean()
    area_cv = float((areas.std(unbiased=False) / areas.mean().clamp_min(1e-12)).item())
    return area_loss, circularity_loss, area_cv, n_faces


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw_density_slice(ax, density_fn, z_value, resolution=140):
    g = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([xx, yy, torch.full_like(xx, z_value)], dim=-1)
    ax.imshow(density_fn(grid).detach().cpu().numpy(), extent=(0, 1, 0, 1),
              origin="lower", cmap="viridis", alpha=0.86, zorder=0)


def draw_projected_points(ax, points, z_value, title, density_fn, slice_width=0.08):
    draw_density_slice(ax, density_fn, z_value)
    distance = np.abs(points[:, 2] - z_value)
    mask = distance <= slice_width
    if mask.any():
        alpha = np.clip(1.0 - distance[mask] / slice_width, 0.25, 1.0)
        ax.scatter(points[mask, 0], points[mask, 1], s=16, c="#ff4b4b",
                   edgecolors="white", linewidths=0.35, alpha=alpha, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def save_plot(initial_points, final_points, loss_history, output_path, field_name, density_fn):
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
    ax_initial = fig.add_subplot(grid[0, 0], projection="3d")
    ax_final = fig.add_subplot(grid[0, 1], projection="3d")
    ax_loss = fig.add_subplot(grid[0, 2])
    for ax, points, title in ((ax_initial, initial_points, "Initial generators"),
                              (ax_final, final_points, "Density-field CVT")):
        values = density_fn(torch.tensor(points, dtype=torch.float64)).detach().cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap="viridis", s=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    ax_loss.semilogy(np.arange(1, len(loss_history) + 1), loss_history,
                     color="#2454a6", linewidth=2)
    ax_loss.set_title("Objective")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Density-weighted CVT energy")
    ax_loss.grid(True, alpha=0.25)
    for i, z_value in enumerate((0.25, 0.50, 0.75)):
        ax = fig.add_subplot(grid[1, i])
        draw_projected_points(ax, final_points, z_value, f"Final slice z={z_value:.2f}",
                              density_fn)
    fig.suptitle(f"3D field: {field_name}", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ParaView output. These writers do not need VTK.
# ---------------------------------------------------------------------------

def _vtk_values(values):
    return " ".join(str(x) for x in np.asarray(values).reshape(-1))


def _xml_array(dtype, name, values, components=None):
    comps = f' NumberOfComponents="{components}"' if components else ""
    name = f' Name="{name}"' if name else ""
    return (f'        <DataArray type="{dtype}"{name}{comps} format="ascii">'
            f"{_vtk_values(values)}</DataArray>\n")


def export_density_field_volume(density_fn, output_path, limits, resolution=51):
    """Write the density, and the spacing h = rho^(-1/5), on a grid (VTI)."""
    axes = [torch.linspace(lo, hi, resolution, dtype=torch.float64) for lo, hi in limits]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    with torch.no_grad():
        density = density_fn(grid).cpu().numpy()
    sizing = np.maximum(density, 1e-12) ** (-1.0 / 5.0)
    spacing = [(hi - lo) / (resolution - 1) for lo, hi in limits]
    origin = [lo for lo, _ in limits]
    extent = f"0 {resolution - 1} 0 {resolution - 1} 0 {resolution - 1}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <ImageData WholeExtent="{extent}" Origin="{_vtk_values(origin)}" '
                f'Spacing="{_vtk_values(spacing)}">\n')
        f.write(f'    <Piece Extent="{extent}">\n')
        f.write('      <PointData Scalars="DensityValue">\n')
        # VTK image data has x as the fastest-varying index.
        f.write(_xml_array("Float64", "SizingValue", sizing.reshape(-1, order="F")))
        f.write(_xml_array("Float64", "DensityValue", density.reshape(-1, order="F")))
        f.write("      </PointData>\n      <CellData/>\n    </Piece>\n")
        f.write("  </ImageData>\n</VTKFile>\n")
    return {"path": str(output_path), "resolution": [resolution] * 3,
            "sizing_min": float(sizing.min()), "sizing_max": float(sizing.max()),
            "density_min": float(density.min()), "density_max": float(density.max())}


def selected_faces(result, points, clip_spec=None):
    """The faces of the selected cells, each listed once, with their vertices.

    result is a geometry result, which stores each face once for each of its
    two cells. A face between two selected cells is kept from the cell with
    the lower ID. Copies of a vertex in different cells are merged when their
    coordinates agree to rounding error, so that the mesh is connected.
    """
    n = len(result)
    owner = np.repeat(np.arange(n), np.diff(result.cell_facet_offsets))
    nb = result.facet_neighbors
    selected = clip_mask_for_points(points, clip_spec)
    nb_selected = np.zeros(len(nb), dtype=bool)
    nb_selected[nb >= 0] = selected[nb[nb >= 0]]
    keep = selected[owner] & ~(nb_selected & (nb < owner))
    fvo = result.facet_vertex_offsets
    lengths = np.diff(fvo)[keep]
    starts = fvo[:-1][keep]
    refs = np.concatenate([np.arange(s, s + k) for s, k in zip(starts, lengths)]) \
        if lengths.size else np.empty(0, dtype=np.int64)
    global_ids = np.repeat(result.cell_vertex_offsets[owner[keep]], lengths) \
        + result.facet_vertices[refs]
    # Merge vertex copies whose coordinates agree to about 1e-10 of the domain.
    vertices = result.vertices[global_ids]
    scale = float(np.ptp(result.limits, axis=1).max())
    keys = np.round(vertices / (scale * 1e-10)).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    unique_vertices = np.zeros((len(unique_keys), 3))
    unique_vertices[inverse.reshape(-1)] = vertices
    face_offsets = np.concatenate([[0], np.cumsum(lengths)])
    return {
        "vertices": unique_vertices,
        "connectivity": inverse.reshape(-1),
        "offsets": face_offsets,
        "owner": owner[keep],
        "neighbor": nb[keep],
        "is_surface": (nb[keep] < 0).astype(np.int32),
    }


def export_faces_vtu(result, mesh, output_path):
    """Write the faces as VTK polygons (VTU)."""
    v = mesh["vertices"]
    conn, offs = mesh["connectivity"], mesh["offsets"]
    areas = np.empty(len(offs) - 1)
    for i in range(len(offs) - 1):
        q = v[conn[offs[i]:offs[i + 1]]]
        areas[i] = 0.5 * np.linalg.norm(np.cross(q[1:-1] - q[0], q[2:] - q[0]).sum(axis=0))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{len(v)}" NumberOfCells="{len(offs) - 1}">\n')
        f.write('      <CellData Scalars="IsSurface">\n')
        f.write(_xml_array("Int64", "OwnerCell", mesh["owner"]))
        f.write(_xml_array("Int64", "NeighborCell", mesh["neighbor"]))
        f.write(_xml_array("Int32", "IsSurface", mesh["is_surface"]))
        f.write(_xml_array("Float64", "FaceArea", areas))
        f.write(_xml_array("Float64", "OwnerVolume", result.volumes[mesh["owner"]]))
        f.write("      </CellData>\n      <Points>\n")
        f.write(_xml_array("Float64", None, v, components=3))
        f.write("      </Points>\n      <Cells>\n")
        f.write(_xml_array("Int64", "connectivity", conn))
        f.write(_xml_array("Int64", "offsets", offs[1:]))
        f.write(_xml_array("UInt8", "types", np.full(len(offs) - 1, 7)))  # VTK_POLYGON
        f.write("      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")


def export_edges_vtp(mesh, output_path):
    """Write the edges of the faces, each listed once, as VTK lines (VTP)."""
    conn, offs = mesh["connectivity"], mesh["offsets"]
    lengths = np.diff(offs)
    face = np.repeat(np.arange(len(lengths)), lengths)
    pos = np.arange(len(conn)) - offs[face]
    nxt = conn[offs[face] + (pos + 1) % lengths[face]]
    edges = np.unique(np.sort(np.stack([conn, nxt], axis=1), axis=1), axis=0)
    v = mesh["vertices"]
    edge_lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{len(v)}" NumberOfVerts="0" '
                f'NumberOfLines="{len(edges)}" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('      <CellData Scalars="EdgeLength">\n')
        f.write(_xml_array("Float64", "EdgeLength", edge_lengths))
        f.write("      </CellData>\n      <Points>\n")
        f.write(_xml_array("Float64", None, v, components=3))
        f.write("      </Points>\n      <Lines>\n")
        f.write(_xml_array("Int64", "connectivity", edges.reshape(-1)))
        f.write(_xml_array("Int64", "offsets", np.arange(2, 2 * len(edges) + 1, 2)))
        f.write("      </Lines>\n    </Piece>\n  </PolyData>\n</VTKFile>\n")


def export_points_vtp(points, output_path, density_fn, point_ids):
    """Write the generators as VTK points (VTP), with their density values."""
    points = np.asarray(points, dtype=np.float64)
    values = density_fn(torch.tensor(points)).detach().cpu().numpy()
    n = len(points)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" '
                'NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('      <PointData Scalars="DensityValue">\n')
        f.write(_xml_array("Int64", "GeneratorId", point_ids))
        f.write(_xml_array("Float64", "DensityValue", values))
        f.write("      </PointData>\n      <Points>\n")
        f.write(_xml_array("Float64", None, points, components=3))
        f.write("      </Points>\n      <Verts>\n")
        f.write(_xml_array("Int64", "connectivity", np.arange(n)))
        f.write(_xml_array("Int64", "offsets", np.arange(1, n + 1)))
        f.write("      </Verts>\n    </Piece>\n  </PolyData>\n</VTKFile>\n")


def export_voronoi_outputs(points, limits, threads, output_prefix, density_fn, clip_spec):
    """Write the faces, edges, and generators of the (selected) cells."""
    points = np.asarray(points, dtype=np.float64)
    result = voroplex.compute_voronoi_3d(points, limits, threads=threads)
    mesh = selected_faces(result, points, clip_spec)
    paths = {"mesh": f"{output_prefix}.vtu", "edges": f"{output_prefix}_edges.vtp",
             "cell_centers": f"{output_prefix}_cell_centers.vtp"}
    export_faces_vtu(result, mesh, paths["mesh"])
    export_edges_vtp(mesh, paths["edges"])
    ids = np.flatnonzero(clip_mask_for_points(points, clip_spec))
    export_points_vtp(points[ids], paths["cell_centers"], density_fn, ids)
    paths["num_selected_centers"] = int(ids.size)
    if clip_spec is not None:
        paths["clip"] = describe_clip_spec(clip_spec)
    return paths


def export_paraview_collection(output_path, named_paths):
    """Write one .vtm file that opens all of the datasets in ParaView."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n')
        f.write("  <vtkMultiBlockDataSet>\n")
        for index, (name, path) in enumerate(named_paths):
            rel = os.path.relpath(path, start=output_path.parent)
            f.write(f'    <DataSet index="{index}" name="{name}" file="{rel}"/>\n')
        f.write("  </vtkMultiBlockDataSet>\n</VTKFile>\n")
    return str(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    threads = parse_option("threads", parse_int_arg(0, 4), int)
    n_points = parse_option("num-points", parse_int_arg(1, 120), int)
    n_iter = parse_option("num-iters", parse_int_arg(2, 150), int)
    schedule_iters = parse_option("lr-schedule-iters", n_iter, int)
    lr = parse_option("lr", parse_float_arg(3, 0.003), float)
    lr_min = parse_option("lr-min", lr * 0.1, float)
    vol_reg = parse_option("vol-reg", 0.0, float)
    face_area_reg = parse_option("face-area-reg", 0.0, float)
    face_circ_reg = parse_option("face-circ-reg", 0.0, float)
    field_resolution = parse_option("field-resolution", 51, int)
    quad_points = parse_option("quad-points", 4, int)
    quad_refine = parse_option("quad-refine", 0, int)
    converge_tol = parse_option("converge-tol", 1e-4, float)
    converge_window = parse_option("converge-window", 30, int)
    init_from = parse_option("init-from", None)
    if schedule_iters < 1:
        raise SystemExit("--lr-schedule-iters must be >= 1")
    if field_resolution < 2:
        raise SystemExit("--field-resolution must be >= 2")
    if EDGE_H_MIN <= 0.0 or EDGE_H_MAX < EDGE_H_MIN:
        raise SystemExit("--edge-h-min must be > 0 and no larger than --edge-h-max")
    if LINEAR_GRADUAL_H_MIN <= 0.0 or LINEAR_GRADUAL_H_MAX < LINEAR_GRADUAL_H_MIN:
        raise SystemExit("--linear-h-min must be > 0 and no larger than --linear-h-max")
    if quad_points not in (1, 4):
        raise SystemExit("--quad-points must be 1 or 4")
    if quad_refine < 0:
        raise SystemExit("--quad-refine must be >= 0")
    if parse_option("impl", None) is not None:
        print("Note: --impl is no longer used; the integration is always vectorized.")
    device = resolve_device(parse_option("device", "auto").lower())
    dtype = torch.float32 if device.type == "mps" else torch.float64
    field_name, is_sizing = parse_field()
    density_fn = density_function(field_name, is_sizing)
    make_vtu = "--no-vtu" not in sys.argv
    output_dir = Path(parse_option("output-dir", os.environ.get("VOROPLEX_OUTPUT_DIR", "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = [[0.0, 1.0]] * 3
    clip_spec = build_clip_spec(limits)

    torch.manual_seed(SEED)
    if init_from is not None:
        with open(init_from, encoding="utf-8") as f:
            initial_points = np.asarray(json.load(f)["final_points"], dtype=np.float64)
        n_points = len(initial_points)
        print(f"Starting from the final points of {init_from} ({n_points} points)")
    else:
        initial_points = np.random.default_rng(SEED).random((n_points, 3)) * 0.9 + 0.05
    kind = "sizing" if is_sizing else "density"
    stem = output_dir / f"cvt_3d_{kind}_{field_name}_{n_points}pts"
    points_t = torch.tensor(initial_points, dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([points_t], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule_iters, eta_min=lr_min)

    history = {name: [] for name in ("objective", "energy", "residual", "volume_penalty",
                                     "face_area_loss", "face_circularity_loss",
                                     "face_area_cv")}
    print(f"3D density-field CVT - {n_points} points, at most {n_iter} iterations, "
          f"{threads} thread(s), device={device}")
    print(f"Field: {field_name} ({kind}), lr={lr}->{lr_min} (cosine), "
          f"quad_points={quad_points}, quad_refine={quad_refine}, vol_reg={vol_reg}, "
          f"face_area_reg={face_area_reg}, face_circ_reg={face_circ_reg}")
    if field_name in EDGE_FIELDS[:2]:
        print(f"Edge sizing: h = {EDGE_H_MIN:g} on the edges to {EDGE_H_MAX:g} at the center")
    print(f"Stopping when the range of the objective over {converge_window} iterations, "
          f"divided by its mean, is below {converge_tol:.1e}")
    print(f"{'Iter':>4}  {'Objective':>14}  {'Mean mass':>12}  {'Mean residual':>14}  "
          f"{'Vol penalty':>12}  {'Bdry CV':>8}")
    print("-" * 78)

    outputs = {}
    if make_vtu:
        outputs["density_field"] = export_density_field_volume(
            density_fn, f"{stem}_density_field.vti", limits, resolution=field_resolution)
        outputs["initial"] = export_voronoi_outputs(
            initial_points, limits, threads, f"{stem}_initial", density_fn, clip_spec)

    t0 = time.perf_counter()
    for it in range(n_iter):
        result = voroplex.compute_voronoi_3d(
            points_t.detach().cpu().numpy().astype(np.float64), limits, threads=threads,
            representation="per_cell", properties=(),
        )
        vertices = voroplex.cell_vertices(result, points_t)
        energies, masses, centroids = cell_terms(
            result, points_t, vertices, density_fn, quad_points, quad_refine)
        energy = energies.sum()
        mean_mass = masses.mean().clamp_min(1e-300)
        volume_penalty = ((masses - mean_mass) / mean_mass).square().mean()
        objective = energy + vol_reg * volume_penalty
        area_loss, circ_loss, area_cv, _ = boundary_face_terms(result, vertices, density_fn)
        if face_area_reg or face_circ_reg:
            objective = objective + face_area_reg * area_loss + face_circ_reg * circ_loss

        optimizer.zero_grad()
        objective.backward()
        with torch.no_grad():
            residuals = torch.linalg.norm(points_t - centroids, dim=1)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            points_t.clamp_(1e-4, 1.0 - 1e-4)

        for name, value in (("objective", objective), ("energy", energy),
                            ("residual", residuals.mean()),
                            ("volume_penalty", volume_penalty),
                            ("face_area_loss", area_loss),
                            ("face_circularity_loss", circ_loss)):
            history[name].append(float(value.item()))
        history["face_area_cv"].append(area_cv)
        if it == 0 or (it + 1) % 10 == 0 or it + 1 == n_iter:
            print(f"{it + 1:>4}  {history['objective'][-1]:>14.8f}  "
                  f"{masses.mean().item():>12.8f}  {history['residual'][-1]:>14.8f}  "
                  f"{history['volume_penalty'][-1]:>12.6f}  {area_cv:>8.4f}")
        if len(history["objective"]) >= converge_window:
            recent = np.asarray(history["objective"][-converge_window:])
            rel_range = (recent.max() - recent.min()) / (abs(recent.mean()) + 1e-300)
            if rel_range < converge_tol:
                print(f"Converged at iteration {it + 1}: relative range "
                      f"{rel_range:.2e} < {converge_tol:.1e}")
                break
    iterations = it + 1
    elapsed = time.perf_counter() - t0
    final_points = points_t.detach().cpu().numpy().astype(np.float64)

    plot_path = f"{stem}.png"
    save_plot(initial_points, final_points, history["objective"], plot_path,
              field_name, density_fn)
    if make_vtu:
        outputs["final"] = export_voronoi_outputs(
            final_points, limits, threads, f"{stem}_final", density_fn, clip_spec)
        outputs["paraview_collection"] = export_paraview_collection(f"{stem}_paraview.vtm", [
            ("Density field", outputs["density_field"]["path"]),
            ("Initial mesh", outputs["initial"]["mesh"]),
            ("Initial edges", outputs["initial"]["edges"]),
            ("Initial cell centers", outputs["initial"]["cell_centers"]),
            ("Final mesh", outputs["final"]["mesh"]),
            ("Final edges", outputs["final"]["edges"]),
            ("Final cell centers", outputs["final"]["cell_centers"]),
        ])

    params_path = f"{stem}_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "script": Path(__file__).name,
            "argv": sys.argv[1:],
            "voroplex_version": voroplex.__version__,
            "seed": SEED,
            "threads": threads,
            "num_points": n_points,
            "num_iters": iterations,
            "num_iters_requested": n_iter,
            "converged_early": iterations < n_iter,
            "converge_tol": converge_tol,
            "converge_window": converge_window,
            "learning_rate": lr,
            "lr_min": lr_min,
            "lr_schedule_iters": schedule_iters,
            "quad_points": quad_points,
            "quad_refine": quad_refine,
            "vol_reg_weight": vol_reg,
            "face_area_reg_weight": face_area_reg,
            "face_circ_reg_weight": face_circ_reg,
            "edge_h_min": EDGE_H_MIN,
            "edge_h_max": EDGE_H_MAX,
            "linear_h_min": LINEAR_GRADUAL_H_MIN,
            "linear_h_max": LINEAR_GRADUAL_H_MAX,
            "device": str(device),
            "dtype": str(dtype),
            "init_from": init_from,
            "field": field_name,
            "field_input": kind,
            "sizing_to_density": "rho = h^(-5)",
            "clip": clip_spec,
            "limits": limits,
            "loop_time_seconds": elapsed,
            "output_plot": plot_path,
            "outputs": outputs,
            "initial_points": initial_points.tolist(),
            "final_points": final_points.tolist(),
            "history": history,
        }, f, indent=2)

    print("-" * 78)
    print(f"Ran:              {iterations}/{n_iter} iterations"
          f"{' (converged)' if iterations < n_iter else ''}")
    print(f"Initial energy:   {history['energy'][0]:.8f}")
    print(f"Final energy:     {history['energy'][-1]:.8f}")
    print(f"Initial residual: {history['residual'][0]:.8f}")
    print(f"Final residual:   {history['residual'][-1]:.8f}")
    print(f"Loop time:        {elapsed:.2f} s")
    print(f"Saved plot:       {plot_path}")
    print(f"Saved params:     {params_path}")
    if make_vtu:
        print(f"Open in ParaView: {outputs['paraview_collection']}")


if __name__ == "__main__":
    main()
