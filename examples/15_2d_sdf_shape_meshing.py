"""
2D SDF Boundary Meshing with VoroPlex + PyTorch
===============================================

Labels 2D generators by a signed distance function (SDF), extracts inner/outer
interface Voronoi edges with ``n_inner``, and optimizes the generators so the
interface edge midpoints and endpoints lie on ``phi = 0``. Differentiable
vertices use VoroPlex's per-cell representation and analytical backward pass.

The inner cells can also use an SDF-distance adaptive density field:

    mu(x, y) = local desired cell size, smaller near ``phi = 0``
    rho(x,y) = 1 / mu(x, y)^4

The loss is:

    L = lambda_mid  * L_mid
      + lambda_end  * L_end
      + lambda_in   * L_in
      + lambda_area * L_area
      + lambda_circ * L_circ

where L_mid and L_end penalize the SDF values at interface-edge midpoints and
endpoints. L_in is a density-weighted inner-cell CVT energy, and L_area and
L_circ regularize the area and circularity of the inner cells adjacent to the
extracted interface.

Run from the project root:
    python examples/15_2d_sdf_shape_meshing.py

The defaults (1000 generators, 600 iterations, learning rate 5e-4, one
thread) are the settings of Section 5.2 of the VoroPlex paper. The six shapes
in the paper are:
    python examples/15_2d_sdf_shape_meshing.py --shape=flower
    python examples/15_2d_sdf_shape_meshing.py --shape=circle
    python examples/15_2d_sdf_shape_meshing.py --shape=star
    python examples/15_2d_sdf_shape_meshing.py --shape=rounded_square
    python examples/15_2d_sdf_shape_meshing.py --shape=heart
    python examples/15_2d_sdf_shape_meshing.py --shape=peanut

With more than one thread, the gradient is summed in a different order, so
the results differ slightly from run to run (by about 1e-3 in the final
generator positions after 600 iterations).

Optional arguments:
    python examples/15_2d_sdf_shape_meshing.py [num_threads] [num_points] [num_iters] [lr]
    python examples/15_2d_sdf_shape_meshing.py 1 180 300 0.0005 --shape=peanut
    python examples/15_2d_sdf_shape_meshing.py --density-mode=uniform
    python examples/15_2d_sdf_shape_meshing.py --no-gif --no-vtu
    python examples/15_2d_sdf_shape_meshing.py --output-dir=/tmp/voroplex-sdf

Available shapes:
    circle, ellipse, flower, star, heart, rounded_square, peanut

Paper figures (run from the project root, using one thread for reproducibility):
    # Figure 12: flower optimization and weighted loss components
    python examples/15_2d_sdf_shape_meshing.py --shape=flower --paper-figure --no-gif --no-vtu

    # Figure 13: first run the five shapes into the same output directory
    python examples/15_2d_sdf_shape_meshing.py --shape=circle --no-gif --no-vtu
    python examples/15_2d_sdf_shape_meshing.py --shape=star --no-gif --no-vtu
    python examples/15_2d_sdf_shape_meshing.py --shape=rounded_square --no-gif --no-vtu
    python examples/15_2d_sdf_shape_meshing.py --shape=heart --no-gif --no-vtu
    python examples/15_2d_sdf_shape_meshing.py --shape=peanut --no-gif --no-vtu
    python examples/15_2d_sdf_shape_meshing.py --paper-gallery

The paper-only modes add 450-dpi PNG and vector PDF outputs without changing
the standard Example 15 plots or optimization outputs. ``--paper-gallery``
reads the five shape-specific parameter JSON files from ``--output-dir``.
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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

import voroplex


SDF_SHAPE = "flower"
LAMBDA_MIDPOINT_SDF = 10.0
LAMBDA_ENDPOINT_SDF = 2.0
LAMBDA_INNER_CVT = 0.0005
LAMBDA_INNER_CVT_UNIFORM = 20.0
LAMBDA_BOUNDARY_AREA = 0.05
LAMBDA_BOUNDARY_CIRCULARITY = 0.02
EPS = 1e-12
GIF_FRAME_COUNT = 60
GIF_FPS = 12
DENSITY_MODE = "sdf-distance"
DENSITY_MU_MIN = 0.030
DENSITY_MU_MAX = 0.060
DENSITY_BOUNDARY_WIDTH = 0.24

POSITIONAL_ARGS = [arg for arg in sys.argv[1:] if not arg.startswith("--")]


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


def parse_option(name, default):
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def parse_output_dir(default):
    prefix = "--output-dir="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return Path(arg[len(prefix):])
    return Path(default)


def parse_int_option(name, default):
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            try:
                return int(arg[len(prefix):])
            except ValueError:
                return default
    return default


def parse_float_option(name, default):
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            try:
                return float(arg[len(prefix):])
            except ValueError:
                return default
    return default


def canonical_density_mode(name):
    aliases = {
        "uniform": "uniform",
        "none": "uniform",
        "off": "uniform",
        "sdf-distance": "sdf-distance",
        "sdf_distance": "sdf-distance",
        "version-a": "sdf-distance",
        "version_a": "sdf-distance",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        choices = ", ".join(sorted(set(aliases.values())))
        raise SystemExit(f"Unknown density mode '{name}'. Choose one of: {choices}") from exc


def sdf_circle(xy):
    """Circle of radius 0.28 centered at (0.5, 0.5); an exact SDF."""
    center = torch.tensor([0.5, 0.5], dtype=xy.dtype, device=xy.device)
    return torch.linalg.norm(xy - center, dim=-1) - 0.28


def sdf_ellipse(xy):
    """Ellipse with semi-axes 0.34 and 0.22, centered at (0.5, 0.5).

    This is negative inside and positive outside, but it is only approximately a
    distance, since the coordinates are scaled differently in x and y.
    """
    center = torch.tensor([0.5, 0.5], dtype=xy.dtype, device=xy.device)
    q = (xy - center) / torch.tensor([0.34, 0.22], dtype=xy.dtype, device=xy.device)
    return torch.linalg.norm(q, dim=-1) - 1.0


def sdf_flower(xy):
    """Five-petal flower, with boundary r = 0.27 + 0.055 cos(5 theta + 0.35).

    The value r - r_*(theta) is negative inside and positive outside. It is only
    approximately a distance, since it is measured along the radial direction.
    """
    center = torch.tensor([0.5, 0.5], dtype=xy.dtype, device=xy.device)
    rel = xy - center
    theta = torch.atan2(rel[..., 1], rel[..., 0])
    radius = torch.linalg.norm(rel, dim=-1)
    target = 0.27 + 0.055 * torch.cos(5.0 * theta + 0.35)
    return radius - target


def signed_distance_to_polygon(xy, vertices):
    """Exact signed distance from points xy (..., 2) to a closed polygon.

    The distance is the smallest distance to any of the polygon's edges, each
    found by projecting the point onto the edge and clamping to its ends. The
    sign is negative inside: a horizontal ray from the point crosses the polygon
    an odd number of times exactly when the point is inside.
    """
    p = xy.unsqueeze(-2)
    a = vertices
    b = torch.roll(vertices, shifts=-1, dims=0)
    ba = b - a
    pa = p - a
    h = torch.clamp(
        torch.sum(pa * ba, dim=-1) / torch.clamp(torch.sum(ba * ba, dim=-1), min=EPS),
        0.0,
        1.0,
    )
    closest = a + h.unsqueeze(-1) * ba
    min_dist = torch.sqrt(torch.min(torch.sum((p - closest) ** 2, dim=-1), dim=-1).values + EPS)

    x = xy[..., 0].unsqueeze(-1)
    y = xy[..., 1].unsqueeze(-1)
    xi = a[:, 0]
    yi = a[:, 1]
    xj = b[:, 0]
    yj = b[:, 1]
    crosses_y = (yi > y) != (yj > y)
    crossing_x = (xj - xi) * (y - yi) / torch.where(
        torch.abs(yj - yi) > EPS,
        yj - yi,
        torch.full_like(yj, EPS),
    ) + xi
    inside = torch.sum(crosses_y & (x < crossing_x), dim=-1).remainder(2).bool()
    return torch.where(inside, -min_dist, min_dist)


def sdf_star(xy):
    """Five-pointed star with slightly rounded corners, from its polygon."""
    return signed_distance_to_polygon(xy, _star_polygon(xy.dtype, xy.device))


@functools.lru_cache(maxsize=None)
def _star_polygon(dtype, device):
    """The vertices of the rounded star, built once for each dtype and device."""
    xy = torch.zeros(1, dtype=dtype, device=device)
    center = torch.tensor([0.5, 0.5], dtype=xy.dtype, device=xy.device)
    ids = torch.arange(10, dtype=xy.dtype, device=xy.device)
    outer_radius = xy.new_tensor(0.32)
    inner_radius = xy.new_tensor(0.125)
    radii = torch.where(ids.remainder(2.0) == 0.0, outer_radius, inner_radius)
    angles = xy.new_tensor(np.pi / 2.0) + ids * xy.new_tensor(np.pi / 5.0)
    vertices = center + radii.unsqueeze(-1) * torch.stack(
        [torch.cos(angles), torch.sin(angles)],
        dim=-1,
    )

    outer_cut = xy.new_tensor(0.26)
    inner_cut = xy.new_tensor(0.16)
    samples_per_corner = 28
    rounded_vertices = []
    for i in range(10):
        vertex = vertices[i]
        previous_vertex = vertices[(i - 1) % 10]
        next_vertex = vertices[(i + 1) % 10]
        cut = outer_cut if i % 2 == 0 else inner_cut
        p_in = vertex + cut * (previous_vertex - vertex)
        p_out = vertex + cut * (next_vertex - vertex)
        t_values = torch.linspace(
            0.0,
            1.0,
            samples_per_corner + 2,
            dtype=xy.dtype,
            device=xy.device,
        )
        for t in t_values[:-1]:
            rounded_vertices.append(
                (1.0 - t) ** 2 * p_in
                + 2.0 * (1.0 - t) * t * vertex
                + t ** 2 * p_out
            )

    return torch.stack(rounded_vertices, dim=0)


def sdf_heart(xy):
    """Heart, from a polygon through a Catmull-Rom spline."""
    return signed_distance_to_polygon(xy, _heart_polygon(xy.dtype, xy.device))


@functools.lru_cache(maxsize=None)
def _heart_polygon(dtype, device):
    """The vertices of the heart, built once for each dtype and device."""
    xy = torch.zeros(1, dtype=dtype, device=device)
    control_points = torch.tensor(
        [
            [0.500, 0.795],
            [0.535, 0.805],
            [0.585, 0.835],
            [0.655, 0.852],
            [0.735, 0.825],
            [0.795, 0.748],
            [0.805, 0.640],
            [0.770, 0.540],
            [0.705, 0.455],
            [0.625, 0.380],
            [0.565, 0.325],
            [0.530, 0.295],
            [0.505, 0.282],
            [0.500, 0.281],
            [0.495, 0.282],
            [0.470, 0.295],
            [0.435, 0.325],
            [0.375, 0.380],
            [0.295, 0.455],
            [0.230, 0.540],
            [0.195, 0.640],
            [0.205, 0.748],
            [0.265, 0.825],
            [0.345, 0.852],
            [0.415, 0.835],
            [0.465, 0.805],
        ],
        dtype=xy.dtype,
        device=xy.device,
    )

    samples_per_segment = 28
    vertices = []
    for i in range(control_points.shape[0]):
        p0 = control_points[(i - 1) % control_points.shape[0]]
        p1 = control_points[i]
        p2 = control_points[(i + 1) % control_points.shape[0]]
        p3 = control_points[(i + 2) % control_points.shape[0]]
        t0 = xy.new_tensor(0.0)
        t1 = t0 + torch.pow(torch.linalg.norm(p1 - p0).clamp_min(EPS), 0.5)
        t2 = t1 + torch.pow(torch.linalg.norm(p2 - p1).clamp_min(EPS), 0.5)
        t3 = t2 + torch.pow(torch.linalg.norm(p3 - p2).clamp_min(EPS), 0.5)
        t_values = torch.linspace(
            t1,
            t2,
            samples_per_segment + 1,
            dtype=xy.dtype,
            device=xy.device,
        )[:-1]
        for t in t_values:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            vertices.append(
                (t2 - t) / (t2 - t1) * b1
                + (t - t1) / (t2 - t1) * b2
            )

    return torch.stack(vertices, dim=0)


def sdf_rounded_square(xy):
    """Square of side 0.58 with corners of radius 0.075; an exact SDF.

    q is the position relative to the square shrunk by the corner radius: outside
    it, the distance is the length of the positive part of q, and inside, it is
    the largest (least negative) component of q.
    """
    center = torch.tensor([0.5, 0.5], dtype=xy.dtype, device=xy.device)
    half_size = torch.tensor([0.29, 0.29], dtype=xy.dtype, device=xy.device)
    radius = xy.new_tensor(0.075)
    q = torch.abs(xy - center) - (half_size - radius)
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.maximum(q[..., 0], q[..., 1]), max=0.0)
    return outside + inside - radius


def sdf_peanut(xy):
    """Union of two circles of radius 0.19, blended into a smooth peanut.

    The smooth minimum -log(exp(-k d1) + exp(-k d2)) / k of the two circles'
    distances replaces min(d1, d2), which would leave a sharp corner where the
    circles meet. It is only approximately a distance near the waist.
    """
    left_center = torch.tensor([0.39, 0.5], dtype=xy.dtype, device=xy.device)
    right_center = torch.tensor([0.61, 0.5], dtype=xy.dtype, device=xy.device)
    radius = xy.new_tensor(0.19)
    blend = xy.new_tensor(35.0)
    d_left = torch.linalg.norm(xy - left_center, dim=-1) - radius
    d_right = torch.linalg.norm(xy - right_center, dim=-1) - radius
    return -torch.logsumexp(torch.stack([-blend * d_left, -blend * d_right], dim=0), dim=0) / blend


SDF_FUNCTIONS = {
    "circle": sdf_circle,
    "ellipse": sdf_ellipse,
    "flower": sdf_flower,
    "heart": sdf_heart,
    "peanut": sdf_peanut,
    "rounded-square": sdf_rounded_square,
    "rounded_square": sdf_rounded_square,
    "star": sdf_star,
}


def get_sdf_function(name):
    try:
        return SDF_FUNCTIONS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SDF_FUNCTIONS))
        raise SystemExit(f"Unknown shape '{name}'. Choose one of: {choices}") from exc


def torch_smooth_ramp01(value):
    """Smoothstep: 0 for value <= 0, 1 for value >= 1, and 3t^2 - 2t^3 between."""
    t = torch.clamp(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sdf_distance_sizing_density(
    xy,
    sdf_fn,
    mu_min=DENSITY_MU_MIN,
    mu_max=DENSITY_MU_MAX,
    boundary_width=DENSITY_BOUNDARY_WIDTH,
):
    """The desired cell size mu and the density rho = mu^-4 at points xy.

    mu rises smoothly from mu_min on the boundary (phi = 0) to mu_max at a
    distance boundary_width or more from it. In 2D, a CVT has cells with a length
    scale proportional to rho^(-1/4), so rho = mu^-4 gives cells of size mu.
    """
    phi = sdf_fn(xy)
    ramp = torch_smooth_ramp01(torch.abs(phi) / boundary_width)
    mu = mu_min + (mu_max - mu_min) * ramp
    rho = 1.0 / torch.clamp(mu, min=EPS) ** 4
    return mu, rho


def make_density_function(mode, sdf_fn, mu_min, mu_max, boundary_width):
    """The density used in the inner CVT energy, as a function of position.

    Returns (mode, density_fn), where density_fn is None for a uniform density.
    """
    mode = canonical_density_mode(mode)
    if mode == "uniform":
        return mode, None

    def density_fn(xy):
        _, rho = sdf_distance_sizing_density(
            xy,
            sdf_fn,
            mu_min=mu_min,
            mu_max=mu_max,
            boundary_width=boundary_width,
        )
        return rho

    return mode, density_fn


def make_loss_weights(density_mode):
    """The weights lambda of the five terms of the objective.

    The inner CVT weight depends on the density mode, since rho = mu^-4 is of
    order 10^6, while a uniform density is 1.
    """
    mode = canonical_density_mode(density_mode)
    inner_cvt_weight = LAMBDA_INNER_CVT_UNIFORM if mode == "uniform" else LAMBDA_INNER_CVT
    return {
        "midpoint_sdf": LAMBDA_MIDPOINT_SDF,
        "endpoint_sdf": LAMBDA_ENDPOINT_SDF,
        "inner_cvt": inner_cvt_weight,
        "boundary_area": LAMBDA_BOUNDARY_AREA,
        "boundary_circularity": LAMBDA_BOUNDARY_CIRCULARITY,
    }


def sorted_by_sdf(points_t, sdf_fn):
    """Order the generators so that those inside the shape (phi <= 0) come first.

    VoroPlex expects the inner generators first, followed by the outer ones.
    Returns (order, n_inner, phi), with order None if all of the generators are on
    one side of the boundary.
    """
    phi = sdf_fn(points_t)
    inside = phi <= 0.0
    inside_ids = torch.where(inside)[0]
    outside_ids = torch.where(~inside)[0]
    if len(inside_ids) == 0 or len(outside_ids) == 0:
        return None, int(len(inside_ids)), phi
    order = torch.cat([inside_ids, outside_ids])
    return order, int(len(inside_ids)), phi


TRIANGLE_RULE = torch.tensor(
    [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
     [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
     [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]],
    dtype=torch.float64,
)


def facet_owners(result):
    """Owner cell ID for every packed facet."""
    return np.repeat(
        np.arange(result.n_cells, dtype=np.int64),
        np.diff(result.cell_facet_offsets),
    )


def interface_edge_rows(result, n_inner):
    """Flat per-cell vertex rows for inner-to-outer interface edges."""
    owners = facet_owners(result)
    neighbors = np.asarray(result.facet_neighbors)
    facet_ids = np.flatnonzero((owners < n_inner) & (neighbors >= n_inner))
    if facet_ids.size == 0:
        return np.empty((0, 2), dtype=np.int64), owners[:0], neighbors[:0]

    starts = result.facet_vertex_offsets[facet_ids]
    stops = result.facet_vertex_offsets[facet_ids + 1]
    if not np.all(stops - starts == 2):
        raise RuntimeError("2D interface facets must have exactly two vertices")

    bases = result.cell_vertex_offsets[owners[facet_ids]]
    rows = np.column_stack([
        bases + result.facet_vertices[starts],
        bases + result.facet_vertices[starts + 1],
    ])
    return rows, owners[facet_ids], neighbors[facet_ids]


def interface_edge_terms(result, vertices, sorted_points_t, n_inner, sdf_fn):
    """The SDF at the midpoint and at both ends of every interface edge.

    The interface edges are those between an inner and an outer cell, and their
    owners are the boundary cells. Returns the SDF values at the midpoints and
    endpoints, the edge lengths and coordinates, the boundary cells, and the
    number of edges whose two generators are on the same side of the boundary,
    which is always zero.
    """
    rows, owners_np, neighbors_np = interface_edge_rows(result, n_inner)
    if rows.shape[0] == 0:
        empty = sorted_points_t.new_empty((0,))
        return empty, empty, empty, sorted_points_t.new_empty((0, 2, 2)), empty, 0

    rows_t = torch.as_tensor(rows, dtype=torch.long, device=vertices.device)
    owners = torch.as_tensor(owners_np, dtype=torch.long, device=vertices.device)
    neighbors = torch.as_tensor(neighbors_np, dtype=torch.long, device=vertices.device)
    edge_xy = vertices[rows_t]
    midpoint_phi = sdf_fn(edge_xy.mean(dim=1))
    endpoint_phi = sdf_fn(edge_xy.reshape(-1, 2)).reshape(edge_xy.shape[0], 2)
    lengths = torch.linalg.norm(edge_xy[:, 1] - edge_xy[:, 0], dim=1)

    owner_phi = sdf_fn(sorted_points_t[owners])
    neighbor_phi = sdf_fn(sorted_points_t[neighbors])
    sign_violations = int((owner_phi * neighbor_phi > 0.0).sum().detach().cpu())
    return (
        midpoint_phi,
        endpoint_phi,
        lengths,
        edge_xy,
        torch.unique(owners),
        sign_violations,
    )


def inner_cell_cvt_loss(result, vertices, sorted_points_t, density_fn=None):
    """Density-weighted CVT energy over the computed inner cells."""
    owners, simplices = voroplex.cell_fan_simplices(
        result, device=sorted_points_t.device
    )
    if owners.numel() == 0:
        zero = sorted_points_t.new_tensor(0.0)
        return zero, float("nan"), float("nan"), 0

    a = sorted_points_t[owners]
    b = vertices[simplices[:, 0]]
    c = vertices[simplices[:, 1]]
    ab, ac = b - a, c - a
    areas = 0.5 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    bary = TRIANGLE_RULE.to(device=sorted_points_t.device, dtype=sorted_points_t.dtype)
    corners = torch.stack([a, b, c], dim=1)
    samples = torch.einsum("qk,tkd->tqd", bary, corners)
    rho = (
        torch.ones_like(samples[..., 0])
        if density_fn is None
        else density_fn(samples).clamp_min(EPS)
    )
    weights = areas[:, None] / 3.0 * rho
    triangle_energy = (
        weights * (samples - a[:, None]).square().sum(dim=-1)
    ).sum(dim=-1)
    triangle_moment = (weights[..., None] * samples).sum(dim=1)

    n_cells = result.n_cells
    energies = sorted_points_t.new_zeros(n_cells).index_add(0, owners, triangle_energy)
    masses = sorted_points_t.new_zeros(n_cells).index_add(0, owners, weights.sum(dim=-1))
    moments = sorted_points_t.new_zeros((n_cells, 2)).index_add(
        0, owners, triangle_moment
    )
    geometric_areas = sorted_points_t.new_zeros(n_cells).index_add(0, owners, areas)
    centroids = moments / masses.clamp_min(EPS)[:, None]
    residuals = torch.linalg.norm(sorted_points_t[:n_cells] - centroids, dim=1)
    return (
        energies.sum(),
        float(residuals.mean().detach().cpu()),
        float(geometric_areas.mean().detach().cpu()),
        n_cells,
    )


def cell_polygon_rows(result, cell_id):
    """Ordered flat per-cell vertex rows for one 2D cell."""
    first = result.cell_facet_offsets[cell_id]
    last = result.cell_facet_offsets[cell_id + 1]
    facet_ids = np.arange(first, last, dtype=np.int64)
    starts = result.facet_vertex_offsets[facet_ids]
    return result.cell_vertex_offsets[cell_id] + result.facet_vertices[starts]


def boundary_cell_quality_loss(result, vertices, boundary_cell_ids):
    """The area and circularity terms for the boundary cells.

    The area term is the mean of ((A_j - mean A) / mean A)^2, which is small when
    the boundary cells have similar areas. The circularity q_j = 4 pi A_j / P_j^2
    is 1 for a circle and smaller for other shapes, and the circularity term is
    the mean of (1 - q_j)^2.
    """
    boundary_ids = boundary_cell_ids.detach().cpu().tolist()
    areas = []
    circularities = []
    for cell_id in boundary_ids:
        rows = cell_polygon_rows(result, int(cell_id))
        if len(rows) < 3:
            continue
        polygon = vertices[
            torch.as_tensor(rows, dtype=torch.long, device=vertices.device)
        ]
        nxt = torch.roll(polygon, shifts=-1, dims=0)
        cross = polygon[:, 0] * nxt[:, 1] - nxt[:, 0] * polygon[:, 1]
        area = torch.abs(0.5 * cross.sum())
        perimeter = torch.linalg.norm(nxt - polygon, dim=1).sum()
        circularity = 4.0 * torch.pi * area / (perimeter.square() + EPS)
        areas.append(area)
        circularities.append(torch.clamp(circularity, min=0.0, max=1.0))

    if not areas:
        zero = vertices.new_tensor(0.0)
        return zero, zero, float("nan"), float("nan"), 0

    areas_t = torch.stack(areas)
    circularities_t = torch.stack(circularities)
    mean_area = areas_t.mean()
    area_loss = torch.mean(((areas_t - mean_area) / (mean_area + EPS)) ** 2)
    circularity_loss = torch.mean((1.0 - circularities_t) ** 2)
    area_cv = torch.sqrt(torch.var(areas_t, unbiased=False)) / (mean_area + EPS)
    return (
        area_loss,
        circularity_loss,
        float(area_cv.detach().cpu()),
        float(circularities_t.mean().detach().cpu()),
        len(areas),
    )


def compute_sdf_boundary_loss(
    points_t,
    limits,
    threads,
    sdf_fn,
    density_fn=None,
    loss_weights=None,
):
    """The objective L and its terms, for the current generator positions.

    The generators are ordered with those inside the shape first, and the cells of
    the inner generators are computed with the per-cell representation, so that
    their vertices are differentiable. The objective is

        L = lambda_mid L_mid + lambda_end L_end + lambda_in L_in
            + lambda_area L_area + lambda_circ L_circ,

    where L_mid and L_end are the sums of the squared SDF at the midpoints and
    ends of the interface edges, L_in is the density-weighted CVT energy of the
    inner cells, and L_area and L_circ are the boundary-cell terms. Returns the
    tessellation and a dictionary of the terms and diagnostics.
    """
    if loss_weights is None:
        loss_weights = make_loss_weights(DENSITY_MODE)

    order, n_inner, phi_points = sorted_by_sdf(points_t, sdf_fn)
    if order is None:
        zero = torch.sum(points_t * 0.0)
        return None, {
            "loss": zero,
            "midpoint_sdf_loss": points_t.new_tensor(0.0),
            "endpoint_sdf_loss": points_t.new_tensor(0.0),
            "inner_cvt_loss": points_t.new_tensor(0.0),
            "boundary_area_loss": points_t.new_tensor(0.0),
            "boundary_circularity_loss": points_t.new_tensor(0.0),
            "mean_abs_mid_phi": float("nan"),
            "mean_abs_endpoint_phi": float("nan"),
            "edge_length_cv": float("nan"),
            "boundary_area_cv": float("nan"),
            "boundary_mean_circularity": float("nan"),
            "num_boundary_cells": 0,
            "num_interface_edges": 0,
            "num_inner_cells": 0,
            "inner_cvt_residual": float("nan"),
            "inner_mean_area": float("nan"),
            "sign_violations": 0,
            "n_inner": n_inner,
            "phi_points": phi_points,
        }

    sorted_points_t = points_t[order]
    result = voroplex.compute_voronoi_2d(
        sorted_points_t.detach().cpu().numpy(),
        limits,
        n_compute=n_inner,
        n_inner=n_inner,
        threads=threads,
        representation="per_cell",
        properties=(),
    )
    vertices = voroplex.cell_vertices(result, sorted_points_t)
    (
        midpoint_phi,
        endpoint_phi,
        lengths,
        edge_xy,
        boundary_cell_ids,
        sign_violations,
    ) = interface_edge_terms(
        result,
        vertices,
        sorted_points_t,
        n_inner,
        sdf_fn,
    )
    inner_cvt_loss_value, inner_cvt_residual, inner_mean_area, num_inner_cells = (
        inner_cell_cvt_loss(result, vertices, sorted_points_t, density_fn=density_fn)
    )
    (
        boundary_area_loss,
        boundary_circularity_loss,
        boundary_area_cv,
        boundary_mean_circularity,
        num_boundary_cells,
    ) = boundary_cell_quality_loss(result, vertices, boundary_cell_ids)

    if len(midpoint_phi) == 0:
        midpoint_sdf_loss = points_t.new_tensor(0.0)
        endpoint_sdf_loss = points_t.new_tensor(0.0)
        mean_abs_mid_phi = float("nan")
        mean_abs_endpoint_phi = float("nan")
        edge_length_cv = float("nan")
    else:
        midpoint_sdf_loss = torch.sum(midpoint_phi**2)
        endpoint_sdf_loss = torch.sum(endpoint_phi**2)
        length_mean = lengths.mean()
        mean_abs_mid_phi = float(torch.mean(torch.abs(midpoint_phi)).detach().cpu().item())
        mean_abs_endpoint_phi = float(torch.mean(torch.abs(endpoint_phi)).detach().cpu().item())
        edge_length_cv = float(
            (torch.sqrt(torch.var(lengths, unbiased=False)) / (length_mean + EPS))
            .detach()
            .cpu()
            .item()
        )

    loss = (
        loss_weights["midpoint_sdf"] * midpoint_sdf_loss
        + loss_weights["endpoint_sdf"] * endpoint_sdf_loss
        + loss_weights["inner_cvt"] * inner_cvt_loss_value
        + loss_weights["boundary_area"] * boundary_area_loss
        + loss_weights["boundary_circularity"] * boundary_circularity_loss
    )

    return result, {
        "loss": loss,
        "midpoint_sdf_loss": midpoint_sdf_loss,
        "endpoint_sdf_loss": endpoint_sdf_loss,
        "inner_cvt_loss": inner_cvt_loss_value,
        "boundary_area_loss": boundary_area_loss,
        "boundary_circularity_loss": boundary_circularity_loss,
        "mean_abs_mid_phi": mean_abs_mid_phi,
        "mean_abs_endpoint_phi": mean_abs_endpoint_phi,
        "edge_length_cv": edge_length_cv,
        "boundary_area_cv": boundary_area_cv,
        "boundary_mean_circularity": boundary_mean_circularity,
        "num_boundary_cells": num_boundary_cells,
        "num_interface_edges": int(len(midpoint_phi)),
        "num_inner_cells": num_inner_cells,
        "inner_cvt_residual": inner_cvt_residual,
        "inner_mean_area": inner_mean_area,
        "sign_violations": sign_violations,
        "n_inner": n_inner,
        "order": order,
        "edge_xy": edge_xy.detach().cpu().numpy() if len(midpoint_phi) else np.empty((0, 2, 2)),
        "phi_points": phi_points,
    }


def snapshot_interface_edges(points, limits, threads, sdf_fn):
    """The interface edges for the points, as segments, for plotting."""
    points_t = torch.tensor(points, dtype=torch.float64)
    order, n_inner, _ = sorted_by_sdf(points_t, sdf_fn)
    if order is None:
        return np.empty((0, 2, 2))
    sorted_points_t = points_t[order]
    result = voroplex.compute_voronoi_2d(
        sorted_points_t.numpy(),
        limits,
        n_compute=n_inner,
        n_inner=n_inner,
        threads=threads,
        properties=(),
    )
    rows, _, _ = interface_edge_rows(result, n_inner)
    return result.vertices[rows] if len(rows) else np.empty((0, 2, 2))


def snapshot_cell_polygons(points, limits, threads, sdf_fn):
    """Every cell as a polygon, with its index and whether it is inner, for plotting."""
    points_t = torch.tensor(points, dtype=torch.float64)
    order, n_inner, _ = sorted_by_sdf(points_t, sdf_fn)
    if order is None:
        return []

    sorted_points_t = points_t[order]
    result = voroplex.compute_voronoi_2d(
        sorted_points_t.numpy(),
        limits,
        n_inner=n_inner,
        threads=threads,
        properties=(),
    )
    cell_polygons = []
    for cell_id in range(result.n_cells):
        rows = cell_polygon_rows(result, cell_id)
        if len(rows) < 3:
            continue
        xy = result.vertices[rows]
        cell_polygons.append((xy, int(cell_id), int(cell_id) < n_inner))
    return cell_polygons


def draw_sdf_contour(ax, sdf_fn, density_fn=None, resolution=220):
    xs = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    ys = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    values = sdf_fn(grid).detach().cpu().numpy()
    if density_fn is None:
        ax.contourf(xs.numpy(), ys.numpy(), values, levels=32, cmap="RdBu_r", alpha=0.35)
        contour_color = "black"
    else:
        density = density_fn(grid).detach().cpu().numpy()
        positive = np.maximum(density, EPS)
        vmin = float(positive.min())
        vmax = max(float(positive.max()), vmin * (1.0 + 1e-12))
        ax.imshow(
            positive,
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=0.9,
            zorder=0,
        )
        contour_color = "black"
    ax.contour(xs.numpy(), ys.numpy(), values, levels=[0.0], colors=contour_color, linewidths=1.5)


def draw_interface(ax, points, edge_xy, sdf_fn, title, cell_polygons=None, density_fn=None):
    draw_sdf_contour(ax, sdf_fn, density_fn=density_fn)
    if cell_polygons is not None:
        for polygon, _, is_inner in cell_polygons:
            closed = np.vstack([polygon, polygon[0]])
            color = "#3b6fb6" if is_inner else "#c74b4b"
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=color,
                linewidth=0.45,
                alpha=0.34,
                zorder=1,
            )
    phi = sdf_fn(torch.tensor(points, dtype=torch.float64)).detach().cpu().numpy()
    ax.scatter(points[phi <= 0, 0], points[phi <= 0, 1], s=6, color="#2454a6", zorder=2)
    ax.scatter(points[phi > 0, 0], points[phi > 0, 1], s=6, color="#d83b3b", zorder=2)
    for edge in edge_xy:
        ax.plot(edge[:, 0], edge[:, 1], color="#f2a900", linewidth=1.0, alpha=0.9, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def save_plot(
    initial_points,
    final_points,
    initial_edges,
    final_edges,
    initial_cells,
    final_cells,
    history,
    output_path,
    shape_name,
    sdf_fn,
    density_mode,
    density_fn,
    loss_weights,
):
    fig = plt.figure(figsize=(17.5, 8.8))
    grid = fig.add_gridspec(
        2,
        16,
        height_ratios=[1.0, 1.05],
        wspace=0.95,
        hspace=0.35,
    )
    ax_initial = fig.add_subplot(grid[0, 0:5])
    ax_final = fig.add_subplot(grid[0, 5:10])
    ax_total = fig.add_subplot(grid[0, 10:16])
    ax_raw = fig.add_subplot(grid[1, 0:7])
    ax_weighted = fig.add_subplot(grid[1, 9:16])

    draw_interface(
        ax_initial,
        initial_points,
        initial_edges,
        sdf_fn,
        "Initial cells",
        initial_cells,
        density_fn=density_fn,
    )
    draw_interface(
        ax_final,
        final_points,
        final_edges,
        sdf_fn,
        "Adaptive cells",
        final_cells,
        density_fn=density_fn,
    )

    iterations = np.arange(1, len(history["loss"]) + 1)
    loss_values = np.asarray(history["loss"], dtype=np.float64)
    positive = loss_values[loss_values > 0]
    ymin = max(positive.min() * 0.8, np.finfo(np.float64).tiny) if len(positive) else 1e-16
    ax_total.semilogy(iterations, loss_values, color="#2454a6", linewidth=2)
    ax_total.set_title("Total adaptive loss")
    ax_total.set_xlabel("Iteration")
    ax_total.set_ylabel("Loss")
    ax_total.set_ylim(bottom=ymin)
    ax_total.grid(True, alpha=0.25)

    component_specs = [
        ("midpoint SDF", "midpoint_sdf_loss", loss_weights["midpoint_sdf"]),
        ("endpoint SDF", "endpoint_sdf_loss", loss_weights["endpoint_sdf"]),
        ("inner CVT", "inner_cvt_loss", loss_weights["inner_cvt"]),
        ("boundary area", "boundary_area_loss", loss_weights["boundary_area"]),
        ("boundary circularity", "boundary_circularity_loss", loss_weights["boundary_circularity"]),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    raw_handles = []
    weighted_handles = []
    for i, (label, key, weight) in enumerate(component_specs):
        if weight == 0.0:
            continue
        raw_values = np.asarray(history[key], dtype=np.float64)
        color = colors[i % len(colors)]
        raw_line, = ax_raw.semilogy(
            iterations,
            raw_values,
            linestyle="-",
            color=color,
            linewidth=2,
            label=label,
        )
        weighted_line, = ax_weighted.semilogy(
            iterations,
            weight * raw_values,
            linestyle="-",
            color=color,
            linewidth=2,
            label=label,
        )
        raw_handles.append(raw_line)
        weighted_handles.append(weighted_line)

    ax_raw.set_title("Raw loss components")
    ax_raw.set_xlabel("Iteration")
    ax_raw.set_ylabel("Raw component loss")
    ax_raw.grid(True, alpha=0.25)
    ax_raw.legend(
        handles=raw_handles,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.82,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )

    ax_weighted.set_title("Weighted loss components")
    ax_weighted.set_xlabel("Iteration")
    ax_weighted.set_ylabel("Weighted contribution")
    ax_weighted.grid(True, alpha=0.25)
    ax_weighted.legend(
        handles=weighted_handles,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.82,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )

    fig.suptitle(f"2D SDF boundary meshing - {shape_name}, density={density_mode}", y=0.97)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.08)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gif(frame_history, output_path, shape_name, sdf_fn, density_mode, density_fn, fps):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    def update(frame_index):
        iteration, points, edge_xy, loss_value = frame_history[frame_index]
        ax.clear()
        title = f"2D SDF boundary - iter {iteration}"
        if loss_value is not None:
            title += f", loss {loss_value:.4g}"
        draw_interface(ax, points, edge_xy, sdf_fn, title, density_fn=density_fn)
        return ax.lines + ax.collections

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_history),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    fig.suptitle(f"SDF shape: {shape_name}, density={density_mode}", y=0.98)
    fig.tight_layout()
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def save_run_parameters(output_path, payload):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_interface_outputs(points, limits, threads, sdf_fn, output_prefix):
    """Write the final SDF interface mesh as VTU, with NPZ fallback."""
    points_t = torch.tensor(points, dtype=torch.float64)
    order, n_inner, _ = sorted_by_sdf(points_t, sdf_fn)
    if order is None:
        return None

    sorted_points_t = points_t[order]
    result = voroplex.compute_voronoi_2d(
        sorted_points_t.numpy(),
        limits,
        n_compute=n_inner,
        n_inner=n_inner,
        threads=threads,
        properties=(),
    )

    vtu_path = Path(f"{output_prefix}.vtu")
    try:
        voroplex.export_vtu(result, sorted_points_t.numpy(), str(vtu_path), facet_type="surface")
        return str(vtu_path)
    except ImportError as exc:
        npz_path = Path(f"{output_prefix}.npz")
        voroplex.export_npz(result, str(npz_path), points=sorted_points_t.numpy())
        print(f"VTU export skipped: {exc}")
        print(f"Saved mesh fallback: {npz_path}")
        return str(npz_path)


def main():
    threads = parse_int_option("threads", parse_int_arg(0, 1))
    n_points = parse_int_option("num-points", parse_int_arg(1, 1000))
    n_iter = parse_int_option("num-iters", parse_int_arg(2, 600))
    lr = parse_float_option("lr", parse_float_arg(3, 0.0005))
    shape_name = parse_option("shape", SDF_SHAPE)
    density_mode = parse_option("density-mode", DENSITY_MODE)
    density_mu_min = parse_float_option("density-mu-min", DENSITY_MU_MIN)
    density_mu_max = parse_float_option("density-mu-max", DENSITY_MU_MAX)
    density_boundary_width = parse_float_option("density-boundary-width", DENSITY_BOUNDARY_WIDTH)
    gif_frame_count = max(2, parse_int_option("gif-frames", GIF_FRAME_COUNT))
    gif_fps = max(1.0, parse_float_option("gif-fps", GIF_FPS))
    make_gif = "--no-gif" not in sys.argv
    make_vtu = "--no-vtu" not in sys.argv
    output_dir = parse_output_dir(os.environ.get("VOROPLEX_OUTPUT_DIR", "results"))
    if threads < 1:
        raise SystemExit("--threads must be positive")
    if n_points < 2:
        raise SystemExit("--num-points must be at least 2")
    if n_iter < 1:
        raise SystemExit("--num-iters must be positive")
    if lr <= 0.0:
        raise SystemExit("--lr must be positive")
    if density_mu_min <= 0.0 or density_mu_max <= 0.0:
        raise SystemExit("--density-mu-min and --density-mu-max must be positive")
    if density_mu_min > density_mu_max:
        raise SystemExit("--density-mu-min cannot exceed --density-mu-max")
    if density_boundary_width <= 0.0:
        raise SystemExit("--density-boundary-width must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_fn = get_sdf_function(shape_name)
    density_mode, density_fn = make_density_function(
        density_mode,
        sdf_fn,
        density_mu_min,
        density_mu_max,
        density_boundary_width,
    )
    loss_weights = make_loss_weights(density_mode)
    limits = [[0, 1], [0, 1]]

    rng = np.random.default_rng(13)
    initial_points = rng.random((n_points, 2)) * 0.9 + 0.05
    points_t = torch.tensor(initial_points, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([points_t], lr=lr)

    initial_edges = snapshot_interface_edges(initial_points, limits, threads, sdf_fn)
    initial_cells = snapshot_cell_polygons(initial_points, limits, threads, sdf_fn)
    frame_history = [(0, initial_points.copy(), initial_edges.copy(), None)]
    frame_stride = max(1, n_iter // (gif_frame_count - 1))
    history = {
        "loss": [],
        "midpoint_sdf_loss": [],
        "endpoint_sdf_loss": [],
        "inner_cvt_loss": [],
        "boundary_area_loss": [],
        "boundary_circularity_loss": [],
        "mean_abs_mid_phi": [],
        "mean_abs_endpoint_phi": [],
        "inner_cvt_residual": [],
        "edge_length_cv": [],
        "boundary_area_cv": [],
        "boundary_mean_circularity": [],
        "sign_violations": [],
    }

    print(f"2D SDF boundary meshing - {n_points} points, {n_iter} iterations, {threads} thread(s)")
    print(f"Shape: {shape_name}, lr={lr}")
    print(
        f"Density mode: {density_mode}"
        + (
            ""
            if density_fn is None
            else (
                f" (mu_min={density_mu_min}, mu_max={density_mu_max}, "
                f"boundary_width={density_boundary_width})"
            )
        )
    )
    print(
        "Loss weights: "
        f"midpoint={loss_weights['midpoint_sdf']}, "
        f"endpoint={loss_weights['endpoint_sdf']}, "
        f"inner_cvt={loss_weights['inner_cvt']}, "
        f"boundary_area={loss_weights['boundary_area']}, "
        f"boundary_circularity={loss_weights['boundary_circularity']}"
    )
    print(
        f"{'Iter':>4}  {'Loss':>12}  {'|phi(m)|':>10}  "
        f"{'|phi(v)|':>10}  {'CVT res':>10}  {'Edges':>6}"
    )
    print("-" * 70)

    t0 = time.perf_counter()
    for it in range(n_iter):
        _, terms = compute_sdf_boundary_loss(
            points_t,
            limits,
            threads,
            sdf_fn,
            density_fn=density_fn,
            loss_weights=loss_weights,
        )
        loss = terms["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            points_t.clamp_(1e-4, 1.0 - 1e-4)

        history["loss"].append(float(loss.item()))
        history["midpoint_sdf_loss"].append(float(terms["midpoint_sdf_loss"].item()))
        history["endpoint_sdf_loss"].append(float(terms["endpoint_sdf_loss"].item()))
        history["inner_cvt_loss"].append(float(terms["inner_cvt_loss"].item()))
        history["boundary_area_loss"].append(float(terms["boundary_area_loss"].item()))
        history["boundary_circularity_loss"].append(
            float(terms["boundary_circularity_loss"].item()))
        history["mean_abs_mid_phi"].append(terms["mean_abs_mid_phi"])
        history["mean_abs_endpoint_phi"].append(terms["mean_abs_endpoint_phi"])
        history["inner_cvt_residual"].append(terms["inner_cvt_residual"])
        history["edge_length_cv"].append(terms["edge_length_cv"])
        history["boundary_area_cv"].append(terms["boundary_area_cv"])
        history["boundary_mean_circularity"].append(terms["boundary_mean_circularity"])
        history["sign_violations"].append(terms["sign_violations"])
        if (it + 1) % frame_stride == 0 or it + 1 == n_iter:
            frame_points = points_t.detach().cpu().numpy().copy()
            frame_edges = snapshot_interface_edges(frame_points, limits, threads, sdf_fn)
            frame_history.append((it + 1, frame_points, frame_edges, float(loss.item())))

        if it == 0 or (it + 1) % 10 == 0 or it + 1 == n_iter:
            print(
                f"{it + 1:>4}  {loss.item():>12.6g}  "
                f"{terms['mean_abs_mid_phi']:>10.4g}  "
                f"{terms['mean_abs_endpoint_phi']:>10.4g}  "
                f"{terms['inner_cvt_residual']:>10.4g}  "
                f"{terms['num_interface_edges']:>6}"
            )

    elapsed = time.perf_counter() - t0
    final_points = points_t.detach().cpu().numpy()
    final_edges = snapshot_interface_edges(final_points, limits, threads, sdf_fn)
    final_cells = snapshot_cell_polygons(final_points, limits, threads, sdf_fn)
    output_path = output_dir / f"sdf_2d_boundary_meshing_{shape_name}.png"
    gif_path = output_dir / f"sdf_2d_boundary_meshing_{shape_name}.gif"
    params_path = output_dir / f"sdf_2d_boundary_meshing_{shape_name}_params.json"
    mesh_output = None
    if make_vtu:
        mesh_output = export_interface_outputs(
            final_points,
            limits,
            threads,
            sdf_fn,
            output_dir / f"sdf_2d_boundary_meshing_{shape_name}_final_interface",
        )
    save_plot(
        initial_points,
        final_points,
        initial_edges,
        final_edges,
        initial_cells,
        final_cells,
        history,
        output_path,
        shape_name,
        sdf_fn,
        density_mode,
        density_fn,
        loss_weights,
    )
    if make_gif:
        save_gif(frame_history, gif_path, shape_name, sdf_fn, density_mode, density_fn, gif_fps)
    save_run_parameters(
        params_path,
        {
            "script": Path(__file__).name,
            "argv": sys.argv[1:],
            "voroplex_version": voroplex.__version__,
            "seed": 13,
            "threads": threads,
            "num_points": n_points,
            "num_iters": n_iter,
            "learning_rate": lr,
            "shape": shape_name,
            "density_mode": density_mode,
            "density": {
                "mode": density_mode,
                "mu_min": density_mu_min,
                "mu_max": density_mu_max,
                "boundary_width": density_boundary_width,
                "rho_formula": "1 / mu^4" if density_fn is not None else "uniform",
            },
            "gif_frame_count": gif_frame_count,
            "gif_fps": gif_fps,
            "output_dir": str(output_dir),
            "limits": limits,
            "loss_weights": {
                "lambda_midpoint_sdf": loss_weights["midpoint_sdf"],
                "lambda_endpoint_sdf": loss_weights["endpoint_sdf"],
                "lambda_inner_cvt": loss_weights["inner_cvt"],
                "lambda_boundary_area": loss_weights["boundary_area"],
                "lambda_boundary_circularity": loss_weights["boundary_circularity"],
            },
            "initial_loss": history["loss"][0],
            "final_loss": history["loss"][-1],
            "loop_time_seconds": elapsed,
            "output_plot": str(output_path),
            "output_gif": str(gif_path) if make_gif else None,
            "output_mesh": mesh_output,
            "initial_points": initial_points.tolist(),
            "final_points": final_points.tolist(),
            "history": history,
        },
    )

    print("-" * 70)
    print(f"Initial loss: {history['loss'][0]:.8f}")
    print(f"Final loss:   {history['loss'][-1]:.8f}")
    print(f"Loop time:    {elapsed:.2f} s")
    print(f"Saved plot:   {output_path}")
    if make_gif:
        print(f"Saved GIF:    {gif_path}")
    if mesh_output:
        print(f"Saved mesh:   {mesh_output}")
    print(f"Saved params: {params_path}")


# Paper-only plotting support.  This block is intentionally additive: the
# original Example 15 optimization and plotting path above remains unchanged.
PAPER_FIGURE_BASENAME = "sdf_2d_boundary_meshing_flower_adaptive"
PAPER_GALLERY_BASENAME = "sdf_2d_boundary_meshing_adaptive_gallery_5x1"
PAPER_GALLERY_SHAPES = ("circle", "star", "rounded_square", "heart", "peanut")
PAPER_DPI = 450
PAPER_VIEW_LIMITS = ((0.10, 0.90), (0.10, 0.90))
PAPER_GALLERY_VIEW_LIMITS = {
    "heart": ((0.10, 0.90), (0.15, 0.95)),
}
PAPER_LOSS_COMPONENTS = (
    ("Midpoint SDF", "midpoint_sdf_loss", "lambda_midpoint_sdf", "#1f4f9a"),
    ("Endpoint SDF", "endpoint_sdf_loss", "lambda_endpoint_sdf", "#a83232"),
    ("Inner CVT", "inner_cvt_loss", "lambda_inner_cvt", "#2f7d46"),
    ("Boundary area", "boundary_area_loss", "lambda_boundary_area", "#8b5fbf"),
    (
        "Boundary circularity",
        "boundary_circularity_loss",
        "lambda_boundary_circularity",
        "#c77620",
    ),
)


def _configure_paper_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 14.0,
            "axes.titlesize": 15.5,
            "axes.labelsize": 15.5,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 12.2,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": PAPER_DPI,
        }
    )


def _paper_density_function(params, sdf_fn):
    density = params.get("density", {})
    density_mode = params.get("density_mode", density.get("mode", DENSITY_MODE))
    return make_density_function(
        density_mode,
        sdf_fn,
        float(density.get("mu_min", DENSITY_MU_MIN)),
        float(density.get("mu_max", DENSITY_MU_MAX)),
        float(density.get("boundary_width", DENSITY_BOUNDARY_WIDTH)),
    )


def _paper_sample_fields(sdf_fn, density_fn, resolution=280):
    xs = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    ys = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    sdf_values = sdf_fn(grid).detach().cpu().numpy()
    density_values = None
    if density_fn is not None:
        density_values = density_fn(grid).detach().cpu().numpy()
        density_values = np.maximum(density_values, EPS)
    return xs.numpy(), ys.numpy(), sdf_values, density_values


def _paper_draw_background(ax, sdf_fn, density_fn, display_mode, resolution=280):
    from matplotlib.colors import TwoSlopeNorm

    xs, ys, sdf_values, density_values = _paper_sample_fields(
        sdf_fn,
        density_fn,
        resolution=resolution,
    )
    if display_mode == "signed-sdf":
        image = ax.imshow(
            sdf_values,
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.2),
            alpha=0.9,
            zorder=0,
        )
    elif display_mode == "normalized-log-density":
        if density_values is None:
            raise ValueError("The paper density panel requires an adaptive density field")
        normalized_density = density_values / np.nanmax(density_values)
        image = ax.imshow(
            normalized_density,
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="viridis",
            norm=LogNorm(vmin=1.0e-2, vmax=1.0),
            alpha=0.9,
            zorder=0,
        )
    else:
        raise ValueError(f"Unknown paper background mode: {display_mode}")

    target_boundary = ax.contour(
        xs,
        ys,
        sdf_values,
        levels=[0.0],
        colors="#6a6a6a",
        linewidths=0.6,
        linestyles="--",
        zorder=30,
    )
    target_boundary.set_linestyle((0, (2.0, 1.35)))
    return image


def _paper_reconstruct(points, limits, threads, sdf_fn):
    edges = snapshot_interface_edges(points, limits, threads, sdf_fn)
    cells = snapshot_cell_polygons(points, limits, threads, sdf_fn)
    return edges, cells


def _paper_draw_interface(
    ax,
    points,
    edges,
    cells,
    sdf_fn,
    density_fn,
    display_mode,
    view_limits=PAPER_VIEW_LIMITS,
    point_size=1.0,
):
    image = _paper_draw_background(ax, sdf_fn, density_fn, display_mode)
    for polygon, _, is_inner in cells:
        if not is_inner:
            continue
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#ffffff",
            linewidth=0.34,
            alpha=1.0,
            zorder=3,
        )

    phi = sdf_fn(torch.tensor(points, dtype=torch.float64)).detach().cpu().numpy()
    ax.scatter(
        points[phi <= 0, 0],
        points[phi <= 0, 1],
        s=point_size,
        color="#3f97bf",
        zorder=2,
    )
    for edge in edges:
        ax.plot(
            edge[:, 0],
            edge[:, 1],
            color="#39d353",
            linewidth=1.35,
            alpha=0.97,
            solid_capstyle="round",
            zorder=4,
        )

    x_limits, y_limits = view_limits
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    return image


def _paper_add_sdf_colorbar(fig, image, colorbar_ax):
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([-0.2, 0.0, 0.2])
    colorbar.set_ticklabels([r"$-0.2$", r"$0$", r"$0.2$"])
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.set_label(r"SDF $\phi(\mathbf{x})$", fontsize=15.5, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=13.5, length=2.8, width=0.7, pad=1.0)
    colorbar.outline.set_linewidth(0.6)
    return colorbar


def _paper_add_density_colorbar(
    fig,
    image,
    colorbar_ax,
    label_position="left",
    label_rotation=90,
    label_pad=1.5,
    tick_position="left",
):
    from matplotlib.ticker import LogFormatterMathtext, NullLocator

    colorbar = fig.colorbar(
        image,
        cax=colorbar_ax,
        ticks=[1.0e-2, 1.0e-1, 1.0],
        format=LogFormatterMathtext(base=10),
    )
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.yaxis.set_label_position(label_position)
    colorbar.ax.yaxis.set_ticks_position(tick_position)
    colorbar.set_label(
        r"Normalized $\rho(\mathbf{x})$",
        fontsize=15.5,
        fontfamily="Times New Roman",
        rotation=label_rotation,
        labelpad=label_pad,
    )
    colorbar.ax.tick_params(labelsize=13.5, length=2.8, width=0.7, pad=1.0)
    colorbar.outline.set_linewidth(0.6)
    return colorbar


def _paper_match_left_colorbar_label_distance(reference, target, target_closer_by=0.0):
    fig = target.ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def label_to_bar_distance(colorbar):
        bar_left = colorbar.ax.get_window_extent(renderer).x0
        label_right = colorbar.ax.yaxis.label.get_window_extent(renderer).x1
        return bar_left - label_right

    difference_px = label_to_bar_distance(target) - label_to_bar_distance(reference)
    target.ax.yaxis.labelpad -= difference_px * 72.0 / fig.dpi + target_closer_by
    fig.canvas.draw()


def _paper_add_panel_label(fig, ax, label, y=-0.005):
    bbox = ax.get_position()
    fig.text(
        0.5 * (bbox.x0 + bbox.x1),
        y,
        label,
        fontsize=22.5,
        fontfamily="Times New Roman",
        fontweight="normal",
        ha="center",
        va="center",
        zorder=20,
    )


def _paper_positive_semilogy(ax, x_values, y_values, **kwargs):
    values = np.maximum(np.asarray(y_values, dtype=np.float64), np.finfo(np.float64).tiny)
    return ax.semilogy(x_values, values, **kwargs)


def _paper_load_shape_params(output_dir, shape_name):
    params_path = output_dir / f"sdf_2d_boundary_meshing_{shape_name}_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing {params_path}. Run Example 15 with --shape={shape_name} "
            f"and --output-dir={output_dir} first."
        )
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    required = ("initial_points", "final_points", "history", "loss_weights")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"{params_path} is missing required data: {', '.join(missing)}")
    return params_path, params


def save_paper_flower_figure(output_dir, threads=1, dpi=PAPER_DPI):
    """Draw the flower figure of the paper from a saved run.

    (a) The initial interface over the SDF, (b) the final interface over the
    normalized density, and (c) the weighted terms of the objective.
    """
    from matplotlib.ticker import LogFormatterMathtext, NullLocator

    _configure_paper_style()
    params_path, params = _paper_load_shape_params(output_dir, "flower")
    if params.get("shape") != "flower":
        raise ValueError(f"Expected a flower result in {params_path}")

    sdf_fn = get_sdf_function("flower")
    _, density_fn = _paper_density_function(params, sdf_fn)
    limits = params.get("limits", [[0, 1], [0, 1]])
    initial_points = np.asarray(params["initial_points"], dtype=np.float64)
    final_points = np.asarray(params["final_points"], dtype=np.float64)
    initial_edges, initial_cells = _paper_reconstruct(initial_points, limits, threads, sdf_fn)
    final_edges, final_cells = _paper_reconstruct(final_points, limits, threads, sdf_fn)

    fig = plt.figure(figsize=(10.0, 3.25), constrained_layout=False)
    panel_h = 0.923
    panel_w = panel_h * fig.get_figheight() / fig.get_figwidth()
    panel_bottom = 0.067
    loss_panel_h = 0.815
    loss_panel_w = loss_panel_h * fig.get_figheight() / fig.get_figwidth()
    loss_panel_bottom = 0.175
    left = 0.012
    cbar_gap = 0.006
    cbar_w = 0.009
    panel_gap = 0.064
    loss_gap = 0.074

    initial_cbar_left = left
    cax_initial = fig.add_axes([initial_cbar_left, panel_bottom, cbar_w, panel_h])
    initial_left = initial_cbar_left + cbar_w + cbar_gap
    ax_initial = fig.add_axes([initial_left, panel_bottom, panel_w, panel_h])
    final_cbar_left = initial_left + panel_w + panel_gap
    cax_final = fig.add_axes([final_cbar_left, panel_bottom, cbar_w, panel_h])
    final_left = final_cbar_left + cbar_w + cbar_gap
    ax_final = fig.add_axes([final_left, panel_bottom, panel_w, panel_h])
    loss_left = final_left + panel_w + loss_gap
    ax_loss = fig.add_axes([loss_left, loss_panel_bottom, loss_panel_w, loss_panel_h])
    ax_loss.set_box_aspect(1)

    initial_image = _paper_draw_interface(
        ax_initial,
        initial_points,
        initial_edges,
        initial_cells,
        sdf_fn,
        None,
        "signed-sdf",
    )
    sdf_colorbar = _paper_add_sdf_colorbar(fig, initial_image, cax_initial)
    final_image = _paper_draw_interface(
        ax_final,
        final_points,
        final_edges,
        final_cells,
        sdf_fn,
        density_fn,
        "normalized-log-density",
    )
    density_colorbar = _paper_add_density_colorbar(fig, final_image, cax_final)
    _paper_match_left_colorbar_label_distance(
        density_colorbar,
        sdf_colorbar,
        target_closer_by=4.0,
    )

    history = params["history"]
    loss_weights = params["loss_weights"]
    iterations = np.arange(1, len(history["loss"]) + 1)
    loss_handles = []
    total_line, = _paper_positive_semilogy(
        ax_loss,
        iterations,
        history["loss"],
        color="black",
        linewidth=2.7,
        linestyle=":",
        label="Total loss",
        zorder=5,
    )
    loss_handles.append(total_line)
    for label, history_key, weight_key, color in PAPER_LOSS_COMPONENTS:
        if history_key not in history:
            continue
        weight = float(loss_weights.get(weight_key, 0.0))
        if weight == 0.0:
            continue
        line, = _paper_positive_semilogy(
            ax_loss,
            iterations,
            weight * np.asarray(history[history_key], dtype=np.float64),
            color=color,
            linewidth=1.45,
            label=label,
        )
        loss_handles.append(line)

    ax_loss.set_xlabel("Iteration", labelpad=1.5)
    ax_loss.set_ylabel("Loss contribution", labelpad=0.5)
    ax_loss.set_xticks(np.arange(0, len(history["loss"]) + 1, 100))
    ax_loss.set_yticks(10.0 ** np.arange(-5, 0))
    ax_loss.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_loss.yaxis.set_minor_locator(NullLocator())
    ax_loss.grid(True, alpha=0.25, linewidth=0.7)
    ax_loss.legend(
        handles=loss_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.78,
        fontsize=10.2,
        handlelength=1.25,
        borderaxespad=0.0,
        borderpad=0.18,
        labelspacing=0.12,
    )

    for label, ax in zip(("(a)", "(b)", "(c)"), (ax_initial, ax_final, ax_loss)):
        _paper_add_panel_label(fig, ax, label)

    output_png = output_dir / f"{PAPER_FIGURE_BASENAME}.png"
    output_pdf = output_dir / f"{PAPER_FIGURE_BASENAME}.pdf"
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_png, output_pdf


def save_paper_gallery(output_dir, threads=1, dpi=PAPER_DPI):
    """Draw the paper's row of five shapes from five saved runs.

    Each run must have been made with the paper's settings, into output_dir.
    """
    _configure_paper_style()
    records = []
    for shape_name in PAPER_GALLERY_SHAPES:
        params_path, params = _paper_load_shape_params(output_dir, shape_name)
        expected = {
            "shape": shape_name,
            "density_mode": "sdf-distance",
            "num_points": 1000,
            "num_iters": 600,
        }
        for key, value in expected.items():
            if params.get(key) != value:
                raise ValueError(
                    f"Unexpected {key}={params.get(key)!r} in {params_path}; "
                    f"expected {value!r}"
                )
        sdf_fn = get_sdf_function(shape_name)
        _, density_fn = _paper_density_function(params, sdf_fn)
        limits = params.get("limits", [[0, 1], [0, 1]])
        final_points = np.asarray(params["final_points"], dtype=np.float64)
        final_edges, final_cells = _paper_reconstruct(final_points, limits, threads, sdf_fn)
        view_limits = PAPER_GALLERY_VIEW_LIMITS.get(shape_name, PAPER_VIEW_LIMITS)
        records.append(
            (final_points, final_edges, final_cells, sdf_fn, density_fn, view_limits)
        )

    fig = plt.figure(figsize=(11.0, 2.16), constrained_layout=False)
    panel_bottom = 0.015
    panel_h = 0.970
    panel_w = panel_h * fig.get_figheight() / fig.get_figwidth()
    panel_gap = 0.004
    left = 0.002
    cbar_gap = 0.008
    cbar_w = 0.010
    cbar_left = (
        left
        + len(records) * panel_w
        + (len(records) - 1) * panel_gap
        + cbar_gap
    )
    axes = [
        fig.add_axes([left + index * (panel_w + panel_gap), panel_bottom, panel_w, panel_h])
        for index in range(len(records))
    ]
    colorbar_ax = fig.add_axes([cbar_left, panel_bottom, cbar_w, panel_h])

    gallery_image = None
    for ax, record in zip(axes, records):
        points, edges, cells, sdf_fn, density_fn, view_limits = record
        gallery_image = _paper_draw_interface(
            ax,
            points,
            edges,
            cells,
            sdf_fn,
            density_fn,
            "normalized-log-density",
            view_limits=view_limits,
            point_size=2.0,
        )

    _paper_add_density_colorbar(
        fig,
        gallery_image,
        colorbar_ax,
        label_position="right",
        label_rotation=270,
        label_pad=16.0,
        tick_position="right",
    )
    output_png = output_dir / f"{PAPER_GALLERY_BASENAME}.png"
    output_pdf = output_dir / f"{PAPER_GALLERY_BASENAME}.pdf"
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output_png, output_pdf


_example15_original_main = main


def main():
    output_dir = parse_output_dir(os.environ.get("VOROPLEX_OUTPUT_DIR", "results"))
    paper_threads = parse_int_option("threads", parse_int_arg(0, 1))
    if "--paper-gallery" in sys.argv:
        output_png, output_pdf = save_paper_gallery(output_dir, threads=paper_threads)
        print(f"Saved paper gallery PNG: {output_png}")
        print(f"Saved paper gallery PDF: {output_pdf}")
        return

    _example15_original_main()
    if "--paper-figure" in sys.argv:
        output_png, output_pdf = save_paper_flower_figure(
            output_dir,
            threads=paper_threads,
        )
        print(f"Saved paper figure PNG: {output_png}")
        print(f"Saved paper figure PDF: {output_pdf}")


if __name__ == "__main__":
    main()
