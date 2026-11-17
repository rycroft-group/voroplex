"""
3D anisotropic foam design with VoroPlex and PyTorch
====================================================

3D counterpart of example 16, used for Sec. 5.3 of the paper. The two share
one objective: every cell's own fabric tensor is matched to a target evaluated
at that cell, so a uniform and a spatially varying orientation are the same
formulation with a different axis field.

In 3D the cell walls are polygonal faces rather than edges, the weight is face
area a_f rather than edge length, the cell measure is volume, and A is 3x3
with trace 1:

    A_c      = sum_f (a_f n_f n_f^T) / sum_f a_f
    A_tgt(c) = alpha u(c) u(c)^T + (1 - alpha)/2 (I - u(c) u(c)^T)
    L        = mean_c ||A_c - A_tgt(c)||_F^2
               + LAMBDA_VOLUME Var(V_c) / mean(V_c)^2

The tessellation uses the per-cell representation: cell_vertices gives each
cell's own vertices, differentiable with respect to the generators, and the
face areas and normals, the tensors, and the volumes are sums over the faces
and fan tetrahedra of every cell, with no loops in Python.

Run from the project root. The three foams of Section 5.3 of the VoroPlex
paper are

    python examples/17_3d_foam_orientation_optimization.py --target=uniaxial
    python examples/17_3d_foam_orientation_optimization.py --target=needle
    python examples/17_3d_foam_orientation_optimization.py --target=circular

The defaults are the paper's settings: 800 generators, 1000 iterations of Adam
with a learning rate of 0.003 that decays to 1/100 of that on a cosine
schedule, and alpha = 0.65 for the uniaxial (lamellar) and circular targets
and 0 for the needle (columnar) one. A run takes a few minutes. It uses one
thread by default, so that it is exactly reproducible: with several threads,
the gradient is summed in a different order, and the results differ slightly
from run to run.

At the start and at the end, the script prints the alignment M of the paper,
the mean of u^T A_c u, which approaches alpha, and, for the columnar target,
the median aspect ratio of the cross-sections of the cells.

Optional positional arguments:
    [num_threads] [num_points] [num_iters] [lr]

Flags:
    --target=<mode>           axis field u(c). "uniaxial" gives plate-like
                              cells stacked along x, "needle" with --alpha=0
                              gives columns along z, and "circular" gives
                              cells in curved slats around the vertical
                              centre line. Default uniaxial
    --alpha=<float>           override the mode's anisotropy strength
    --lambda-vol=<float>      weight of the cell-volume penalty, default 0.30
    --lr-decay={none|cosine}  anneal the step length, default cosine
    --lr-min-frac=<float>     final lr as a fraction of lr, default 0.01
    --out=<path>              summary figure; the training history is written
                              beside it as <name>_history.npz
    --vtu                     also write ParaView files

The figures of the paper are rendered from the saved history by
render_foam_pyvista.py and composed by make_foam_3d_grid.py.
"""

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import proj3d

import voroplex
from foam_figure_style import (
    apply_paper_style, draw_tensor_difference, draw_tensor_absolute,
    style_loss_axes, new_panel, save_panel, plot_loss_curve,
)


EPS = 1e-12

# Default uniaxial target tensor, diag(0.65, 0.175, 0.175).
TARGET_ALPHA_X = 0.65
TARGET_ALPHA_Y = 0.175
TARGET_ALPHA_Z = 0.175

# Planar target: walls avoid normals along z, with trace 1 and two equal
# large eigenvalues.
PLANAR_ALPHA_XY = 0.475      # A_xx = A_yy in the xy-plane preferred mode
PLANAR_ALPHA_Z  = 0.050      # A_zz — deficit axis
# Needle (columnar) target: strong preference AGAINST walls whose normal is
# along z, so cells extend into columns along z.
#
# A_zz cannot be driven arbitrarily close to zero at fixed N. A bounded convex
# cell must be capped by walls perpendicular to z, and for a hexagonal prism of
# side s and height h,
#       A_zz = 1 / (1 + 1.155 * h/s).
# With N cells of volume 1/N in a unit box, h is capped at the box height, so
# h/s <= sqrt(2.598 N) and
#       A_zz >= 1 / (1 + 1.155 * sqrt(2.598 N)).
# At N = 800 that floor is about 0.019 for ideal full-height prisms; real
# (irregular, boundary-affected) cells sit well above it.
#
# The paper targets A_zz = 0 (alpha = 0), which asks for prisms that span the
# whole height of the domain. It is not reached, but it gives the strongest
# columnar structure; --alpha= sets a weaker target.
NEEDLE_ALPHA_Z  = 0.0
NEEDLE_ALPHA_XY = (1.0 - NEEDLE_ALPHA_Z) / 2.0   # = 0.5

# ── Loss weight ────────────────────────────────────────────────────────────
# The loss has two terms,
#
#     L = mean_c ||A_c - A_tgt(c)||_F^2  +  LAMBDA_VOLUME Var(V_i) / mean(V_i)^2
#
# and LAMBDA_VOLUME is the weight of the second. Without it the first term is
# satisfied just as well by degenerate cells, which match the target tensor
# while making a useless foam. Overridable with --lambda-vol=.
LAMBDA_VOLUME = 0.30


# ── Face geometry from the per-cell representation ─────────────────────────


def facet_arrays(result):
    """Index arrays for the faces of every cell of a per-cell result.

    Each cell lists its own faces, so an interior face appears once in each of
    the two cells that share it. For every vertex reference of every face,
    `this` and `nxt` are the rows, in cell_vertices(result, ...), of the
    vertex and of the next vertex around the face, and `face` is the index of
    the face.
    """
    owners = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    offsets = np.asarray(result.facet_vertex_offsets)
    counts = np.diff(offsets)
    face = np.repeat(np.arange(len(counts)), counts)
    local = np.asarray(result.facet_vertices, dtype=np.int64)
    position = np.arange(len(local)) - offsets[:-1][face]
    base = np.asarray(result.cell_vertex_offsets)[owners[face]]
    this = base + local
    nxt = base + local[offsets[:-1][face] + (position + 1) % counts[face]]
    return {
        "owners": owners, "neighbors": np.asarray(result.facet_neighbors),
        "counts": counts, "face": face, "this": this, "nxt": nxt,
    }


def face_area_vectors(faces, vertices):
    """The area vector of every face: its area times its unit normal.

    Each face is split into triangles around the centroid of its vertices, and
    their area vectors are added. This is Newell's formula, and it does not
    depend on which vertex of the face is listed first.
    """
    device = vertices.device
    face = torch.as_tensor(faces["face"], device=device)
    this = vertices[torch.as_tensor(faces["this"], device=device)]
    nxt = vertices[torch.as_tensor(faces["nxt"], device=device)]
    n_faces = len(faces["counts"])
    counts = torch.as_tensor(faces["counts"], dtype=vertices.dtype, device=device)
    centers = vertices.new_zeros((n_faces, 3)).index_add(0, face, this)
    centers = centers / counts[:, None]
    cross = torch.linalg.cross(this - centers[face], nxt - centers[face], dim=1)
    return 0.5 * vertices.new_zeros((n_faces, 3)).index_add(0, face, cross)


# ── Per-cell targets ───────────────────────────────────────────────────────


def per_cell_targets_3d(points_t, target_mode, alpha, axis=None,
                        center=(0.5, 0.5)):
    """Target fabric tensor and axis of every cell, from one formula.

        A_tgt(c) = alpha * u(c) u(c)^T + (1-alpha)/2 * (I - u(c) u(c)^T)

    The preferred axis u(c) is the only thing that distinguishes the
    experiments: it is a fixed vector for the uniaxial, planar, and needle
    targets, and the local radial direction for the circular one. In the
    circular case, u(c) depends on the generator position, so gradients flow
    through it. The radial direction is undefined only exactly on the axis.
    """
    n = points_t.shape[0]
    device, dtype = points_t.device, points_t.dtype
    if target_mode == "circular":
        delta = points_t[:, :2] - points_t.new_tensor(center)
        r = torch.linalg.norm(delta, dim=1)
        u_xy = delta / r.clamp_min(1e-9).unsqueeze(1)
        u = torch.cat([u_xy, torch.zeros((n, 1), dtype=dtype, device=device)],
                      dim=1)
    else:
        axis_t = points_t.new_tensor(axis, dtype=dtype)
        axis_t = axis_t / torch.linalg.norm(axis_t).clamp_min(EPS)
        u = axis_t.unsqueeze(0).expand(n, 3)

    uu = u[:, :, None] * u[:, None, :]
    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    A_tgt = alpha * uu + ((1.0 - alpha) / 2.0) * (eye - uu)
    return A_tgt, u


def get_target_config(mode):
    """Return (A_target_np, u_target_np, label) for a named orientation mode.

    Modes (all trace = 1):
        uniaxial : diag(0.65, 0.175, 0.175)   -- walls perpendicular to x
        planar   : diag(0.475, 0.475, 0.05)   -- walls avoid z-normals
        needle   : diag(0.5, 0.5, 0)          -- columns along z
        circular : radial axis about the vertical centre line

    NOTE ON NAMING: high A_uu means most wall AREA has its normal along u,
    i.e. walls lie perpendicular to u and cells are FLATTENED along u. The
    structure is lamellar (a stack of plates normal to u), not columnar
    along u. Cells are elongated in the plane perpendicular to u.

    u_target_np is the axis used for the (n . u)^2 colouring:
        bright = walls whose normal is along u (the plate faces)
        dark   = walls whose normal is perpendicular to u (the plate rims)
    except for `planar` and `needle`, where u = z marks the deficit axis, so
    bright is rare.
    """
    if mode == "uniaxial":
        A_target = np.diag([TARGET_ALPHA_X, TARGET_ALPHA_Y, TARGET_ALPHA_Z])
        u_target = np.array([1.0, 0.0, 0.0])
        label = "uniaxial (x-axis)"
    elif mode == "planar":
        A_target = np.diag([PLANAR_ALPHA_XY, PLANAR_ALPHA_XY, PLANAR_ALPHA_Z])
        u_target = np.array([0.0, 0.0, 1.0])
        label = "planar (xy-plane; colouring axis = z)"
    elif mode == "needle":
        A_target = np.diag([NEEDLE_ALPHA_XY, NEEDLE_ALPHA_XY, NEEDLE_ALPHA_Z])
        u_target = np.array([0.0, 0.0, 1.0])
        label = "needle (columns along z; colouring axis = z, deficit)"
    elif mode == "circular":
        # Position-dependent target: at each point, the preferred wall-normal
        # direction is the RADIAL direction from the central z-axis. There is
        # no single A_target; an isotropic placeholder keeps the summary
        # figure's tensor panel meaningful.
        A_target = np.eye(3) / 3.0
        u_target = np.array([0.0, 0.0, 1.0])  # colour by z alignment for viz
        label = "circular field (radial-normal, tangential slats)"
    else:
        raise ValueError(
            f"Unknown target mode {mode!r}; use uniaxial|planar|needle|circular")
    return A_target, u_target, label


def get_per_cell_config(mode):
    """(alpha, axis) for the unified per-cell objective.

    Every mode uses the same target formula
        A_tgt(c) = alpha u(c) u(c)^T + (1-alpha)/2 (I - u(c) u(c)^T);
    they differ only in the preferred axis u(c). For `circular` the axis is the
    local radial direction and `axis` is unused.

    alpha > 1/3 concentrates wall normals along u, flattening cells across it
    (lamellar); alpha < 1/3 starves that direction of walls, extending cells
    along it (columnar).
    """
    if mode == "uniaxial":
        return TARGET_ALPHA_X, (1.0, 0.0, 0.0)
    if mode == "needle":
        return NEEDLE_ALPHA_Z, (0.0, 0.0, 1.0)
    if mode == "planar":
        return PLANAR_ALPHA_Z, (0.0, 0.0, 1.0)
    if mode == "circular":
        # Same anisotropy strength as the uniaxial case; only the axis varies
        # with position, which is the point the example is making.
        return TARGET_ALPHA_X, None
    raise ValueError(f"No per-cell config for target mode {mode!r}")


def get_slice_axis(mode):
    """Which axis to slice along for the summary-figure 2D slice panel.
    0 = x, 1 = y, 2 = z. For most modes we slice at z=0.5. For the needle
    mode we slice at x=0.5 (parallel to the needle axis) so the tall
    skinny rectangles are visible. For the circular mode we slice at z=0.5
    where the ring pattern is most visible."""
    if mode == "needle":
        return 0
    return 2


# ── Loss ───────────────────────────────────────────────────────────────────


def compute_loss_3d(points_t, limits, threads, target_mode="uniaxial",
                    per_cell_alpha=None, per_cell_axis=None):
    """L = mean_c ||A_c - A_tgt(c)||_F^2 + LAMBDA_VOLUME Var(V)/mean(V)^2.

    The fabric tensor of each cell uses only the faces that it shares with
    other cells: faces on the boundary of the box come from cutting the foam to
    the size of the domain, and are not part of its structure. The volumes
    are found from the tetrahedra that join each generator to its faces.
    """
    result = voroplex.compute_voronoi_3d(
        points_t.detach().cpu().numpy(), limits, representation="per_cell",
        threads=threads)
    vertices = voroplex.cell_vertices(result, points_t)
    n_cells = points_t.shape[0]
    dtype, device = points_t.dtype, points_t.device

    faces = facet_arrays(result)
    area_vec = face_area_vectors(faces, vertices)
    areas = torch.linalg.norm(area_vec, dim=1)
    normals = area_vec / areas.clamp_min(EPS)[:, None]
    inner = torch.as_tensor(faces["neighbors"] >= 0, device=device)
    weights = torch.where(inner, areas, torch.zeros_like(areas))
    weighted = weights[:, None, None] * normals[:, :, None] * normals[:, None, :]

    owners = torch.as_tensor(faces["owners"], device=device)
    num = torch.zeros((n_cells, 3, 3), dtype=dtype, device=device).index_add(
        0, owners, weighted)
    den = torch.zeros(n_cells, dtype=dtype, device=device).index_add(
        0, owners, weights)
    valid = den > EPS
    A_cells = num / den.clamp_min(EPS)[:, None, None]
    # The area-weighted tensor of the whole foam. Each interior face is
    # counted once in each of its two cells, which does not change it.
    A = weighted.sum(dim=0) / weights.sum().clamp_min(EPS)

    A_tgt, u = per_cell_targets_3d(points_t, target_mode, per_cell_alpha,
                                   axis=per_cell_axis)
    resid = ((A_cells - A_tgt) ** 2).sum(dim=(1, 2))
    L_main = resid[valid].mean() if valid.any() else points_t.new_tensor(0.0)

    tet_owners, tets = voroplex.cell_fan_simplices(result, device=device)
    q = [vertices[tets[:, i]] - points_t[tet_owners] for i in range(3)]
    tet_volume = (q[0] * torch.linalg.cross(q[1], q[2], dim=1)).sum(dim=1) / 6.0
    volumes = torch.zeros(n_cells, dtype=dtype, device=device).index_add(
        0, tet_owners, tet_volume)

    mean_V = volumes.mean()
    L_volume = torch.var(volumes, unbiased=False) / (mean_V ** 2 + EPS)
    total = L_main + LAMBDA_VOLUME * L_volume

    return total, {
        "L_main": L_main, "L_volume": L_volume,
        "A": A, "A_cells": A_cells, "u": u, "valid": valid,
        "volumes": volumes, "result": result,
        "n_cells": n_cells, "n_inner_faces": int(inner.sum()) // 2,
    }


def foam_metrics(terms, alpha, axis_for_aspect=None):
    """The alignment M of the paper and related measures of a foam.

    M is the mean of (e_c . u(c))^2, where e_c is the eigenvector of A_c with
    the largest eigenvalue when alpha > 1/3, and with the smallest when
    alpha < 1/3. It is 1 when every cell is aligned with its own target axis,
    and 1/3 on average for axes unrelated to u. The strength, the mean of
    u(c)^T A_c u(c), is the fraction of each cell's wall area that faces
    along u(c), and approaches alpha.

    If axis_for_aspect is given, the aspect ratio of the cross-section of each
    cell perpendicular to that axis is also found, as the square root of the
    ratio of the eigenvalues of the covariance of the cell's vertices,
    projected onto the plane, and its median is returned.
    """
    valid = terms["valid"]
    A_cells = terms["A_cells"].detach()[valid]
    u = terms["u"].detach()[valid]
    _, vecs = torch.linalg.eigh(A_cells)
    e = vecs[:, :, -1] if alpha > 1.0 / 3.0 else vecs[:, :, 0]
    metrics = {
        "M": float((((e * u).sum(dim=1)) ** 2).mean()),
        "strength": float(torch.einsum("ci,cij,cj->c", u, A_cells, u).mean()),
        "mean_A_cells": A_cells.mean(dim=0).cpu().numpy(),
    }
    if axis_for_aspect is not None:
        result = terms["result"]
        keep = [i for i in range(3) if i != axis_for_aspect]
        offsets = result.cell_vertex_offsets
        ratios = []
        for c in range(len(result)):
            xy = result.vertices[offsets[c]:offsets[c + 1]][:, keep]
            ev = np.linalg.eigvalsh(np.cov(xy.T))
            ratios.append(np.sqrt(ev[1] / ev[0]))
        metrics["median_aspect"] = float(np.median(ratios))
    return metrics


# ── Visualization helpers ──────────────────────────────────────────────────


def _slice_plane_axes(axis):
    """The two 3D axis indices to project onto when slicing along `axis`.
    axis=0 (x-slice)  → project onto (y, z)
    axis=1 (y-slice)  → project onto (x, z)
    axis=2 (z-slice)  → project onto (x, y)"""
    if axis == 0:
        return 1, 2
    if axis == 1:
        return 0, 2
    return 0, 1


def slice_face_at_z(face_vertices_np, z_value, eps=1e-9, axis=2):
    """Compute the intersection segment where a planar 3D polygon crosses
    the plane <axis> = z_value.  axis: 0=x, 1=y, 2=z (default 2 for
    backward compatibility with the original z-only signature).

    Convex planar faces cross a plane in either 0 or 2 points (modulo
    degenerate cases). Returns either a (2, 3) ndarray of those two
    intersection points, or None if the face doesn't cross.
    """
    n = len(face_vertices_np)
    intersections = []
    for i in range(n):
        v0 = face_vertices_np[i]
        v1 = face_vertices_np[(i + 1) % n]
        s0 = v0[axis] - z_value
        s1 = v1[axis] - z_value
        # Strict crossing of the edge
        if (s0 > eps and s1 < -eps) or (s0 < -eps and s1 > eps):
            t = -s0 / (s1 - s0)
            intersections.append(v0 + t * (v1 - v0))
        elif abs(s0) <= eps and abs(s1) > eps:
            # Vertex sits exactly on the plane (rare)
            intersections.append(v0.copy())
    if len(intersections) < 2:
        return None
    return np.array(intersections[:2])


def foam_faces(result, include_boundary=False):
    """Each interior face of the foam once, as an array of its vertices.

    `result` is a geometry or per-cell result. An interior face is listed by
    both of its cells, and is taken from the one with the lower index. With
    include_boundary, the faces on the boundary of the box are also returned.
    Returns (polygons, owners, neighbors).
    """
    faces = facet_arrays(result)
    owners, neighbors = faces["owners"], faces["neighbors"]
    keep = neighbors > owners
    if include_boundary:
        keep |= neighbors < 0
    offsets = result.facet_vertex_offsets
    polygons = []
    for f in np.flatnonzero(keep):
        base = result.cell_vertex_offsets[owners[f]]
        ids = base + result.facet_vertices[offsets[f]:offsets[f + 1]]
        polygons.append(np.asarray(result.vertices[ids], dtype=np.float64))
    return polygons, owners[keep], neighbors[keep]


def polygon_area_normal(polygon):
    """Area and unit normal of a planar polygon, by Newell's formula."""
    center = polygon.mean(axis=0)
    area_vec = 0.5 * np.cross(polygon - center,
                              np.roll(polygon, -1, axis=0) - center).sum(axis=0)
    area = np.linalg.norm(area_vec)
    return area, area_vec / max(area, EPS)


def compute_z_slice_segments(result, points_np, z_value, u_target_np, axis=2):
    """For each interior Voronoi face, compute its intersection with the plane
    <axis> = z_value. Returns (segments_2d, alignment_colors) with one entry
    per crossing face; segments_2d are projected onto the two axes
    perpendicular to `axis` (see _slice_plane_axes).

    alignment_colors: ndarray of (n_segments,) values in [0, 1] equal to
                      (face_normal . u_target)^2, used to color each segment.
    """
    a0, a1 = _slice_plane_axes(axis)
    polygons, _, _ = foam_faces(result)
    segments = []
    colors = []
    for polygon in polygons:
        slice_pts = slice_face_at_z(polygon, z_value, axis=axis)
        if slice_pts is None:
            continue
        _, normal = polygon_area_normal(polygon)
        segments.append(slice_pts[:, [a0, a1]])
        colors.append(float((normal @ u_target_np) ** 2))
    return segments, np.asarray(colors)


def draw_z_slice(ax, segments, colors, z_value, points_np, title,
                 slice_band=0.08, axis=2):
    """Render a slice (default z=z_value) as coloured line segments plus
    generators near the slicing plane. axis picks which coordinate is held
    fixed; the two orthogonal coordinates are the 2D plot axes."""
    a0, a1 = _slice_plane_axes(axis)
    for seg, c in zip(segments, colors):
        ax.plot(seg[:, 0], seg[:, 1],
                color=plt.cm.viridis(float(c)),
                linewidth=0.9, alpha=0.95, zorder=1)
    if points_np is not None and len(points_np):
        dist_to_plane = np.abs(points_np[:, axis] - z_value)
        mask = dist_to_plane <= slice_band
        if mask.any():
            alpha = np.clip(1.0 - dist_to_plane[mask] / slice_band, 0.3, 1.0)
            ax.scatter(points_np[mask, a0], points_np[mask, a1],
                       s=14, color="#d83b3b", edgecolor="white",
                       linewidth=0.3, alpha=alpha, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    axis_labels = ["x", "y", "z"]
    ax.set_xlabel(axis_labels[a0]); ax.set_ylabel(axis_labels[a1])


# Clearance, in points, from an axis to its name in the 3D panels.
AXIS_NAME_PAD = 16.0


def _place_axis_names(ax, pad_pt=AXIS_NAME_PAD):
    """Put x/y/z halfway along their own axis, between that axis's 0 and 1.

    matplotlib pushes a 3D axis label toward one end of the axis, so the name
    reads as though it belonged to the nearer tick. Projecting the axis's two
    endpoints to the screen, taking their midpoint and stepping perpendicular
    to the axis -- away from the centre of the cube -- puts each name exactly
    midway between its 0 and its 1, at the same clearance on every axis.
    """
    fig = ax.figure
    fig.canvas.draw()
    proj = ax.get_proj()
    size = ax.xaxis.label.get_fontsize()
    pad = pad_pt * fig.dpi / 72.0
    to_px = ax.transData.transform
    to_proj = ax.transData.inverted().transform

    def px(point):
        return to_px(proj3d.proj_transform(*point, proj)[:2])

    cx, cy = px((0.5, 0.5, 0.5))
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    edges = (("x", (0, 0, 0), (1, 0, 0)),
             ("y", (1, 0, 0), (1, 1, 0)),
             ("z", (1, 1, 0), (1, 1, 1)))
    for name, p0, p1 in edges:
        (x0, y0), (x1, y1) = px(p0), px(p1)
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        ux, uy = -dy / norm, dx / norm           # perpendicular to the axis
        if (mx - cx) * ux + (my - cy) * uy < 0:  # point it away from the cube
            ux, uy = -ux, -uy
        ax.text2D(*to_proj((mx + ux * pad, my + uy * pad)), name,
                  transform=ax.transData, fontsize=size,
                  ha="center", va="center")


def _draw_3d_scatter(ax, points_np, label_axes=True):
    """Scatter the generator positions in the unit cube.

    No title is drawn: panels are identified by a bold "(a)"-style label and
    described in the LaTeX caption. When `label_axes` is False the x/y/z axis
    names are suppressed, which is used for the second of a side-by-side pair
    that shares the first panel's axes.
    """
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
               s=12, color="#d83b3b", edgecolor="white", linewidth=0.25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    if label_axes:
        _place_axis_names(ax)
    # Only the endpoints are ticked: a mid tick collides with the axis name on
    # a projected 3D axis at this figure scale.
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.tick_params(labelsize=14, pad=1)


def _log_ymin(values):
    arr = np.asarray(values, dtype=np.float64)
    pos = arr[arr > 0]
    return max(pos.min() * 0.8, np.finfo(np.float64).tiny) if len(pos) else 1e-16


def save_summary_figure_3d(initial_points, final_points, history,
                           output_path, u_target_np, limits, threads,
                           slice_axis=2):
    """Four-panel summary figure for 3D foam orientation, laid out 1 row x 4.

    Panels, in reading order:
        (a) initial generator positions
        (b) optimized generator positions
        (c) slice through the optimized foam, coloured by (n_f . u)^2
        (d) total and main loss versus iteration

    The objective matches every cell's own fabric tensor to a target evaluated
    at that cell, so the foam has no single mean fabric tensor to report, and
    the two panels that showed one have been dropped.

    No titles are drawn inside the axes. Each panel is lettered beneath its
    centre, the way a LaTeX subfigure grid would letter it, so that this one
    image can be included at \\textwidth in place of four separate panels.
    """
    apply_paper_style()

    fig = plt.figure(figsize=(18.0, 4.8))
    # Six columns for four panels: columns 2 and 4 are empty spacers. The
    # slice needs room for its colour bar and the loss panel for its y axis,
    # while the two point clouds need almost nothing between them. gridspec
    # gives every gap the same width, so without the spacers the gap (c) and
    # (d) require would also be forced between (a) and (b), shrinking the
    # point clouds for no reason.
    grid = fig.add_gridspec(1, 6, wspace=SUMMARY_WSPACE,
                            width_ratios=SUMMARY_WIDTH_RATIOS)

    ax_init = fig.add_subplot(grid[0, 0], projection="3d")
    ax_final3d = fig.add_subplot(grid[0, 1], projection="3d")
    ax_slice = fig.add_subplot(grid[0, 3])
    ax_loss = fig.add_subplot(grid[0, 5])

    # Both 3D panels carry their own x/y/z names. They are far enough apart
    # to read as separate plots rather than a shared-axis pair, and the
    # caption does not say (b) borrows (a)'s axes.
    _draw_3d_scatter(ax_init, initial_points, label_axes=True)
    _draw_3d_scatter(ax_final3d, final_points, label_axes=True)

    # Slice through the optimized foam, sitting next to the 3D point views so
    # the geometry panels read together.
    result = voroplex.compute_voronoi_3d(final_points, limits, threads=threads)
    segments, colors = compute_z_slice_segments(result, final_points,
                                                z_value=0.5,
                                                u_target_np=u_target_np,
                                                axis=slice_axis)
    draw_z_slice(ax_slice, segments, colors, z_value=0.5,
                 points_np=final_points, title=None, axis=slice_axis)

    iters = np.arange(1, len(history["loss"]) + 1)
    plot_loss_curve(ax_loss, iters, history["loss"], "#2454a6",
                    r"$\mathcal{L}$")
    plot_loss_curve(ax_loss, iters, history["L_main"], "#a64124",
                    r"$\mathcal{L}_{\mathrm{main}}$")
    style_loss_axes(ax_loss, history["loss"], history["L_main"])
    ax_loss.legend()
    ax_loss.set_ylabel(r"$\mathcal{L}$")

    # The slice panel is aspect="equal", so the first draw shrinks it to a
    # square inside its gridspec cell, while the loss panel keeps the whole
    # cell. Left alone, (d) comes out half again as tall as (c) and straddles
    # it, with its top and bottom edges outside (c)'s. Giving the loss panel
    # the slice panel's vertical box lines the two up as one row.
    fig.canvas.draw()
    box_slice = ax_slice.get_position()
    box_loss = ax_loss.get_position()
    ax_loss.set_position([box_loss.x0, box_slice.y0,
                          box_loss.width, box_slice.height])

    # Colour bar for the slice, on the same 0..1 viridis scale as the
    # composed 2D row, so the two figures carry the same bar and label.
    box_slice = ax_slice.get_position()
    cax = fig.add_axes([box_slice.x1 + SLICE_CBAR_PAD, box_slice.y0,
                        SLICE_CBAR_WIDTH, box_slice.height])
    mappable = plt.cm.ScalarMappable(cmap="viridis",
                                     norm=plt.Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$(n_f \cdot u)^2$", labelpad=SLICE_CBAR_LABEL_PAD)
    cbar.set_ticks([0.0, 0.5, 1.0])

    # ---- (a)..(d) beneath the middle of each panel -----------------------
    # Drawn at the same size as the letters under the composed 2D row and the
    # schematic, so all three figures letter their panels identically. The
    # baseline is the lowest point the row reaches -- an x-axis label, or the
    # foot of the colour bar.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def extent(group):
        """(x0, y0, x1, y1) of a panel as drawn, in figure coordinates."""
        pts = [inv.transform(a.get_tightbbox(renderer).get_points())
               for a in group]
        return (min(p[0][0] for p in pts), min(p[0][1] for p in pts),
                max(p[1][0] for p in pts), max(p[1][1] for p in pts))

    # Each letter is centred on its panel's PLOT RECTANGLE, not on the tight
    # bounding box. The tight box includes the y-axis label and the colour
    # bar, which would pull the letter off the graph it names.
    row = [[ax_init], [ax_final3d], [ax_slice, cax], [ax_loss]]
    letters = iter("abcd")
    y_bottom = min(extent(g)[1] for g in row)
    for group in row:
        box = group[0].get_position()
        fig.text(0.5 * (box.x0 + box.x1), y_bottom - SUMMARY_LABEL_GAP,
                 f"({next(letters)})", fontsize=SUMMARY_LABEL_FONT,
                 ha="center", va="top")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Common width/height ratio for every 3D panel, so that each row of
# three renders at uniform height once LaTeX scales them to a common
# width.
PANEL_ASPECT = 1.09

# Vertical gaps in the summary figure, in figure fractions: between the bottom
# of a row and the top of its (a)..(f) letters, and between those letters and
# the top of the row below.
SUMMARY_LABEL_GAP = 0.012

# Horizontal layout of the summary row. Columns 2 and 4 of the gridspec
# are empty spacers, so the width (c) and (d) need for a colour bar and a
# y axis is not also forced between the two point clouds.
SUMMARY_WSPACE = 0.40
SUMMARY_WIDTH_RATIOS = (1.4, 1.4, 0.20, 1.0, 0.55, 1.0)
SLICE_CBAR_WIDTH = 0.010
SLICE_CBAR_PAD = 0.009
SLICE_CBAR_LABEL_PAD = 3.0
SUMMARY_ROW_GAP = 0.022

# Letter size for the summary figure's (a)..(f). Smaller than the 2D row's
# FONT_ROW_LABEL: that row is ~17 in wide in the source and so is scaled to
# about 0.39 when included at \textwidth, whereas this figure is ~11.6 in wide
# and scaled to about 0.56. The same source size would print these letters
# half again as large as those.
SUMMARY_LABEL_FONT = 24


def save_panels_3d(out_dir, initial_points, final_points, history,
                   A_target_np, A_final_np, u_target_np, limits, threads,
                   slice_axis=2, tensor_panel="difference"):
    """Write each panel of the 3D summary as its own image file.

    The paper composes these with LaTeX subfigures, which add the (a)/(b)/(c)
    labels, so no letters or titles are drawn here. Files written:

        a_initial_generators.png
        b_optimized_generators.png
        c_slice.png
        d_tensor.png
        e_loss.png
        f_diagonals.png
    """
    written = []

    # (a) and (b): generator positions before and after. As standalone figures
    # they no longer share axes, so both carry full axis labels.
    for name, pts in (("a_initial_generators", initial_points),
                      ("b_optimized_generators", final_points)):
        fig, ax = new_panel(figsize=(4.4, 4.0), projection="3d")
        _draw_3d_scatter(ax, pts, label_axes=True)
        written.append(save_panel(fig, out_dir, name, pad_aspect=PANEL_ASPECT))

    # (c) Slice through the optimized foam.
    result = voroplex.compute_voronoi_3d(final_points, limits, threads=threads)
    segments, colors = compute_z_slice_segments(result, final_points,
                                                z_value=0.5,
                                                u_target_np=u_target_np,
                                                axis=slice_axis)
    fig, ax = new_panel(figsize=(4.2, 4.2))
    draw_z_slice(ax, segments, colors, z_value=0.5, points_np=final_points,
                 title=None, axis=slice_axis)
    written.append(save_panel(fig, out_dir, "c_slice", pad_aspect=PANEL_ASPECT))

    # (d) Tensor panel: signed mismatch, or the achieved tensor on its own when
    # the target is spatially varying and has no global matrix form.
    fig, ax = new_panel(figsize=(4.3, 4.3))
    if tensor_panel == "absolute":
        draw_tensor_absolute(ax, A_final_np, axis_labels=["x", "y", "z"],
                             cbar=True, fig=fig, cell_fontsize=15)
    else:
        draw_tensor_difference(ax, A_target_np, A_final_np,
                               axis_labels=["x", "y", "z"],
                               cbar=True, fig=fig, cell_fontsize=15)
    written.append(save_panel(fig, out_dir, "d_tensor", pad_aspect=PANEL_ASPECT))

    # (e) Loss curves.
    iters = np.arange(1, len(history["loss"]) + 1)
    fig, ax = new_panel()
    plot_loss_curve(ax, iters, history["loss"], "#2454a6",
                    r"$\mathcal{L}$")
    plot_loss_curve(ax, iters, history["L_main"], "#a64124",
                    r"$\mathcal{L}_{\mathrm{main}}$")
    style_loss_axes(ax, history["loss"], history["L_main"])
    ax.legend()
    written.append(save_panel(fig, out_dir, "e_loss", pad_aspect=PANEL_ASPECT))

    # (f) Diagonal entries of A against their targets.
    fig, ax = new_panel()
    # For a uniaxial target the two minor diagonal entries converge to nearly
    # the same value (0.198 vs 0.194 in the lamellar run), so equal-width
    # curves would hide one beneath the other. Drawing them at staggered
    # widths, widest first, keeps every curve visible where they coincide.
    # Widest curve is drawn first so the narrower ones stay visible on top of
    # it; the legend is then reordered to read A_xx, A_yy, A_zz.
    h_yy, = ax.plot(iters, history["A_yy"], color="#2454a6", linewidth=4.6,
                    solid_capstyle="round", zorder=2)
    h_zz, = ax.plot(iters, history["A_zz"], color="#007f5f", linewidth=2.0,
                    zorder=3)
    h_xx, = ax.plot(iters, history["A_xx"], color="#d83b3b", linewidth=2.6,
                    zorder=4)
    for value, colour in zip(A_target_np.diagonal(),
                             ("#d83b3b", "#2454a6", "#007f5f")):
        ax.axhline(value, color=colour, linestyle="--", linewidth=1.4,
                   alpha=0.65)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$A_{ii}$")
    ax.set_ylim(0, 1)
    ax.legend([h_xx, h_yy, h_zz],
              [r"$A_{xx}$", r"$A_{yy}$", r"$A_{zz}$"],
              ncol=3, columnspacing=0.8, handlelength=1.1,
              loc="upper center", fontsize=14)
    ax.grid(True, alpha=0.25)
    written.append(save_panel(fig, out_dir, "f_diagonals", pad_aspect=PANEL_ASPECT))

    print(f"  Wrote {len(written)} panels to {out_dir}/")
    return written


# ── VTU export for ParaView ────────────────────────────────────────────────


def export_foam_vtu(result, points_np, filename, u_target_np):
    """Export the foam to a VTU file for ParaView.

    Includes BOTH inner faces (cell-to-cell walls) and surface faces
    (cell-to-boundary). Each face becomes a polygon cell with attributes:

        OwnerCell         (int)    cell ID that owns this face
        NeighborCell      (int)    other cell ID for inner faces;
                                   boundary code (< 0) for surface faces
                                   (-1/-2 = xmin/xmax, -3/-4 = ymin/ymax,
                                    -5/-6 = zmin/zmax)
        IsSurface         (int)    0 = inner face, 1 = surface face
        FaceArea          (float)  polygon area
        Nx, Ny, Nz        (float)  face's unit normal
        AlignmentSquared  (float)  (n . u_target)^2 in [0, 1]
        RadialAlignment   (float)  (n . r_hat)^2, with r_hat the horizontal
                                   direction away from the centre line

    Including surface faces lets ParaView render the cube's outer skin,
    producing a clean bounding-box look. Use a Threshold filter on
    IsSurface to hide/show either subset. The copies of a vertex in
    different cells are merged, so that the faces share their vertices.
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        raise ImportError(
            "VTU export needs vtk. Install with: pip install vtk"
        )

    polygons, owners, neighbors = foam_faces(result, include_boundary=True)
    n_faces = len(polygons)
    is_surface_flag = (neighbors < 0).astype(np.int32)

    # Merge the copies of each vertex, which agree to within rounding error.
    all_vertices = np.concatenate(polygons)
    keys = np.round(all_vertices / 1e-10).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    inverse = inverse.ravel()
    vertex_xyz = all_vertices[first]
    starts = np.concatenate([[0], np.cumsum([len(p) for p in polygons])])

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(vertex_xyz, dtype=np.float64), deep=True))
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    areas = np.zeros(n_faces, dtype=np.float64)
    normals = np.zeros((n_faces, 3), dtype=np.float64)
    face_centroids_np = np.zeros((n_faces, 3), dtype=np.float64)
    for i, polygon in enumerate(polygons):
        areas[i], normals[i] = polygon_area_normal(polygon)
        face_centroids_np[i] = polygon.mean(axis=0)
        ids = inverse[starts[i]:starts[i + 1]]
        vtk_polygon = vtk.vtkPolygon()
        vtk_polygon.GetPointIds().SetNumberOfIds(len(ids))
        for j, vid in enumerate(ids):
            vtk_polygon.GetPointIds().SetId(j, int(vid))
        grid.InsertNextCell(vtk_polygon.GetCellType(), vtk_polygon.GetPointIds())

    alignments = (normals @ u_target_np) ** 2  # (n.u)^2 in [0, 1]

    # For circular-field visualization: also compute per-face RADIAL alignment
    # (n.r_hat)^2 where r_hat is the local radial direction from the vertical
    # central axis at (0.5, 0.5). Bright = walls whose normal is along the
    # radial direction (the walls that CIRCULAR foam optimization prefers).
    xy_delta = face_centroids_np[:, :2] - np.array([0.5, 0.5])
    r_safe = np.maximum(np.linalg.norm(xy_delta, axis=1), 1e-6)
    r_hat = np.zeros_like(face_centroids_np)
    r_hat[:, :2] = xy_delta / r_safe[:, None]
    radial_alignment = (normals * r_hat).sum(axis=1) ** 2

    arrays_to_add = [
        ("OwnerCell",        owners.astype(np.int64)),
        ("NeighborCell",     neighbors.astype(np.int64)),
        ("IsSurface",        is_surface_flag),
        ("FaceArea",         areas),
        ("Nx",               normals[:, 0]),
        ("Ny",               normals[:, 1]),
        ("Nz",               normals[:, 2]),
        ("AlignmentSquared", alignments),
        ("RadialAlignment",  radial_alignment),
    ]
    for name, arr in arrays_to_add:
        vtk_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(arr), deep=True)
        vtk_arr.SetName(name)
        grid.GetCellData().AddArray(vtk_arr)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTU: {filename}")
    n_surface = int(is_surface_flag.sum())
    n_inner = n_faces - n_surface
    print(f"  Exported {n_inner} inner + {n_surface} surface = {n_faces} faces to {filename}")


def export_foam_molecular_vtu(result, points_np, filename, u_target_np):
    """Export the foam as a ball-and-stick VTU for a "molecule-like" ParaView
    visualization. Each generator becomes a vertex cell (a ball), and each
    pair of neighboring cells (sharing an inner Voronoi face) becomes a line
    cell (a stick).

    Cell attributes:
        CellKind          (int)    0 = ball (a generator), 1 = stick (an
                                    adjacency edge between two generators)
        FaceArea          (float)  area of the shared face for sticks; 0 for balls
        AlignmentSquared  (float)  (n_f . u_target)^2 for the shared face;
                                    0 for balls

    Point attributes:
        GeneratorId       (int)    generator index 0..N-1

    In ParaView, apply a Threshold filter on CellKind to isolate the balls
    (0..0.5) and the sticks (0.5..1.5), then Glyph->Sphere on the balls and
    Tube on the sticks.
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        raise ImportError(
            "VTU export needs vtk. Install with: pip install vtk"
        )

    n_pts = points_np.shape[0]
    # Each interior face once, so each pair of neighbors gives one stick.
    polygons, owners, neighbors = foam_faces(result)
    n_sticks = len(polygons)
    stick_face_areas = np.zeros(n_sticks, dtype=np.float64)
    stick_alignments = np.zeros(n_sticks, dtype=np.float64)
    for i, polygon in enumerate(polygons):
        stick_face_areas[i], normal = polygon_area_normal(polygon)
        stick_alignments[i] = float((normal @ u_target_np) ** 2)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(points_np, dtype=np.float64), deep=True))
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    # Balls: one vtkVertex per generator
    for i in range(n_pts):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        grid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())

    # Sticks: one vtkLine per pair of neighboring cells
    for owner, neighbor in zip(owners, neighbors):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(owner))
        line.GetPointIds().SetId(1, int(neighbor))
        grid.InsertNextCell(line.GetCellType(), line.GetPointIds())

    # Cell-data arrays (length = n_pts + n_sticks). Balls get zero values so
    # ParaView's Threshold on CellKind cleanly separates them from sticks.
    cell_kind = np.concatenate([
        np.zeros(n_pts, dtype=np.int32),
        np.ones(n_sticks, dtype=np.int32),
    ])
    all_face_area = np.concatenate([np.zeros(n_pts), stick_face_areas])
    all_alignment = np.concatenate([np.zeros(n_pts), stick_alignments])
    for name, arr in [
        ("CellKind", cell_kind),
        ("FaceArea", all_face_area),
        ("AlignmentSquared", all_alignment),
    ]:
        vtk_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(arr), deep=True)
        vtk_arr.SetName(name)
        grid.GetCellData().AddArray(vtk_arr)

    # Point data: generator index (useful for labeling)
    gen_ids = np.arange(n_pts, dtype=np.int32)
    id_vtk = numpy_support.numpy_to_vtk(np.ascontiguousarray(gen_ids), deep=True)
    id_vtk.SetName("GeneratorId")
    grid.GetPointData().AddArray(id_vtk)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTU: {filename}")
    print(f"  Exported molecular VTU: {n_pts} balls, {n_sticks} sticks to {filename}")


# ── CLI helpers ─────────────────────────────────────────────────────────────


def _positional_args():
    return [a for a in sys.argv[1:] if not a.startswith("--")]


def parse_int_arg(idx, default):
    try:
        return int(_positional_args()[idx])
    except (IndexError, ValueError):
        return default


def parse_float_arg(idx, default):
    try:
        return float(_positional_args()[idx])
    except (IndexError, ValueError):
        return default


def parse_flag_value(prefix, default):
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


# ── Training loop ───────────────────────────────────────────────────────────


def train(threads=1, n_points=800, n_iter=1000, lr=0.003, seed=15,
          target_mode="uniaxial", lr_decay="cosine", lr_min_frac=0.01):
    """Adam-based optimization. Returns (history, final_points_numpy).

    `lr_decay` selects the learning-rate schedule:
        "none"   -- constant learning rate
        "cosine" -- cosine annealing from `lr` down to lr * lr_min_frac

    A Voronoi tessellation is only piecewise smooth in the generator
    positions: as generators move, faces appear and disappear, and each such
    topology change kicks the loss discontinuously. At a constant step size
    these kicks keep re-exciting the optimizer once the loss has decayed to
    their own magnitude, so the loss curve develops a persistent sawtooth even
    though the tensor itself has converged. Annealing the step size shrinks
    the per-iteration generator motion and therefore the rate of topology
    changes, which settles the tail.
    """
    A_target_np, u_target_np, target_label = get_target_config(target_mode)
    pc_alpha, pc_axis = get_per_cell_config(target_mode)
    # --alpha= overrides the mode's default anisotropy strength, for sweeping
    # the target without adding a new mode.
    _alpha_override = parse_flag_value("--alpha=", None)
    if _alpha_override is not None:
        pc_alpha = float(_alpha_override)
        if pc_axis is not None:
            # Keep the reported/plotted target consistent with what is actually
            # being optimised, otherwise the difference panel would compare
            # against the mode's default rather than the overridden target.
            _u = np.asarray(pc_axis, dtype=float)
            _u = _u / np.linalg.norm(_u)
            _uu = np.outer(_u, _u)
            A_target_np = (pc_alpha * _uu
                           + (1.0 - pc_alpha) / 2.0 * (np.eye(3) - _uu))
        target_label += f"  [alpha={pc_alpha:g}]"
    # The cross-sections of columnar cells are measured perpendicular to u.
    aspect_axis = (int(np.argmax(np.abs(pc_axis)))
                   if pc_axis is not None and pc_alpha < 1.0 / 3.0 else None)

    rng = np.random.default_rng(seed)
    initial_points = rng.random((n_points, 3)) * 0.9 + 0.05
    points_t = torch.tensor(initial_points, dtype=torch.float64, requires_grad=True)
    limits = [[0, 1], [0, 1], [0, 1]]
    optimizer = torch.optim.Adam([points_t], lr=lr)
    if lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iter, eta_min=lr * lr_min_frac)
    elif lr_decay == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_decay {lr_decay!r}; use none|cosine")

    tgt = A_target_np.diagonal()
    print(f"3D foam orientation (target={target_label}) — "
          f"{n_points} pts, {n_iter} iters, {threads} thread(s), lr={lr}")
    print(f"Target A_diag = ({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f})")
    print(f"{'Iter':>4}  {'Loss':>10}  {'L_main':>10}  "
          f"{'A_xx':>6}  {'A_yy':>6}  {'A_zz':>6}  {'#inner_f':>9}")
    print("-" * 80)

    history = {"loss": [], "L_main": [], "L_volume": [],
               "A_xx": [], "A_yy": [], "A_zz": [],
               "A_xy": [], "A_xz": [], "A_yz": []}

    initial_metrics = None
    t0 = time.perf_counter()
    for it in range(n_iter):
        loss, terms = compute_loss_3d(points_t, limits, threads,
                                      target_mode=target_mode,
                                      per_cell_alpha=pc_alpha,
                                      per_cell_axis=pc_axis)
        if initial_metrics is None:
            initial_metrics = foam_metrics(terms, pc_alpha, aspect_axis)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            points_t.clamp_(1e-4, 1.0 - 1e-4)

        A_np = terms["A"].detach().cpu().numpy()
        history["loss"].append(float(loss.item()))
        history["L_main"].append(float(terms["L_main"].item()))
        history["L_volume"].append(float(terms["L_volume"].item()))
        history["A_xx"].append(A_np[0, 0])
        history["A_yy"].append(A_np[1, 1])
        history["A_zz"].append(A_np[2, 2])
        history["A_xy"].append(A_np[0, 1])
        history["A_xz"].append(A_np[0, 2])
        history["A_yz"].append(A_np[1, 2])

        if it == 0 or (it + 1) % 10 == 0 or it + 1 == n_iter:
            print(f"{it + 1:>4}  {loss.item():>10.5g}  "
                  f"{terms['L_main'].item():>10.5g}  "
                  f"{A_np[0, 0]:>6.3f}  {A_np[1, 1]:>6.3f}  {A_np[2, 2]:>6.3f}  "
                  f"{terms['n_inner_faces']:>9}")

    elapsed = time.perf_counter() - t0

    # Final state — recompute once with no grad for the summary figure
    final_points = points_t.detach().cpu().numpy()
    with torch.no_grad():
        pts_const = torch.tensor(final_points, dtype=torch.float64)
        _, final_terms = compute_loss_3d(pts_const, limits, threads,
                                         target_mode=target_mode,
                                         per_cell_alpha=pc_alpha,
                                         per_cell_axis=pc_axis)
    A_final_np = final_terms["A"].detach().cpu().numpy()
    final_metrics = foam_metrics(final_terms, pc_alpha, aspect_axis)

    print("-" * 80)
    print(f"Initial loss : {history['loss'][0]:.5g}")
    print(f"Final loss   : {history['loss'][-1]:.5g}")
    print(f"Final A diag : ({A_final_np[0,0]:.4f}, "
          f"{A_final_np[1,1]:.4f}, {A_final_np[2,2]:.4f})")
    print(f"Final A off  : (xy={A_final_np[0,1]:+.4f}, "
          f"xz={A_final_np[0,2]:+.4f}, yz={A_final_np[1,2]:+.4f})")
    print(f"Target A_diag: ({tgt[0]:.4f}, {tgt[1]:.4f}, {tgt[2]:.4f})")
    print(f"Alignment M  : {initial_metrics['M']:.4f} -> {final_metrics['M']:.4f}")
    print(f"Mean u^T A_c u: {initial_metrics['strength']:.4f} -> "
          f"{final_metrics['strength']:.4f} (target alpha = {pc_alpha:g})")
    mean_A = final_metrics["mean_A_cells"]
    print(f"Mean of A_c diag: ({mean_A[0,0]:.4f}, {mean_A[1,1]:.4f}, "
          f"{mean_A[2,2]:.4f})")
    if aspect_axis is not None:
        print(f"Median aspect ratio of the cross-sections: "
              f"{initial_metrics['median_aspect']:.3f} -> "
              f"{final_metrics['median_aspect']:.3f}")
    print(f"Wall time    : {elapsed:.2f} s "
          f"({elapsed / n_iter * 1000:.1f} ms/iter)")

    # Save summary figure
    default_fig = f"examples/foam_3d_orientation_{target_mode}.png"
    fig_name = parse_flag_value("--out=", default_fig)
    # Create the output directory up front. Matplotlib will not create it, and
    # a missing directory would otherwise abort the run at the very last step,
    # after the full optimisation has already been paid for.
    Path(fig_name).parent.mkdir(parents=True, exist_ok=True)

    # Persist the training history BEFORE plotting. The optimisation is the
    # expensive part; every figure below can be regenerated from this archive
    # with replot_foam_3d.py. Writing it first means a plotting failure costs
    # a re-render, not a re-train.
    history_path = fig_name.replace(".png", "_history.npz")
    np.savez_compressed(
        history_path,
        initial_points=initial_points,
        final_points=final_points,
        A_target=A_target_np,
        A_final=A_final_np,
        u_target=u_target_np,
        initial_M=initial_metrics["M"],
        final_M=final_metrics["M"],
        **{f"hist_{k}": np.asarray(v) for k, v in history.items()},
    )
    print(f"Saved history: {history_path}")

    save_summary_figure_3d(
        initial_points=initial_points,
        final_points=final_points,
        history=history,
        output_path=fig_name,
        u_target_np=u_target_np,
        limits=limits,
        threads=threads,
        slice_axis=get_slice_axis(target_mode),
    )
    print(f"Saved figure : {fig_name}")

    # Optionally also write every panel as a standalone image, for LaTeX
    # subfigure composition (the (a)/(b)/(c) labels are added by LaTeX).
    panel_dir = parse_flag_value("--panel-dir=", None)
    if panel_dir is not None:
        save_panels_3d(
            out_dir=panel_dir,
            initial_points=initial_points,
            final_points=final_points,
            history=history,
            A_target_np=A_target_np,
            A_final_np=A_final_np,
            u_target_np=u_target_np,
            limits=limits,
            threads=threads,
            slice_axis=get_slice_axis(target_mode),
            tensor_panel=("absolute" if target_mode == "circular"
                          else "difference"),
        )

    # Optionally export the foam to a ParaView VTU file
    if "--vtu" in sys.argv:
        vtu_name = parse_flag_value(
            "--vtu-out=", fig_name.replace(".png", ".vtu")
        )
        # Second VTU with the ball-and-stick (molecular) representation
        mol_vtu_name = vtu_name.replace(".vtu", "_molecular.vtu")
        result = voroplex.compute_voronoi_3d(final_points, limits, threads=threads)
        export_foam_vtu(result, final_points, vtu_name, u_target_np)
        export_foam_molecular_vtu(result, final_points, mol_vtu_name,
                                  u_target_np)

    return history, final_points


def main():
    """Entry point: optimize one foam and write its figure and history."""
    threads = parse_int_arg(0, 1)
    n_points = parse_int_arg(1, 800)
    n_iter = parse_int_arg(2, 1000)
    lr = parse_float_arg(3, 0.003)
    target_mode = parse_flag_value("--target=", "uniaxial")
    lr_decay = parse_flag_value("--lr-decay=", "cosine")
    lr_min_frac = float(parse_flag_value("--lr-min-frac=", "0.01"))
    global LAMBDA_VOLUME
    _raw_vol = parse_flag_value("--lambda-vol=", None)
    if _raw_vol is not None:
        LAMBDA_VOLUME = float(_raw_vol)
        print(f"Overriding LAMBDA_VOLUME = {LAMBDA_VOLUME}")
    train(threads=threads, n_points=n_points, n_iter=n_iter, lr=lr,
          target_mode=target_mode, lr_decay=lr_decay, lr_min_frac=lr_min_frac)


if __name__ == "__main__":
    main()
