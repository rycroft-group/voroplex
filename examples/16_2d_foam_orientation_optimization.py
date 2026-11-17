"""
2D anisotropic foam design with VoroPlex and PyTorch
====================================================

Optimizes 2D generator positions so that every Voronoi cell's length-weighted
edge-normal tensor matches an anisotropic target. For one cell,

    A_c = sum_e (l_e n_e n_e^T) / sum_e l_e          # 2x2 symmetric, trace 1

over that cell's interior edges, with l_e the edge length and n_e its unit
normal. The target at cell c is built from a preferred axis u(c) and an
anisotropy strength alpha,

    A_tgt(c) = alpha u(c) u(c)^T + (1 - alpha) (I - u(c) u(c)^T)

so alpha > 1/2 concentrates wall normals along u(c), which makes the walls
perpendicular to u(c) and the cells elongated across it. The loss is the mean
squared Frobenius distance between the two, plus a penalty on the variation in
cell area:

    L = mean_c ||A_c - A_tgt(c)||_F^2 + lambda_area Var(a_c) / mean(a_c)^2

Without the second term the first is satisfied just as well by cells collapsed
into slivers, which match the target tensor but make a useless foam.

The tessellation uses the per-cell representation: cell_vertices gives each
cell's own vertices, differentiable with respect to the generators, and the
tensors and areas are sums over the edges and fan triangles of every cell,
with no loops in Python.

Run from the project root:

    python examples/16_2d_foam_orientation_optimization.py
    python examples/16_2d_foam_orientation_optimization.py --angle=30 --movie

The defaults are the settings of Section 5.3 of the VoroPlex paper: 120
generators, 1000 iterations of Adam with a learning rate of 0.004 that decays
to 1/100 of that on a cosine schedule, alpha = 0.75, and the axis u = (1, 0).
The paper's figure is made with

    python examples/16_2d_foam_orientation_optimization.py --panel-dir=foam_2d

The run uses one thread by default, so that it is exactly reproducible: with
several threads, the gradient is summed in a different order, and the results
differ slightly from run to run.

The alignment M of the paper is printed at the start and at the end, together
with the mean of u^T A_c u, the fraction of wall length that faces along u,
which approaches alpha.

Optional positional arguments:
    [num_threads] [num_points] [num_iters] [lr]

Flags:
    --angle=<degrees>         target axis angle, default 0
    --anisotropy=<float>      target alpha in [0, 1], default 0.75
    --field={uniaxial|circular}
                              spatial form of u(c): a fixed direction, or
                              radial about the domain centre. Default uniaxial
    --lambda-area=<float>     weight of the cell-area penalty, default 0.30
    --lr-decay={none|cosine}  anneal the step length, default cosine
    --lr-min-frac=<float>     final lr as a fraction of lr, default 0.01
    --out=<path>              summary figure
    --panel-dir=<dir>         standalone panels and the data archive
    --movie                   save evolution mp4 (requires ffmpeg)
"""

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import voroplex
from foam_figure_style import (
    apply_paper_style, panel_label, draw_tensor_difference, style_loss_axes,
    hide_shared_y, new_panel, save_panel, plot_loss_curve, FONT_ROW_LABEL,
)

try:
    import imageio_ffmpeg

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


EPS = 1e-12

# ── Loss weight ────────────────────────────────────────────────────────────
# The loss has two terms,
#
#     L = mean_c ||A_c - A_tgt(c)||_F^2  +  LAMBDA_AREA Var(a_i) / mean(a_i)^2
#
# and LAMBDA_AREA is the weight of the second. Overridable with --lambda-area=.
LAMBDA_AREA = 0.30


# ── Argument parsing ────────────────────────────────────────────────────────

POSITIONAL_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]


def parse_int_arg(i, default):
    try:
        return int(POSITIONAL_ARGS[i])
    except (IndexError, ValueError):
        return default


def parse_float_arg(i, default):
    try:
        return float(POSITIONAL_ARGS[i])
    except (IndexError, ValueError):
        return default


def parse_flag_value(prefix, default):
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


# ── Target tensor and cell geometry ─────────────────────────────────────────


def build_target_tensor(angle_degrees, anisotropy, dtype, device):
    """The constant target tensor and its axis, for the summary figure."""
    alpha = float(anisotropy)
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("--anisotropy must be between 0 and 1")
    theta = torch.tensor(np.deg2rad(angle_degrees), dtype=dtype, device=device)
    u = torch.stack([torch.cos(theta), torch.sin(theta)])
    u_perp = torch.stack([-u[1], u[0]])
    A = alpha * torch.outer(u, u) + (1.0 - alpha) * torch.outer(u_perp, u_perp)
    return A, u, alpha


def target_axes(points_t, angle_degrees, field, center=(0.5, 0.5), r_floor=0.05):
    """The target axis u(c) of every cell, and the cells that are used.

    For the uniaxial field, u is a fixed direction. For the circular field, it
    points away from the centre of the domain, and depends on the generator
    position, so gradients flow through it. The radial direction is not
    defined at the centre, so cells within r_floor of it are left out.
    """
    n = points_t.shape[0]
    usable = torch.ones(n, dtype=torch.bool, device=points_t.device)
    if field == "circular":
        delta = points_t - points_t.new_tensor(center)
        r = torch.linalg.norm(delta, dim=1)
        u = delta / r.clamp_min(1e-9).unsqueeze(1)
        usable = r >= r_floor
    elif field == "uniaxial":
        theta = np.deg2rad(angle_degrees)
        u = points_t.new_tensor([np.cos(theta), np.sin(theta)]).expand(n, 2)
    else:
        raise SystemExit(f"Unknown --field={field}; use uniaxial|circular")
    return u, usable


def edge_arrays(result):
    """Endpoints, owners, and neighbors of every edge of every cell.

    In the per-cell representation, each cell lists its own edges, so an
    interior edge appears once in each of the two cells that share it. The
    endpoint indices refer to the rows of cell_vertices(result, ...).
    """
    owners = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    start = result.facet_vertex_offsets[:-1]
    base = result.cell_vertex_offsets[owners]
    first = base + result.facet_vertices[start]
    second = base + result.facet_vertices[start + 1]
    return owners, np.asarray(result.facet_neighbors), first, second


def cell_terms(points_t, limits, threads, alpha, angle_degrees, field):
    """Fabric tensors, areas, and the loss, from the per-cell representation.

    For each cell c, A_c = sum_e l_e n_e n_e^T / sum_e l_e over the edges that
    c shares with other cells. Edges on the boundary of the domain are left out,
    since they come from cutting the foam to the size of the domain.
    """
    result = voroplex.compute_voronoi_2d(
        points_t.detach().cpu().numpy(), limits, representation="per_cell",
        threads=threads)
    vertices = voroplex.cell_vertices(result, points_t)
    n_cells = points_t.shape[0]
    dtype, device = points_t.dtype, points_t.device

    owners, neighbors, first, second = edge_arrays(result)
    vec = vertices[torch.as_tensor(second)] - vertices[torch.as_tensor(first)]
    lengths = torch.linalg.norm(vec, dim=1).clamp_min(EPS)
    normals = torch.stack([-vec[:, 1], vec[:, 0]], dim=1) / lengths[:, None]
    inner = torch.as_tensor(neighbors >= 0, device=device)
    weights = torch.where(inner, lengths, torch.zeros_like(lengths))
    weighted = weights[:, None, None] * normals[:, :, None] * normals[:, None, :]

    owners_t = torch.as_tensor(owners, device=device)
    num = torch.zeros((n_cells, 2, 2), dtype=dtype, device=device).index_add(
        0, owners_t, weighted)
    den = torch.zeros(n_cells, dtype=dtype, device=device).index_add(
        0, owners_t, weights)
    valid = den > EPS
    A_cells = num / den.clamp_min(EPS)[:, None, None]
    # Each interior edge is counted once in each of its two cells, which does
    # not change this length-weighted average over the whole foam.
    A = weighted.sum(dim=0) / weights.sum().clamp_min(EPS)

    u, usable = target_axes(points_t, angle_degrees, field)
    uu = u[:, :, None] * u[:, None, :]
    eye = torch.eye(2, dtype=dtype, device=device).unsqueeze(0)
    A_tgt = alpha * uu + (1.0 - alpha) * (eye - uu)
    keep = valid & usable
    resid = ((A_cells - A_tgt) ** 2).sum(dim=(1, 2))
    L_main = resid[keep].mean() if keep.any() else points_t.new_tensor(0.0)

    # Cell areas, from the triangles that join each generator to its edges.
    tri_owners, tri = voroplex.cell_fan_simplices(result, device=device)
    q1 = vertices[tri[:, 0]] - points_t[tri_owners]
    q2 = vertices[tri[:, 1]] - points_t[tri_owners]
    tri_area = 0.5 * (q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0])
    areas = torch.zeros(n_cells, dtype=dtype, device=device).index_add(
        0, tri_owners, tri_area)

    # Each interior edge once, for plotting and diagnostics.
    once = (neighbors >= 0) & (owners < neighbors)
    return {
        "result": result, "vertices": vertices, "A_cells": A_cells,
        "A": A, "u": u, "keep": keep, "L_main": L_main, "areas": areas,
        "edge_xy": torch.stack([vertices[torch.as_tensor(first[once])],
                                vertices[torch.as_tensor(second[once])]], dim=1),
        "normals": normals[torch.as_tensor(once)],
        "lengths": lengths[torch.as_tensor(once)],
    }


def alignment(A_cells, u, alpha, keep):
    """The alignment M of the paper, and the mean strength u^T A_c u.

    e_c is the eigenvector of A_c with the largest eigenvalue when alpha > 1/2,
    and with the smallest when alpha < 1/2, and M is the mean of
    (e_c . u(c))^2: 1 when every cell is aligned with its own target axis, and
    1/2 on average for axes unrelated to u. The strength, the mean of
    u(c)^T A_c u(c), is the fraction of wall length facing along u(c), and
    approaches alpha.
    """
    A_cells, u = A_cells[keep].detach(), u[keep].detach()
    _, vecs = torch.linalg.eigh(A_cells)
    e = vecs[:, :, -1] if alpha > 0.5 else vecs[:, :, 0]
    M = float((((e * u).sum(dim=1)) ** 2).mean())
    strength = float(torch.einsum("ci,cij,cj->c", u, A_cells, u).mean())
    return M, strength


def compute_loss(points_t, limits, threads, alpha, angle_degrees, field):
    terms = cell_terms(points_t, limits, threads, alpha, angle_degrees, field)
    areas = terms["areas"]
    L_area = torch.var(areas, unbiased=False) / (areas.mean() + EPS) ** 2
    total = terms["L_main"] + LAMBDA_AREA * L_area
    eigvals = torch.linalg.eigvalsh(terms["A"].detach())
    normals = terms["normals"].detach()
    return {
        "loss": total,
        "L_main": terms["L_main"],
        "L_area": L_area,
        "A": terms["A"],
        "A_cells": terms["A_cells"],
        "u": terms["u"],
        "keep": terms["keep"],
        "lam_min": float(eigvals[0]),
        "lam_max": float(eigvals[1]),
        "mean_abs_nx": float(normals[:, 0].abs().mean()) if len(normals) else float("nan"),
        "n_edges": int(normals.shape[0]),
    }


# ── Plotting ────────────────────────────────────────────────────────────────


def snapshot_foam(points_np, limits, threads):
    """Non-differentiable snapshot used for the initial / final plots."""
    points_t = torch.tensor(points_np, dtype=torch.float64)
    with torch.no_grad():
        terms = cell_terms(points_t, limits, threads, 0.5, 0.0, "uniaxial")
    return {
        "edge_xy": terms["edge_xy"].cpu().numpy(),
        "normals": terms["normals"].cpu().numpy(),
        "lengths": terms["lengths"].cpu().numpy(),
        "A": terms["A"].cpu().numpy(),
    }


def draw_foam(ax, points, edge_xy, normals, u_target, title=None):
    if len(edge_xy):
        # Color edges by how aligned their normal is with u_target (1 = aligned, 0 = orthogonal)
        cos2 = (normals @ u_target) ** 2
        for edge, c in zip(edge_xy, cos2):
            ax.plot(edge[:, 0], edge[:, 1], color=plt.cm.viridis(float(c)),
                    linewidth=1.1, alpha=0.95, zorder=1)
    # Scale the generator markers with the local spacing ~ 1/sqrt(N); a fixed
    # size that reads well at N=120 turns the panel into a solid mass of dots
    # at N=800.
    n_pts = max(len(points), 1)
    marker_area = float(np.clip((45.0 / np.sqrt(n_pts)) ** 2, 1.5, 22.0))
    ax.scatter(points[:, 0], points[:, 1], s=marker_area, color="#d83b3b",
               edgecolor="white", linewidth=0.2, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y")


def draw_tensor(ax, M, title):
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=["x", "y"])
    ax.set_yticks([0, 1], labels=["x", "y"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white")
    return im


def _log_ymin(values):
    arr = np.asarray(values, dtype=np.float64)
    pos = arr[arr > 0]
    if not len(pos):
        return 1e-16
    return max(pos.min() * 0.8, np.finfo(np.float64).tiny)


def save_summary_figure(initial_points, final_points, initial_snap, final_snap,
                        history, target_A_np, u_target_np, output_path,
                        angle, alpha):
    """Five-panel summary figure for 2D foam orientation, laid out in one row.

    Panels:
        (a) initial foam
        (b) optimized foam, edges coloured by (n_e . u)^2 -- shares (a)'s y axis
        (c) tensor mismatch A_tgt - A as a single signed heatmap
        (d) total and main loss versus iteration
        (e) diagonal entries of A versus iteration, against their targets

    No titles are drawn inside the axes: each panel carries a bold letter and
    is described in the LaTeX caption.
    """
    apply_paper_style()

    fig = plt.figure(figsize=(16.0, 3.6))
    # Generous column spacing: the loss panel's decade tick labels are wide and
    # push its y-axis label left into the neighbouring heatmap otherwise.
    grid = fig.add_gridspec(1, 5, wspace=0.62)
    ax_initial = fig.add_subplot(grid[0, 0])
    ax_final = fig.add_subplot(grid[0, 1], sharey=ax_initial)
    ax_diff = fig.add_subplot(grid[0, 2])
    ax_loss = fig.add_subplot(grid[0, 3])
    ax_diag = fig.add_subplot(grid[0, 4])

    u_t = torch.tensor(u_target_np, dtype=torch.float64)
    draw_foam(ax_initial, initial_points,
              initial_snap["edge_xy"], torch.tensor(initial_snap["normals"]),
              u_t)
    draw_foam(ax_final, final_points,
              final_snap["edge_xy"], torch.tensor(final_snap["normals"]),
              u_t)
    # (b) shares (a)'s vertical axis, so its tick labels are redundant.
    hide_shared_y(ax_final)
    panel_label(ax_initial, "a")
    panel_label(ax_final, "b")

    # (c) One heatmap of the signed mismatch, replacing the previous pair of
    # "target A" / "achieved A" panels.
    draw_tensor_difference(ax_diff, target_A_np, final_snap["A"],
                           axis_labels=["x", "y"], cbar=True, fig=fig,
                           cell_fontsize=15)
    panel_label(ax_diff, "c")

    iters = np.arange(1, len(history["loss"]) + 1)
    ax_loss.semilogy(iters, history["loss"], color="#2454a6",
                     label=r"$\mathcal{L}$")
    ax_loss.semilogy(iters, history["L_main"], color="#a64124", alpha=0.9,
                     label=r"$\mathcal{L}_{\mathrm{main}}$")
    style_loss_axes(ax_loss, history["loss"], history["L_main"])
    ax_loss.legend()
    panel_label(ax_loss, "d")

    ax_diag.plot(iters, history["A_xx"], color="#d83b3b", label=r"$A_{xx}$")
    ax_diag.plot(iters, history["A_yy"], color="#2454a6", label=r"$A_{yy}$")
    for value, colour in zip(target_A_np.diagonal(), ("#d83b3b", "#2454a6")):
        ax_diag.axhline(value, color=colour, linestyle="--",
                        linewidth=1.4, alpha=0.65)
    ax_diag.set_xlabel("Iteration")
    ax_diag.set_ylabel(r"$A_{ii}$")
    ax_diag.set_ylim(0, 1)
    # The two curves settle near 0.75 and 0.25, leaving the middle band clear.
    ax_diag.legend(ncol=1, handlelength=1.2, loc="center",
                   framealpha=0.9, borderpad=0.3, labelspacing=0.3)
    ax_diag.grid(True, alpha=0.25)
    panel_label(ax_diag, "e")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Common width/height ratio for every 2D panel, so that a row of five
# renders at uniform height once LaTeX scales them to a common width.
PANEL_ASPECT = 1.09

# Vertical gap, in figure fractions, between the bottom of the drawn row and
# the top of the (a)..(e) letters.
LABEL_GAP = 0.008

# Colour bar beside panel (b) of the composed row: width and gap as
# fractions of the figure, and the clearance from its tick labels to its
# own label, in points. The label is kept close so that it reads as the
# bar's and not as the loss panel's y axis.
CBAR_WIDTH = 0.013
CBAR_PAD = 0.012
CBAR_LABEL_PAD = 3.0

# Width of the composed row, in inches. Wide enough that the bar's label
# and the loss panel's y axis do not crowd each other.
COMBINED_WIDTH = 14.4


def save_combined_2d(out_path, initial_points, final_points, initial_snap,
                     final_snap, history, target_A_np, u_target_np):
    """Write the three 2D panels as ONE wide image with (a)..(c) beneath.

    Composing the row here rather than with LaTeX subfigures lets the panels
    be packed much closer together, so each one is drawn larger for the same
    total figure width. The letters are drawn in the same Times face as the
    rest of the figure and are not emboldened.

    The objective matches every cell's own fabric tensor to a target evaluated
    at that cell, so the foam has no single mean fabric tensor to report, and
    the two panels that showed one have been dropped. `target_A_np` is still
    accepted, and still stored in the archive, but is no longer drawn.
    """
    apply_paper_style()

    fig = plt.figure(figsize=(COMBINED_WIDTH, 4.4))
    grid = fig.add_gridspec(1, 3, wspace=0.34, left=0.035, right=0.995,
                            top=0.97, bottom=0.30)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_initial, ax_final, ax_loss = axes

    u_t = torch.tensor(u_target_np, dtype=torch.float64)
    draw_foam(ax_initial, initial_points, initial_snap["edge_xy"],
              torch.tensor(initial_snap["normals"]), u_t)
    draw_foam(ax_final, final_points, final_snap["edge_xy"],
              torch.tensor(final_snap["normals"]), u_t)
    hide_shared_y(ax_final)

    iters = np.arange(1, len(history["loss"]) + 1)
    plot_loss_curve(ax_loss, iters, history["loss"], "#2454a6",
                    r"$\mathcal{L}$")
    plot_loss_curve(ax_loss, iters, history["L_main"], "#a64124",
                    r"$\mathcal{L}_{\mathrm{main}}$")
    style_loss_axes(ax_loss, history["loss"], history["L_main"])
    ax_loss.legend()
    ax_loss.set_ylabel(r"$\mathcal{L}$")

    # The two foam panels are aspect="equal", so the first draw shrinks each
    # to a square inside its gridspec cell while the loss panel keeps the
    # whole cell. Giving the loss panel the square panels' box lines the
    # three up as one row.
    fig.canvas.draw()
    box_square = ax_initial.get_position()
    box_loss = ax_loss.get_position()
    ax_loss.set_position([box_loss.x0, box_square.y0,
                          box_square.width, box_square.height])

    # One colour bar for both foam panels, beside (b). They share a scale, so
    # a single bar serves both and the caption says so.
    box_final = ax_final.get_position()
    cax = fig.add_axes([box_final.x1 + CBAR_PAD, box_final.y0,
                        CBAR_WIDTH, box_final.height])
    mappable = plt.cm.ScalarMappable(cmap="viridis",
                                     norm=plt.Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$(n_f \cdot u)^2$", labelpad=CBAR_LABEL_PAD)
    cbar.set_ticks([0.0, 0.5, 1.0])

    # ---- Space the three panels ------------------------------------------
    # Each panel occupies a different amount of width once its furniture is
    # counted: (a) carries a y axis, (b) does not but drags the colour bar.
    # Pin the outer two to the margins and centre the middle one between
    # them, which puts the three letters at equal spacing.
    groups = [[ax_initial], [ax_final, cax], [ax_loss]]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def group_extent(group):
        x0 = min(inv.transform(a.get_tightbbox(renderer).get_points())[0][0]
                 for a in group)
        x1 = max(inv.transform(a.get_tightbbox(renderer).get_points())[1][0]
                 for a in group)
        return x0, x1

    def shift_group(group, dx):
        for a in group:
            pos = a.get_position()
            a.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])

    def centre(ax):
        pos = ax.get_position()
        return 0.5 * (pos.x0 + pos.x1)

    left, right = 0.012, 0.995
    shift_group(groups[0], left - group_extent(groups[0])[0])
    shift_group(groups[2], right - group_extent(groups[2])[1])
    shift_group(groups[1], 0.5 * (centre(ax_initial) + centre(ax_loss))
                - centre(ax_final))

    # (a)..(c) centred on each panel's PLOT RECTANGLE, on a common baseline.
    # Centring on the tight bounding box instead would pull (a)'s letter left
    # relative to (b)'s, because only (a) carries a y axis.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y_bottom = min(
        inv.transform(a.get_tightbbox(renderer).get_points())[0][1]
        for group in groups for a in group)

    for ax, letter in zip(axes, "abc"):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), y_bottom - LABEL_GAP, f"({letter})",
                 fontsize=FONT_ROW_LABEL, ha="center", va="top")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Wrote combined 2D figure to {out_path}")
    return out_path


def save_panels_2d(out_dir, initial_points, final_points, initial_snap,
                   final_snap, history, target_A_np, u_target_np):
    """Write each panel of the 2D summary as its own image file.

    The paper composes these with LaTeX subfigures, which add the (a)/(b)/(c)
    labels, so no letters or titles are drawn here. Files written:

        a_initial_foam.png
        b_optimized_foam.png
        c_tensor.png
        d_loss.png
        e_diagonals.png
    """
    written = []
    u_t = torch.tensor(u_target_np, dtype=torch.float64)

    # (a) and (b): the foam before and after. As standalone figures they no
    # longer share an axis, so both keep their y tick labels.
    # Panel figure sizes are tuned so that all five panels end up with the
    # SAME aspect ratio after a tight bounding box, and therefore render at
    # equal height when LaTeX scales them to a common width. Panels differ in
    # how much furniture they carry -- (b) has no y axis, (c) carries a
    # colourbar -- so equal figsizes would not give equal aspects.
    for name, pts, snap, fsize in (
            ("a_initial_foam", initial_points, initial_snap, (4.2, 4.2)),
            ("b_optimized_foam", final_points, final_snap, (4.7, 4.2))):
        fig, ax = new_panel(figsize=fsize)
        draw_foam(ax, pts, snap["edge_xy"],
                  torch.tensor(snap["normals"]), u_t)
        # (a) and (b) are placed side by side and span the same range, so the
        # second one drops its vertical axis: the two read as sharing it, even
        # though they are separate images composed by LaTeX subfigures.
        if name.startswith("b_"):
            hide_shared_y(ax)
        written.append(save_panel(fig, out_dir, name, pad_aspect=PANEL_ASPECT))

    # (c) Signed tensor mismatch.
    fig, ax = new_panel(figsize=(5.1, 4.2))
    draw_tensor_difference(ax, target_A_np, final_snap["A"],
                           axis_labels=["x", "y"], cbar=True, fig=fig,
                           cell_fontsize=17)
    written.append(save_panel(fig, out_dir, "c_tensor", pad_aspect=PANEL_ASPECT))

    # (d) Loss curves.
    iters = np.arange(1, len(history["loss"]) + 1)
    fig, ax = new_panel(figsize=(4.2, 4.35))
    plot_loss_curve(ax, iters, history["loss"], "#2454a6",
                    r"$\mathcal{L}$")
    plot_loss_curve(ax, iters, history["L_main"], "#a64124",
                    r"$\mathcal{L}_{\mathrm{main}}$")
    style_loss_axes(ax, history["loss"], history["L_main"])
    ax.legend()
    written.append(save_panel(fig, out_dir, "d_loss", pad_aspect=PANEL_ASPECT))

    # (e) Diagonal entries of A against their targets.
    fig, ax = new_panel(figsize=(4.2, 4.2))
    ax.plot(iters, history["A_xx"], color="#d83b3b", label=r"$A_{xx}$")
    ax.plot(iters, history["A_yy"], color="#2454a6", label=r"$A_{yy}$")
    for value, colour in zip(target_A_np.diagonal(), ("#d83b3b", "#2454a6")):
        ax.axhline(value, color=colour, linestyle="--", linewidth=1.4,
                   alpha=0.65)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$A_{ii}$")
    ax.set_ylim(0, 1)
    ax.legend(loc="center", framealpha=0.9, borderpad=0.3, labelspacing=0.3)
    ax.grid(True, alpha=0.25)
    written.append(save_panel(fig, out_dir, "e_diagonals", pad_aspect=PANEL_ASPECT))

    print(f"Wrote {len(written)} panels to {out_dir}/")
    return written


def save_evolution_movie(snapshot_points, limits, threads, u_target_np,
                         output_path, fps=24):
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(i):
        ax.clear()
        pts = snapshot_points[i]
        snap = snapshot_foam(pts, limits, threads)
        u_t = torch.tensor(u_target_np, dtype=torch.float64)
        draw_foam(ax, pts, snap["edge_xy"],
                  torch.tensor(snap["normals"]), u_t, f"Iter {i + 1}")

    ani = animation.FuncAnimation(
        fig, draw_frame, frames=len(snapshot_points), interval=1000 / fps
    )
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(str(output_path), writer=writer, dpi=140)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    threads = parse_int_arg(0, 1)
    n_points = parse_int_arg(1, 120)
    n_iter = parse_int_arg(2, 1000)
    lr = parse_float_arg(3, 0.004)
    angle = float(parse_flag_value("--angle=", "0"))
    alpha = float(parse_flag_value("--anisotropy=", "0.75"))
    field = parse_flag_value("--field=", "uniaxial")
    make_movie = "--movie" in sys.argv
    limits = [[0, 1], [0, 1]]

    global LAMBDA_AREA
    _raw_area = parse_flag_value("--lambda-area=", None)
    if _raw_area is not None:
        LAMBDA_AREA = float(_raw_area)
        print(f"Overriding LAMBDA_AREA = {LAMBDA_AREA}")

    rng = np.random.default_rng(14)
    initial_points = rng.random((n_points, 2)) * 0.9 + 0.05
    points_t = torch.tensor(initial_points, dtype=torch.float64, requires_grad=True)

    A_target, u_target, alpha_used = build_target_tensor(
        angle, alpha, dtype=points_t.dtype, device=points_t.device
    )
    A_target_np = A_target.detach().cpu().numpy()
    u_target_np = u_target.detach().cpu().numpy()

    print(f"2D foam orientation — angle={angle:g}°, α={alpha_used:g}, "
          f"{n_points} pts, {n_iter} iters")
    print(f"{'Iter':>4}  {'Loss':>11}  {'L_main':>11}  "
          f"{'λ_max':>7}  {'λ_min':>7}  {'mean|n_x|':>10}  {'#edges':>7}")
    print("-" * 75)

    optimizer = torch.optim.Adam([points_t], lr=lr)
    # Adam normalises by gradient magnitude, so its step length stays close to
    # `lr` however small the gradient becomes. Left constant, the generators go
    # on shifting by ~lr per iteration long after the tensor has converged,
    # which keeps re-triggering Voronoi topology changes and throws the loss
    # back up by orders of magnitude. Annealing the step length stops that,
    # provided the decay is spread over a budget long enough to converge first.
    lr_decay = parse_flag_value("--lr-decay=", "cosine")
    if lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iter,
            eta_min=lr * float(parse_flag_value("--lr-min-frac=", "0.01")))
    elif lr_decay == "none":
        scheduler = None
    else:
        raise SystemExit(f"Unknown --lr-decay={lr_decay}; use none|cosine")
    history = {"loss": [], "L_main": [], "lam_max": [], "lam_min": [],
               "mean_abs_nx": [], "A_xx": [], "A_yy": [], "A_xy": []}
    snapshot_points = []

    initial_snap = snapshot_foam(initial_points, limits, threads)
    initial_M = None

    t0 = time.perf_counter()
    for it in range(n_iter):
        terms = compute_loss(points_t, limits, threads, alpha_used,
                             angle, field)
        loss = terms["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            points_t.clamp_(1e-4, 1.0 - 1e-4)

        history["loss"].append(float(loss.item()))
        history["L_main"].append(float(terms["L_main"].item()))
        history["lam_max"].append(terms["lam_max"])
        history["lam_min"].append(terms["lam_min"])
        history["mean_abs_nx"].append(terms["mean_abs_nx"])
        A_np = terms["A"].detach().cpu().numpy()
        history["A_xx"].append(float(A_np[0, 0]))
        history["A_yy"].append(float(A_np[1, 1]))
        history["A_xy"].append(float(A_np[0, 1]))
        if initial_M is None:
            initial_M, initial_strength = alignment(
                terms["A_cells"], terms["u"], alpha_used, terms["keep"])

        if make_movie:
            snapshot_points.append(points_t.detach().cpu().numpy().copy())

        if it == 0 or (it + 1) % 25 == 0 or it + 1 == n_iter:
            print(
                f"{it + 1:>4}  {loss.item():>11.6g}  "
                f"{terms['L_main'].item():>11.6g}  "
                f"{terms['lam_max']:>7.4f}  {terms['lam_min']:>7.4f}  "
                f"{terms['mean_abs_nx']:>10.4g}  {terms['n_edges']:>7}"
            )

    elapsed = time.perf_counter() - t0
    final_points = points_t.detach().cpu().numpy()
    final_snap = snapshot_foam(final_points, limits, threads)

    summary_path = Path(parse_flag_value(
        "--out=", "examples/foam_2d_orientation.png"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary_figure(
        initial_points, final_points, initial_snap, final_snap,
        history, A_target_np, u_target_np, summary_path,
        angle, alpha_used,
    )

    # Optionally also write every panel as a standalone image, for LaTeX
    # subfigure composition (the (a)/(b)/(c) labels are added by LaTeX).
    panel_dir = parse_flag_value("--panel-dir=", None)
    if panel_dir is not None:
        save_panels_2d(panel_dir, initial_points, final_points,
                       initial_snap, final_snap, history,
                       A_target_np, u_target_np)
        save_combined_2d(Path(panel_dir) / "combined_row.png",
                         initial_points, final_points, initial_snap,
                         final_snap, history, A_target_np, u_target_np)
        # Everything the panels are drawn from, cached so the layout can be
        # re-tuned with replot_combined_2d.py instead of retraining.
        np.savez(Path(panel_dir) / "combined_row_data.npz",
                 initial_points=initial_points, final_points=final_points,
                 target_A=A_target_np, u_target=u_target_np,
                 **{f"init_{k}": np.asarray(v) for k, v in initial_snap.items()},
                 **{f"final_{k}": np.asarray(v) for k, v in final_snap.items()},
                 **{f"hist_{k}": np.asarray(v) for k, v in history.items()})

    if make_movie:
        movie_path = Path("examples/foam_2d_orientation_evolution.mp4")
        print(f"Saving movie ({len(snapshot_points)} frames) to {movie_path}...")
        save_evolution_movie(snapshot_points, limits, threads, u_target_np,
                             movie_path, fps=24)
        print(f"Saved movie: {movie_path}")

    print("-" * 75)
    print(f"Initial loss:  {history['loss'][0]:.6g}")
    print(f"Final loss:    {history['loss'][-1]:.6g}")
    print(f"Target A:      diag({alpha_used:.3f}, {1 - alpha_used:.3f}) "
          f"rotated by {angle:g}°")
    print(f"Achieved A:    [[{final_snap['A'][0,0]:.3f}, {final_snap['A'][0,1]:.3f}], "
          f"[{final_snap['A'][1,0]:.3f}, {final_snap['A'][1,1]:.3f}]]")
    print(f"Loop time:     {elapsed:.2f} s")
    print(f"Saved plot:    {summary_path}")

    # Alignment of the final foam, as in the paper.
    with torch.no_grad():
        final_terms = compute_loss(torch.tensor(final_points), limits, threads,
                                   alpha_used, angle, field)
    final_M, final_strength = alignment(
        final_terms["A_cells"], final_terms["u"], alpha_used, final_terms["keep"])
    print(f"Alignment M:   {initial_M:.4f} -> {final_M:.4f}")
    print(f"Mean u^T A_c u: {initial_strength:.4f} -> {final_strength:.4f} "
          f"(target alpha = {alpha_used:g})")

    # Orientation diagnostics. These verify the claim that concentrating wall
    # normals along an axis produces walls PERPENDICULAR to that axis, and
    # cells ELONGATED perpendicular to it -- i.e. that the principal axis of A
    # is perpendicular to the direction of cell elongation.
    normals_np = final_snap["normals"]
    lengths_np = final_snap["lengths"]
    if len(normals_np):
        tot_len = lengths_np.sum()
        root_half = np.sqrt(0.5)
        frac_vertical = float(
            lengths_np[np.abs(normals_np[:, 0]) > root_half].sum() / tot_len)
        frac_horizontal = float(
            lengths_np[np.abs(normals_np[:, 1]) > root_half].sum() / tot_len)
    else:
        frac_vertical = frac_horizontal = float("nan")

    # The extent of each cell in x and y, from its own vertices.
    final_result = voroplex.compute_voronoi_2d(
        final_points, limits, representation="per_cell", threads=threads)
    starts = final_result.cell_vertex_offsets[:-1]
    xy = final_result.vertices
    extent_x = (np.maximum.reduceat(xy[:, 0], starts)
                - np.minimum.reduceat(xy[:, 0], starts))
    extent_y = (np.maximum.reduceat(xy[:, 1], starts)
                - np.minimum.reduceat(xy[:, 1], starts))
    aspect_yx = float((extent_y / np.maximum(extent_x, EPS)).mean())
    print(
        "ORIENTATION_CHECK "
        f"frac_len_vertical_walls={frac_vertical:.4f} "
        f"frac_len_horizontal_walls={frac_horizontal:.4f} "
        f"mean_extent_x={extent_x.mean():.4f} "
        f"mean_extent_y={extent_y.mean():.4f} "
        f"mean_aspect_y_over_x={aspect_yx:.4f}"
    )


if __name__ == "__main__":
    main()
