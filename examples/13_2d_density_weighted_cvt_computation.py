"""
2D Density-Field CVT Optimization with VoroPlex + PyTorch
========================================================

Optimizes 2D generators for a density field rho(x), which is the weight in the
centroidal Voronoi tessellation (CVT) energy

    E = sum_i integral_{cell_i} rho(x) ||x - p_i||^2 dx.

At each iteration, VoroPlex computes the tessellation with the per-cell
representation. voroplex.cell_vertices gives the vertices of each cell as a
PyTorch tensor whose gradient flows back to the generators, and
voroplex.cell_fan_simplices splits each cell into triangles that share its
generator as a corner. The energy is integrated over each triangle with a
three-point quadrature rule that is exact for quadratic integrands, so it is
exact for a uniform density.

Use --density=name to give rho directly, or --sizing=name to give a target
relative spacing h, which is converted to rho = h^(-4): for a 2D CVT, the cell
length scale is proportional to rho^(-1/4), so cells of size h need
rho = h^(-(d+2)). Add a field to FIELDS to use your own.

Run from the project root:
    python examples/13_2d_density_weighted_cvt_computation.py

Optional arguments:
    python examples/13_2d_density_weighted_cvt_computation.py \
        [num_threads] [num_points] [num_iters] [lr]
    python examples/13_2d_density_weighted_cvt_computation.py --density=two_blobs
    python examples/13_2d_density_weighted_cvt_computation.py --sizing=linear_xy_moderate
    python examples/13_2d_density_weighted_cvt_computation.py --device=auto   # or cpu, cuda, mps
    python examples/13_2d_density_weighted_cvt_computation.py --gif-frames=80 --gif-fps=12
    python examples/13_2d_density_weighted_cvt_computation.py --no-gif --no-vtu

The three 2D examples in Section 5.1 of the VoroPlex paper:
    python examples/13_2d_density_weighted_cvt_computation.py \
        --sizing=linear_xy_moderate --num-points=400 --num-iters=400 --lr=0.005
    python examples/13_2d_density_weighted_cvt_computation.py \
        --density=two_blobs --num-points=400 --num-iters=400 --lr=0.005
    python examples/13_2d_density_weighted_cvt_computation.py \
        --density=five_petal --num-points=400 --num-iters=400 --lr=0.005

Available fields (each can be used with --density or --sizing):
    uniform, gaussian, two_blobs, left_heavy, ring, linear_xy,
    linear_xy_moderate, linear_density, five_petal
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

import voroplex


DEFAULT_FIELD = "gaussian"
GIF_FRAME_COUNT = 60
GIF_FPS = 12
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


# ---------------------------------------------------------------------------
# Fields. Each takes points of shape (..., 2) and returns values of shape (...).
# With --density, the value is rho; with --sizing, it is h and rho = h^(-4).
# ---------------------------------------------------------------------------

def _gaussian(xy, center, sigma):
    c = torch.tensor(center, dtype=xy.dtype, device=xy.device)
    return torch.exp(-((xy - c) ** 2).sum(dim=-1) / (2.0 * sigma ** 2))


def field_uniform(xy):
    return torch.ones_like(xy[..., 0])


def field_gaussian(xy):
    return 0.15 + 3.0 * _gaussian(xy, (0.68, 0.62), 0.16)


def field_two_blobs(xy):
    """Two Gaussian peaks, as in Section 5.1 of the VoroPlex paper."""
    return (0.12 + 4.8 * _gaussian(xy, (0.30, 0.72), 0.10)
            + 3.6 * _gaussian(xy, (0.74, 0.32), 0.13))


def field_left_heavy(xy):
    return 0.2 + 2.5 * (1.0 - xy[..., 0]) ** 2


def field_ring(xy):
    radius = torch.linalg.norm(xy - 0.5, dim=-1)
    return 0.15 + 2.6 * torch.exp(-((radius - 0.28) ** 2) / (2.0 * 0.055 ** 2))


def field_linear_xy(xy):
    return 1.0 + 3.0 * xy[..., 0] + 3.0 * xy[..., 1]


def field_linear_xy_moderate(xy):
    """h = 1 + 4x + 4y, used with --sizing in Section 5.1 of the VoroPlex paper."""
    return 1.0 + 4.0 * xy[..., 0] + 4.0 * xy[..., 1]


def field_linear_density(xy):
    return 1.0 + xy[..., 0] + xy[..., 1]


def field_five_petal(xy):
    """A band around a five-petal curve, as in Section 5.1 of the VoroPlex paper."""
    rel = xy - 0.5
    theta = torch.atan2(rel[..., 1], rel[..., 0])
    radius = torch.linalg.norm(rel, dim=-1)
    petal_radius = 0.30 + 0.075 * torch.cos(5.0 * theta + 0.35)
    return 0.10 + 3.0 * torch.exp(-((radius - petal_radius) ** 2) / (2.0 * 0.045 ** 2))


FIELDS = {
    "uniform": field_uniform,
    "gaussian": field_gaussian,
    "two_blobs": field_two_blobs,
    "left_heavy": field_left_heavy,
    "ring": field_ring,
    "linear_xy": field_linear_xy,
    "linear_xy_moderate": field_linear_xy_moderate,
    "linear_density": field_linear_density,
    "five_petal": field_five_petal,
}
# Earlier names, kept so that old commands still work.
FIELDS["two_gaussian_density"] = field_two_blobs
FIELDS["five_petal_density"] = field_five_petal


def density_function(name, is_sizing):
    try:
        field = FIELDS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(FIELDS))
        raise SystemExit(f"Unknown field '{name}'. Choose one of: {choices}") from exc
    if not is_sizing:
        return field
    return lambda xy: field(xy).clamp_min(1e-12).pow(-4.0)


# ---------------------------------------------------------------------------
# The density-weighted CVT energy, from the per-cell vertices
# ---------------------------------------------------------------------------

# Three-point degree-2 rule on a triangle, in barycentric coordinates.
TRIANGLE_RULE = torch.tensor(
    [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
     [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
     [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]],
    dtype=torch.float64,
)


def cell_terms(result, points_t, density_fn):
    """The energy, mass, and density-weighted centroid of every cell.

    Each triangle from cell_fan_simplices has the generator p_i as one corner
    and two vertices of the cell as the others. Its signed area is positive,
    and the integrals over it are found with the three-point rule.
    """
    device = points_t.device
    vertices = voroplex.cell_vertices(result, points_t)
    owners, simplices = voroplex.cell_fan_simplices(result, device=device)
    a = points_t[owners]
    b, c = vertices[simplices[:, 0]], vertices[simplices[:, 1]]
    ab, ac = b - a, c - a
    area = 0.5 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    bary = TRIANGLE_RULE.to(device=device, dtype=points_t.dtype)
    corners = torch.stack([a, b, c], dim=1)                    # (T, 3, 2)
    samples = torch.einsum("qk,tkd->tqd", bary, corners)        # (T, 3, 2)
    weights = area[:, None] / 3.0 * density_fn(samples).clamp_min(1e-12)
    energy = (weights * (samples - a[:, None]).square().sum(-1)).sum(-1)
    moment = (weights[..., None] * samples).sum(1)
    n = points_t.shape[0]
    energies = points_t.new_zeros(n).index_add(0, owners, energy)
    masses = points_t.new_zeros(n).index_add(0, owners, weights.sum(-1))
    moments = points_t.new_zeros((n, 2)).index_add(0, owners, moment)
    return energies, masses, moments / masses.clamp_min(1e-300)[:, None]


# ---------------------------------------------------------------------------
# Plots and output
# ---------------------------------------------------------------------------

def tessellation(points, limits, threads):
    """The tessellation of the points, and its edges (each listed once)."""
    result = voroplex.compute_voronoi_2d(points, limits, threads=threads)
    owners = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    keep = (result.facet_neighbors < 0) | (owners < result.facet_neighbors)
    start = result.facet_vertex_offsets[:-1][keep]
    base = result.cell_vertex_offsets[owners[keep]]
    v = result.vertices
    edges = np.stack([v[base + result.facet_vertices[start]],
                      v[base + result.facet_vertices[start + 1]]], axis=1)
    return result, edges


def draw_density(ax, density_fn, resolution=180):
    xs = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    density = density_fn(torch.stack([xx, yy], dim=-1)).detach().cpu().numpy()
    ax.imshow(density, extent=(0, 1, 0, 1), origin="lower", cmap="viridis",
              alpha=0.82, zorder=0)


def draw_tessellation(ax, points, edges, title, density_fn):
    draw_density(ax, density_fn)
    ax.add_collection(LineCollection(edges, colors="white", linewidths=0.55,
                                     alpha=0.85, zorder=1))
    ax.scatter(points[:, 0], points[:, 1], s=13, color="#ff4b4b",
               edgecolor="white", linewidth=0.35, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def save_plot(initial_points, final_points, initial_edges, final_edges,
              loss_history, output_path, field_name, density_fn):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    draw_tessellation(axes[0], initial_points, initial_edges, "Initial", density_fn)
    draw_tessellation(axes[1], final_points, final_edges, "Density-field CVT", density_fn)
    axes[2].semilogy(np.arange(1, len(loss_history) + 1), loss_history,
                     color="#2454a6", linewidth=2)
    axes[2].set_title("Weighted CVT energy")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Density-weighted CVT energy")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle(f"Field: {field_name}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gif(frames, limits, threads, output_path, field_name, density_fn, fps):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    def update(index):
        iteration, points, loss_value = frames[index]
        ax.clear()
        _, edges = tessellation(points, limits, threads)
        title = f"Density-field CVT - iter {iteration}"
        if loss_value is not None:
            title += f", energy {loss_value:.4g}"
        draw_tessellation(ax, points, edges, title, density_fn)
        return ax.collections + ax.images

    animation = FuncAnimation(fig, update, frames=len(frames),
                              interval=1000.0 / fps, blit=False, repeat=True)
    fig.suptitle(f"Field: {field_name}", y=0.98)
    fig.tight_layout()
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def export_mesh(result, points, output_prefix):
    """Write a VTU file for ParaView, or an NPZ archive if VTK is unavailable."""
    try:
        path = f"{output_prefix}.vtu"
        voroplex.export_vtu(result, torch.tensor(points), path, facet_type="all")
    except ImportError as exc:
        path = f"{output_prefix}.npz"
        voroplex.export_npz(result, path, points=points)
        print(f"VTU export skipped: {exc}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    threads = parse_option("threads", parse_int_arg(0, 4), int)
    n_points = parse_option("num-points", parse_int_arg(1, 80), int)
    n_iter = parse_option("num-iters", parse_int_arg(2, 300), int)
    lr = parse_option("lr", parse_float_arg(3, 0.01), float)
    field_name, is_sizing = parse_field()
    device = resolve_device(parse_option("device", "auto").lower())
    dtype = torch.float32 if device.type == "mps" else torch.float64
    gif_frame_count = max(2, parse_option("gif-frames", GIF_FRAME_COUNT, int))
    gif_fps = max(1.0, parse_option("gif-fps", GIF_FPS, float))
    make_gif = "--no-gif" not in sys.argv
    make_vtu = "--no-vtu" not in sys.argv
    output_dir = Path(parse_option("output-dir", os.environ.get("VOROPLEX_OUTPUT_DIR", "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    density_fn = density_function(field_name, is_sizing)
    limits = [[0.0, 1.0], [0.0, 1.0]]

    torch.manual_seed(SEED)
    initial_points = np.random.default_rng(SEED).random((n_points, 2)) * 0.9 + 0.05
    points_t = torch.tensor(initial_points, dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([points_t], lr=lr)

    loss_history, residual_history = [], []
    frames = [(0, initial_points.copy(), None)]
    frame_stride = max(1, n_iter // (gif_frame_count - 1))

    kind = "sizing" if is_sizing else "density"
    print(f"2D density-field CVT - {n_points} points, {n_iter} iterations, {threads} thread(s)")
    print(f"Field: {field_name} ({kind}), lr={lr}, device={device}, dtype={dtype}")
    print(f"{'Iter':>4}  {'Energy':>14}  {'Mean mass':>12}  {'Mean residual':>14}")
    print("-" * 52)

    t0 = time.perf_counter()
    for it in range(n_iter):
        result = voroplex.compute_voronoi_2d(
            points_t.detach().cpu().numpy(), limits, threads=threads,
            representation="per_cell", properties=(),
        )
        energies, masses, centroids = cell_terms(result, points_t, density_fn)
        loss = energies.sum()

        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            residuals = torch.linalg.norm(points_t - centroids, dim=1)
        optimizer.step()
        with torch.no_grad():
            points_t.clamp_(1e-4, 1.0 - 1e-4)

        loss_history.append(float(loss.item()))
        residual_history.append(float(residuals.mean().item()))
        if (it + 1) % frame_stride == 0 or it + 1 == n_iter:
            frames.append((it + 1, points_t.detach().cpu().numpy().copy(), loss_history[-1]))
        if it == 0 or (it + 1) % 10 == 0 or it + 1 == n_iter:
            print(f"{it + 1:>4}  {loss_history[-1]:>14.8f}  "
                  f"{masses.mean().item():>12.8f}  {residual_history[-1]:>14.8f}")
    elapsed = time.perf_counter() - t0

    final_points = points_t.detach().cpu().numpy().astype(np.float64)
    _, initial_edges = tessellation(initial_points, limits, threads)
    final_result, final_edges = tessellation(final_points, limits, threads)

    stem = output_dir / f"cvt_2d_{kind}_{field_name}"
    plot_path = f"{stem}.png"
    save_plot(initial_points, final_points, initial_edges, final_edges,
              loss_history, plot_path, field_name, density_fn)
    gif_path = None
    if make_gif:
        gif_path = f"{stem}.gif"
        save_gif(frames, limits, threads, gif_path, field_name, density_fn, gif_fps)
    mesh_path = export_mesh(final_result, final_points, f"{stem}_final_mesh") if make_vtu else None

    params_path = f"{stem}_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "script": Path(__file__).name,
            "argv": sys.argv[1:],
            "voroplex_version": voroplex.__version__,
            "seed": SEED,
            "threads": threads,
            "num_points": n_points,
            "num_iters": n_iter,
            "learning_rate": lr,
            "device": str(device),
            "dtype": str(dtype),
            "field": field_name,
            "field_input": kind,
            "sizing_to_density": "rho = h^(-4)" if is_sizing else None,
            "limits": limits,
            "initial_energy": loss_history[0],
            "final_energy": loss_history[-1],
            "initial_residual": residual_history[0],
            "final_residual": residual_history[-1],
            "loop_time_seconds": elapsed,
            "output_plot": plot_path,
            "output_gif": gif_path,
            "output_mesh": mesh_path,
            "initial_points": initial_points.tolist(),
            "final_points": final_points.tolist(),
            "energy_history": loss_history,
            "residual_history": residual_history,
        }, f, indent=2)

    print("-" * 52)
    print(f"Initial energy:   {loss_history[0]:.8f}")
    print(f"Final energy:     {loss_history[-1]:.8f}")
    print(f"Initial residual: {residual_history[0]:.8f}")
    print(f"Final residual:   {residual_history[-1]:.8f}")
    print(f"Loop time:        {elapsed:.2f} s")
    print(f"Saved plot:       {plot_path}")
    if gif_path:
        print(f"Saved GIF:        {gif_path}")
    if mesh_path:
        print(f"Saved mesh:       {mesh_path}")
    print(f"Saved params:     {params_path}")


if __name__ == "__main__":
    main()
