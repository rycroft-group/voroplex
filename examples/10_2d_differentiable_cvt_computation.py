"""
2D CVT Voronoi Optimization with VoroPlex + PyTorch
====================================================

Optimizes random 2D generator points toward a Centroidal Voronoi
Tessellation (CVT), by gradient descent on the quantization energy

    E = sum_i  integral_{cell_i}  rho(x) * ||x - p_i||^2  dA

Two density modes are provided -- these are the only two cases where the
energy above is computed *correctly*, and the only two modes this example
keeps:

  --density uniform   (default)
      rho(x) = 1. The integral has an exact closed form: fan-triangulate
      each polygon cell from the generator to every edge, and sum each
      triangle's exact second moment about the generator. No
      approximation, no sampling.

  --density adaptive
      rho(x) is an arbitrary user function (here, a Gaussian bump so
      points cluster where density is higher). No closed form exists for
      an arbitrary density, so the integral is evaluated with a
      numerical quadrature rule on each fan-triangulated triangle (a
      3-point, degree-2-exact rule) instead.

Each iteration computes the tessellation with the per-cell representation,
gets the vertices of every cell as a differentiable tensor
(voroplex.cell_vertices), and splits every cell into triangles that share
its generator, one per edge (voroplex.cell_fan_simplices). VoroPlex computes
the derivative of each vertex with respect to the generators that define it,
so the gradient of the energy flows back to the generator positions.

Run from the project root:
    python examples/10_2d_differentiable_cvt_computation.py

Options:
    --density {uniform,adaptive}   energy mode (default: uniform)
    --threads N                    OpenMP threads (default: 4)
    --device {cpu,cuda}            computation device (default: cpu)
    --points N                     number of generators (default: 80)
    --iters N                      optimization iterations (default: 300)
    --lr LR                        Adam learning rate (default: 0.01)
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

import voroplex

EXAMPLES_DIR = Path(__file__).resolve().parent


# 3-point, degree-2-exact quadrature rule for a triangle, in barycentric
# coordinates over its three corners (generator, v0, v1). All three points
# carry equal weight (1/3 of the triangle's area).
TRI_QUADRATURE_BARYCENTRIC = torch.tensor(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=torch.float64,
)


def uniform_cvt_energy_2d(points, vertices, tri_cells, tri_vertices):
    """Exact uniform-density (rho=1) CVT energy of a triangular fan.

    For a triangle with one corner at the generator p and edge vectors
    q1, q2 to its other two corners, the second moment about p is
    (Area/6) * (q1.q1 + q2.q2 + q1.q2) -- exact, no quadrature needed.
    Area = (q1 x q2)/2 is the signed area, which is positive for the
    triangles from cell_fan_simplices whenever the generator is inside its
    cell.
    """
    generators = points[tri_cells]
    edges = vertices[tri_vertices]

    q1 = edges[:, 0] - generators
    q2 = edges[:, 1] - generators

    area = 0.5 * (q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0])
    quadratic_term = (
        torch.sum(q1 * q1, dim=1) + torch.sum(q2 * q2, dim=1) + torch.sum(q1 * q2, dim=1)
    )
    return torch.sum(area * quadratic_term / 6.0)


def gaussian_bump_density(x, center, sigma=0.12, base=1.0, amplitude=20.0):
    """A smooth density with no closed-form CVT integral: higher near `center`."""
    sq_dist = ((x - center) ** 2).sum(dim=-1)
    return base + amplitude * torch.exp(-sq_dist / (2.0 * sigma * sigma))


def adaptive_cvt_energy_2d(points, vertices, tri_cells, tri_vertices, density_fn):
    """Quadrature CVT energy for an arbitrary density rho(x) = density_fn(x).

    No closed form exists for a general density, so each fan triangle's
    integral is approximated with the 3-point degree-2 rule above: sample
    the integrand at three fixed barycentric combinations of the
    triangle's corners, weight equally, and scale by the triangle's
    (exact, differentiable) signed area.
    """
    generators = points[tri_cells]
    edges = vertices[tri_vertices]
    corners = torch.cat([generators.unsqueeze(1), edges], dim=1)  # (T, 3, 2)

    q1 = edges[:, 0] - generators
    q2 = edges[:, 1] - generators
    area = 0.5 * (q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0])

    bary = TRI_QUADRATURE_BARYCENTRIC.to(device=corners.device, dtype=corners.dtype)
    quad_points = torch.einsum("qk,tkd->tqd", bary, corners)  # (T, 3, 2)

    sq_dist = ((quad_points - generators.unsqueeze(1)) ** 2).sum(dim=-1)  # (T, 3)
    integrand = density_fn(quad_points) * sq_dist
    return torch.sum(area * integrand.mean(dim=1))


def tessellation_snapshot(points, limits, threads):
    """The edges of every cell, as line segments, for plotting.

    Uses the default geometry representation: each cell's vertices as
    computed by Voro++. An edge shared by two cells is drawn twice.
    """
    result = voroplex.compute_voronoi_2d(points, limits, threads=threads)
    owners = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    ends = (result.facet_vertices.reshape(-1, 2)
            + result.cell_vertex_offsets[owners][:, None])
    return result.vertices[ends]                          # (n_edges, 2, 2)


def draw_tessellation(ax, points, segments, title):
    ax.add_collection(LineCollection(segments, colors="0.72", linewidths=0.75, zorder=1))
    ax.scatter(points[:, 0], points[:, 1], s=12, color="#d83b3b", edgecolor="white",
               linewidth=0.35, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def save_plot(initial_points, final_points, initial_mesh, final_mesh,
              loss_history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    draw_tessellation(axes[0], initial_points, initial_mesh, "Initial")
    draw_tessellation(axes[1], final_points, final_mesh, "After CVT Steps")

    loss_values = np.asarray(loss_history, dtype=np.float64)
    positive = loss_values[loss_values > 0]
    ymin = max(positive.min() * 0.8, np.finfo(np.float64).tiny) if len(positive) else 1e-16

    axes[2].semilogy(np.arange(1, len(loss_values) + 1), loss_values, color="#2454a6", linewidth=2)
    axes[2].set_title("CVT Energy")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("CVT energy")
    axes[2].set_ylim(bottom=ymin)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", choices=["uniform", "adaptive"], default="uniform")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--points", type=int, default=80)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    limits = [[0, 1], [0, 1]]

    # The generator positions are the variables of the optimization, so they
    # are a PyTorch tensor that records gradients.
    rng = np.random.default_rng(42)
    initial_points = rng.random((args.points, 2)) * 0.9 + 0.05
    points_t = torch.tensor(
        initial_points, dtype=torch.float64, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([points_t], lr=args.lr)

    center = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)

    def energy_fn(points, vertices, tri_cells, tri_vertices):
        if args.density == "uniform":
            return uniform_cvt_energy_2d(points, vertices, tri_cells, tri_vertices)
        def density_fn(x):
            return gaussian_bump_density(x, center)

        return adaptive_cvt_energy_2d(points, vertices, tri_cells, tri_vertices, density_fn)

    initial_mesh = tessellation_snapshot(initial_points, limits, args.threads)

    loss_history = []
    print(
        f"2D CVT optimization -- {args.points} points, {args.iters} iterations, "
        f"{args.threads} thread(s), density={args.density}, device={device}"
    )
    print(f"{'Iter':>4}  {'CVT energy':>16}")
    print("-" * 24)

    t0 = time.perf_counter()
    for it in range(args.iters):
        # Compute the tessellation of the current generators with Voro++. The
        # cells change as the generators move, so this is done every time.
        points_np = points_t.detach().cpu().numpy()
        result = voroplex.compute_voronoi_2d(
            points_np, limits, threads=args.threads, representation="per_cell",
        )

        # Each cell's vertices, as a tensor that is a differentiable function
        # of points_t, and a split of every cell into triangles, each made from
        # the generator and one edge of the cell.
        vertices = voroplex.cell_vertices(result, points_t)
        tri_cells, tri_vertices = voroplex.cell_fan_simplices(result, device=device)

        # E = sum_i int_{V_i} rho(x) |x - p_i|^2 dA, as a sum over triangles.
        energy = energy_fn(points_t, vertices, tri_cells, tri_vertices)

        # The backward pass finds dE/dp_i. PyTorch differentiates E with
        # respect to the vertices, and VoroPlex converts this into derivatives
        # with respect to the generators that define each vertex. Adam then
        # moves the generators downhill.
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        # Keep the generators inside the square.
        with torch.no_grad():
            points_t.clamp_(1e-4, 1.0 - 1e-4)

        loss_history.append(energy.item())
        if it == 0 or (it + 1) % 30 == 0 or it + 1 == args.iters:
            print(f"{it + 1:>4}  {energy.item():>16.8e}")

    elapsed = time.perf_counter() - t0
    final_points = points_t.detach().cpu().numpy()
    final_mesh = tessellation_snapshot(final_points, limits, args.threads)

    output_path = EXAMPLES_DIR / f"cvt_2d_optimization_{args.density}.png"
    save_plot(initial_points, final_points, initial_mesh, final_mesh, loss_history, output_path)

    print("-" * 24)
    print(f"Initial energy: {loss_history[0]:.8e}")
    print(f"Final energy:   {loss_history[-1]:.8e}")
    print(f"Loop time:      {elapsed:.2f} s")
    print(f"Saved plot:     {output_path}")


if __name__ == "__main__":
    main()
