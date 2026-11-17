"""
CVT 3D Voronoi Optimization with VoroPlex + PyTorch
====================================================

Optimizes N random 3D generator points toward a Centroidal Voronoi
Tessellation (CVT), by gradient descent on the quantization energy

    E = sum_i  integral_{cell_i}  rho(x) * ||x - p_i||^2  dV

Two density modes are provided -- these are the only two cases where the
energy above is computed *correctly*, and the only two modes this example
keeps:

  --density uniform   (default)
      rho(x) = 1. The integral has an exact closed form: decompose each
      cell into tetrahedra by fan-triangulating every face from the
      generator, and sum each tetrahedron's exact second moment about the
      generator. No approximation, no sampling.

  --density adaptive
      rho(x) is an arbitrary user function (here, a Gaussian bump so
      points cluster where density is higher). No closed form exists for
      an arbitrary density, so the integral is evaluated with a
      numerical quadrature rule on each fan-triangulated tetrahedron (a
      4-point, degree-2-exact rule) instead.

Each iteration:
  1. Compute the tessellation of the current points, with the per-cell
     representation (one parallel C++ call)
  2. Get the vertices of every cell as a differentiable tensor
     (voroplex.cell_vertices), and split every cell into tetrahedra that
     share its generator (voroplex.cell_fan_simplices)
  3. Evaluate the CVT energy (exact or quadrature, per --density)
  4. Take a gradient step. VoroPlex computes the derivative of each vertex
     with respect to the generators that define it, so the gradient of the
     energy flows back to the generator positions

Usage:
  python examples/07_differentiable_cvt_computation.py [options]

Options:
  --density {uniform,adaptive}   energy mode (default: uniform)
  --threads N                    OpenMP threads (default: 8)
  --device {cpu,cuda}            computation device (default: cpu)
  --points N                     number of generators (default: 500)
  --iters N                      optimization iterations (default: 100)
  --lr LR                        Adam learning rate (default: 1e-3)
  --vtu                          also export cvt_initial.vtu / cvt_final.vtu
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import voroplex

EXAMPLES_DIR = Path(__file__).resolve().parent


# 4-point, degree-2-exact quadrature rule for a tetrahedron, in barycentric
# coordinates over its four corners (generator, t0, t1, t2). All four points
# carry equal weight (1/4 of the tetrahedron's volume) -- see e.g. Keast
# (1986) for the standard reference rule.
_TET_QUAD_A = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
_TET_QUAD_B = (5.0 - math.sqrt(5.0)) / 20.0
TET_QUADRATURE_BARYCENTRIC = torch.tensor(
    [
        [_TET_QUAD_A, _TET_QUAD_B, _TET_QUAD_B, _TET_QUAD_B],
        [_TET_QUAD_B, _TET_QUAD_A, _TET_QUAD_B, _TET_QUAD_B],
        [_TET_QUAD_B, _TET_QUAD_B, _TET_QUAD_A, _TET_QUAD_B],
        [_TET_QUAD_B, _TET_QUAD_B, _TET_QUAD_B, _TET_QUAD_A],
    ],
    dtype=torch.float64,
)


def uniform_cvt_energy(points, vertices, tetrahedron_cells, tetrahedron_vertices):
    """Exact uniform-density (rho=1) CVT energy of a tetrahedral decomposition.

    Each tetrahedron has the generator points[tetrahedron_cells] as one
    corner, and vertices[tetrahedron_vertices] as the other three, as
    returned by voroplex.cell_fan_simplices. For a tetrahedron with one
    corner at the generator p and edge vectors q1, q2, q3 to its other three
    corners, the second moment about p is
    (V/10) * (q1.q1 + q2.q2 + q3.q3 + q1.q2 + q1.q3 + q2.q3) -- exact, no
    quadrature needed. V = q1.(q2 x q3)/6 is the signed volume, which is
    positive for the tetrahedra from cell_fan_simplices whenever the
    generator is inside its cell.
    """
    generators = points[tetrahedron_cells]
    triangles = vertices[tetrahedron_vertices]

    q1 = triangles[:, 0] - generators
    q2 = triangles[:, 1] - generators
    q3 = triangles[:, 2] - generators

    volumes = torch.sum(q1 * torch.cross(q2, q3, dim=1), dim=1) / 6.0
    quadratic_term = (
        torch.sum(q1 * q1, dim=1)
        + torch.sum(q2 * q2, dim=1)
        + torch.sum(q3 * q3, dim=1)
        + torch.sum(q1 * q2, dim=1)
        + torch.sum(q1 * q3, dim=1)
        + torch.sum(q2 * q3, dim=1)
    )
    return torch.sum(volumes * quadratic_term / 10.0)


def gaussian_bump_density(x, center, sigma=0.12, base=1.0, amplitude=20.0):
    """A smooth density with no closed-form CVT integral: higher near `center`."""
    sq_dist = ((x - center) ** 2).sum(dim=-1)
    return base + amplitude * torch.exp(-sq_dist / (2.0 * sigma * sigma))


def adaptive_cvt_energy(
    points, vertices, tetrahedron_cells, tetrahedron_vertices, density_fn
):
    """Quadrature CVT energy for an arbitrary density rho(x) = density_fn(x).

    No closed form exists for a general density, so each tetrahedron's
    integral is approximated with the 4-point degree-2 rule above: sample
    the integrand at four fixed barycentric combinations of the
    tetrahedron's corners, weight equally, and scale by the tetrahedron's
    (exact, differentiable) volume.
    """
    generators = points[tetrahedron_cells]
    triangles = vertices[tetrahedron_vertices]
    corners = torch.cat([generators.unsqueeze(1), triangles], dim=1)  # (T, 4, 3)

    q1 = triangles[:, 0] - generators
    q2 = triangles[:, 1] - generators
    q3 = triangles[:, 2] - generators
    volumes = torch.sum(q1 * torch.cross(q2, q3, dim=1), dim=1) / 6.0

    bary = TET_QUADRATURE_BARYCENTRIC.to(device=corners.device, dtype=corners.dtype)
    quad_points = torch.einsum("qk,tkd->tqd", bary, corners)  # (T, 4, 3)

    sq_dist = ((quad_points - generators.unsqueeze(1)) ** 2).sum(dim=-1)  # (T, 4)
    integrand = density_fn(quad_points) * sq_dist  # (T, 4), all weights equal 1/4
    return torch.sum(volumes * integrand.mean(dim=1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", choices=["uniform", "adaptive"], default="uniform")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vtu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    limits = [[-0.05, 1.05], [-0.05, 1.05], [-0.05, 1.05]]

    # The generator positions are the variables of the optimization, so they
    # are a PyTorch tensor that records gradients.
    rng = np.random.default_rng(42)
    points_t = torch.tensor(
        rng.random((args.points, 3)), dtype=torch.float64,
        device=device, requires_grad=True,
    )
    optimizer = torch.optim.Adam([points_t], lr=args.lr)

    center = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64, device=device)

    def energy_fn(points, vertices, tet_cells, tet_vertices):
        if args.density == "uniform":
            return uniform_cvt_energy(points, vertices, tet_cells, tet_vertices)
        def density_fn(x):
            return gaussian_bump_density(x, center)

        return adaptive_cvt_energy(
            points, vertices, tet_cells, tet_vertices, density_fn
        )

    if args.vtu:
        result_vis = voroplex.compute_voronoi_3d(
            points_t.detach().cpu().numpy(), limits, threads=args.threads,
        )
        vtu_path = EXAMPLES_DIR / "cvt_initial.vtu"
        voroplex.export_vtu(result_vis, points_t.detach().cpu().numpy(), str(vtu_path))
        print(f"Saved {vtu_path}")

    print(
        f"CVT optimization -- {args.points} points, {args.iters} iterations, "
        f"{args.threads} thread(s), density={args.density}, device={device}"
    )
    print(f"{'Iter':>4}  {'CVT energy':>16}")
    print("-" * 24)

    loss_history = []
    t0 = time.perf_counter()

    for it in range(args.iters):
        # Compute the tessellation of the current generators with Voro++. The
        # cells change as the generators move, so this is done every time.
        pts_np = points_t.detach().cpu().numpy()
        result = voroplex.compute_voronoi_3d(
            pts_np, limits, threads=args.threads, representation="per_cell",
        )

        # Each cell's vertices, as a tensor that is a differentiable function
        # of points_t, and a split of every cell into tetrahedra, each with
        # the generator as one corner and a triangle of one face as the base.
        vertices = voroplex.cell_vertices(result, points_t)
        tet_cells, tet_vertices = voroplex.cell_fan_simplices(result, device=device)

        # E = sum_i int_{V_i} rho(x) |x - p_i|^2 dV, as a sum over tetrahedra.
        energy = energy_fn(points_t, vertices, tet_cells, tet_vertices)

        # The backward pass finds dE/dp_i. PyTorch differentiates E with
        # respect to the vertices, and VoroPlex converts this into derivatives
        # with respect to the generators that define each vertex. Adam then
        # moves the generators downhill.
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        # Keep the generators inside the cube.
        with torch.no_grad():
            points_t.clamp_(0.0, 1.0)

        loss_history.append(energy.item())
        if it == 0 or (it + 1) % 10 == 0:
            print(f"{it + 1:>4}  {energy.item():>16.8e}")

    t_total = time.perf_counter() - t0
    print("-" * 24)
    print(f"Done. Final generator positions: {tuple(points_t.shape)}")
    print(f"\nTiming ({args.threads} thread(s)):")
    print(f"  Total loop time : {t_total:.2f} s")
    print(f"  Per iteration   : {t_total / args.iters * 1000:.1f} ms")

    output_path = EXAMPLES_DIR / f"cvt_loss_{args.density}.png"
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.iters + 1), loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("CVT energy")
    plt.title(f"CVT Optimization ({args.density} density) -- {args.points} points")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")

    if args.vtu:
        result_vis = voroplex.compute_voronoi_3d(
            points_t.detach().cpu().numpy(), limits, threads=args.threads,
        )
        vtu_path = EXAMPLES_DIR / "cvt_final.vtu"
        voroplex.export_vtu(result_vis, points_t.detach().cpu().numpy(), str(vtu_path))
        print(f"Saved {vtu_path}")


if __name__ == "__main__":
    main()
