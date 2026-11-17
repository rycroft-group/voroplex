"""
VoroPlex example: Lloyd's algorithm with representation=None
=============================================================

Lloyd's algorithm computes a centroidal Voronoi tessellation (CVT) by moving
each generator to the centroid of its cell, over and over. Each iteration only
needs the centroids, so the tessellation is computed with
representation=None, which stores nothing about the cells except the
requested properties. Here, those are the centroids, and the second moments,
whose sum is the CVT energy

    E = sum_i int_{V_i} |x - p_i|^2 dx.

The energy is shown relative to that of a hexagonal lattice, which is the
lowest possible energy in two dimensions: a regular hexagon of area A has
int |x - p|^2 dx = 5 A^2 / (18 sqrt(3)). The domain is periodic, so a
hexagonal pattern can form without a boundary. In a periodic domain, a centroid can
lie slightly outside the box, so it is wrapped back in.

Usage:
    python 12_lloyd_algorithm.py [--n 10000] [--iterations 100] [--plot]
"""
import argparse
import time

import numpy as np

import voroplex


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n", type=int, default=10000, help="number of generators")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--plot", action="store_true",
                        help="plot the final tessellation (needs matplotlib)")
    args = parser.parse_args()

    n = args.n
    limits = [[0.0, 1.0], [0.0, 1.0]]
    points = np.random.default_rng(0).random((n, 2))
    # The energy of n regular hexagons of area 1/n, the lowest possible value.
    e_hex = 5.0 / (18.0 * np.sqrt(3.0) * n)

    t0 = time.perf_counter()
    for it in range(args.iterations + 1):
        # Compute only the centroid c_i of each cell and its second moment
        # about the generator, int_{V_i} |x - p_i|^2 dx. With
        # representation=None, nothing else about the cells is stored.
        result = voroplex.compute_voronoi_2d(
            points, limits, periodic=True, representation=None,
            properties=("centroid", "second_moment"),
        )
        # The CVT energy is the sum of the second moments. Each Lloyd step
        # lowers it, since moving p_i to c_i minimizes the second moment of
        # each cell, and then recomputing the cells lowers it further.
        energy = result.second_moments.sum()
        if it % 10 == 0:
            print(f"iteration {it:4d}: E / E_hex = {energy / e_hex:.5f}")
        # The Lloyd step: move each generator to the centroid of its cell.
        # A centroid can lie slightly outside the periodic box, so it is
        # wrapped back in.
        points = np.mod(result.centroids, 1.0)
    elapsed = time.perf_counter() - t0
    print(f"{args.iterations} iterations of {n} generators in {elapsed:.2f} s")

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        # The final tessellation, with each cell's vertices and edges. Each
        # edge is drawn from its first vertex to its second; an edge shared by
        # two cells is drawn twice.
        geometry = voroplex.compute_voronoi_2d(points, limits, periodic=True)
        owners = np.repeat(np.arange(len(geometry)),
                           np.diff(geometry.cell_facet_offsets))
        start = geometry.facet_vertex_offsets[:-1]
        base = geometry.cell_vertex_offsets[owners]
        edges = np.stack([geometry.vertices[base + geometry.facet_vertices[start]],
                          geometry.vertices[base + geometry.facet_vertices[start + 1]]],
                         axis=1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_collection(LineCollection(edges, colors="k", linewidths=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"Lloyd's algorithm, {args.iterations} iterations")
        fig.savefig("lloyd_algorithm.png", dpi=200)
        print("Saved lloyd_algorithm.png")


if __name__ == "__main__":
    main()
