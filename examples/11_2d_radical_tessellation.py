"""
VoroPlex 2D radical tessellation example
========================================

Compares the standard Voronoi tessellation and the radical (power)
tessellation of a packing of disks with different radii. In the radical
tessellation, the cell of generator i is the region where
|x - p_i|^2 - r_i^2 is smallest, so the cell faces lie between the disks,
rather than cutting through the larger ones.

The last panel increases every radius by the same amount t, so that the
disks overlap. The power distance then contains the term -2 t r_i, which
favors the larger disks, so their cells grow and the cells of some small
disks, which lie entirely inside larger disks, become empty (orange). The
face between two overlapping disks passes through the points where their
circles intersect. As t increases further, the largest disk eventually takes
over the whole domain.

Usage:
    python 11_2d_radical_tessellation.py [--output radical_tessellation.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle

import voroplex


def disk_packing(n_attempts=20000, r_min=0.012, r_max=0.07, seed=3):
    """Random sequential addition of non-overlapping disks in the unit square.

    Radii are drawn first and sorted from largest to smallest; each disk is
    then placed at the first random position where it overlaps no other disk.
    """
    rng = np.random.default_rng(seed)
    radii = np.sort(r_min * (r_max / r_min) ** rng.random(400))[::-1]
    centers, placed = [], []
    for r in radii:
        for _ in range(n_attempts // len(radii)):
            c = r + (1 - 2 * r) * rng.random(2)
            if all(np.hypot(*(c - q)) >= r + s for q, s in zip(centers, placed)):
                centers.append(c)
                placed.append(r)
                break
    return np.array(centers), np.array(placed)


def cell_edges(result):
    """The edges of all of the cells, as line segments for plotting."""
    owners = np.repeat(np.arange(len(result)), np.diff(result.cell_facet_offsets))
    start = result.facet_vertex_offsets[:-1]
    base = result.cell_vertex_offsets[owners]
    v0 = result.vertices[base + result.facet_vertices[start]]
    v1 = result.vertices[base + result.facet_vertices[start + 1]]
    return np.stack([v0, v1], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", default="radical_tessellation.png")
    args = parser.parse_args()

    points, radii = disk_packing()
    limits = [[0, 1], [0, 1]]
    t = 0.04
    panels = [
        ("(a) Voronoi", voroplex.compute_voronoi_2d(points, limits), radii),
        ("(b) Radical", voroplex.compute_voronoi_2d(points, limits, radii=radii),
         radii),
        (f"(c) Radical, radii + {t}",
         voroplex.compute_voronoi_2d(points, limits, radii=radii + t),
         radii + t),
    ]
    print(f"{len(points)} disks, radii from {radii.min():.3f} to {radii.max():.3f}")
    for title, result, _ in panels:
        print(f"{title}: {np.sum(result.areas == 0)} empty cells")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (title, result, r) in zip(axes, panels):
        empty = result.areas == 0
        disks = [Circle(c, s) for c, s in zip(points, r)]
        if np.any(r != radii):
            # The enlarged disks overlap, so draw them as outlines, and fill
            # only those with empty cells.
            ax.add_collection(PatchCollection(
                [d for d, e in zip(disks, empty) if not e], facecolors="none",
                edgecolors="#4a7fb5", linewidths=0.6))
            ax.add_collection(PatchCollection(
                [d for d, e in zip(disks, empty) if e], facecolors="#e8a87c",
                edgecolors="#c0622b", linewidths=0.6, alpha=0.8))
        else:
            ax.add_collection(PatchCollection(disks, facecolors="#9fc3e7",
                                              edgecolors="none"))
        ax.add_collection(LineCollection(cell_edges(result), colors="k",
                                         linewidths=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
