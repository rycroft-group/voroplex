"""Basic VoroPlex usage with the three result representations."""

import numpy as np
import torch

import voroplex


rng = np.random.default_rng(42)
points = rng.random((100, 3))
limits = [[0.0, 1.0]] * 3

# Default: the geometry representation, with each cell's vertices, faces, and
# neighbors as computed by Voro++.
geometry = voroplex.compute_voronoi_3d(points, limits, threads="auto")
print(geometry)
print("Total volume:", geometry.volumes.sum())
first = geometry.cell(0)
print("First-cell vertices:", first.vertices.shape)
print("First-cell neighbors:", first.neighbors)
print("First-cell faces:", len(first.faces))

# Inner points first, outer clipping points second. Only inner cells are
# computed, but the outer points still cut them.
inner_only = voroplex.compute_voronoi_3d(points, limits, n_compute=80, n_inner=80)
cells, cell_ids = voroplex.cell_polytopes(inner_only)
print("Computed inner cells:", len(cells))
print("Inner/outer interface cells:", inner_only.interface_cells)
print("Cells on the boundary of the domain:", inner_only.boundary_cells)

# Per-cell representation: the same cells, plus what is needed to
# differentiate the vertices with respect to the generator positions.
points_t = torch.tensor(points, dtype=torch.float64, requires_grad=True)
per_cell = voroplex.compute_voronoi_3d(points, limits, representation="per_cell")
vertices = voroplex.cell_vertices(per_cell, points_t)
vertices.square().sum().backward()
print("Per-cell vertices:", tuple(vertices.shape),
      "finite gradients:", bool(torch.isfinite(points_t.grad).all()))

# Topology representation: each vertex appears once, shared between cells,
# and its position is recomputed from the generators in PyTorch.
topology = voroplex.compute_voronoi_3d(points, limits, representation="topology")
points_t.grad = None
vertex_xyz = voroplex.compute_unique_vertices(topology, points_t)
vertex_xyz.square().sum().backward()
print("Unique vertices:", tuple(vertex_xyz.shape),
      "finite gradients:", bool(torch.isfinite(points_t.grad).all()))
