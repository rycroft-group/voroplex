"""Basic 2D usage, with the default geometry representation."""

import numpy as np

import voroplex


points = np.random.default_rng(8).random((50, 2))
limits = [[0.0, 1.0], [0.0, 1.0]]

# The default geometry representation: each cell's vertices, edges, and
# neighbors, as computed by Voro++.
geometry = voroplex.compute_voronoi_2d(points, limits)

print(geometry)
print("Total area:", geometry.areas.sum())
print("Boundary cells:", geometry.surface_cells)
# Each edge between two cells is listed once by each of them;
# unique_facets lists it once.
print("Unique inner edges:", len(voroplex.unique_facets(
    geometry, facet_type="inner"
)[0]))

# The vertices of cell 0, the neighbor across each of its edges (negative for
# the boundary of the domain), and the indices of the two vertices of each edge.
cell = geometry.cell(0)
print("First-cell vertices:\n", cell.vertices)
print("First-cell edge neighbors:", cell.neighbors)
print("First-cell edges:", [edge.vertex_ids.tolist() for edge in cell.edges])
