"""
Example: VoroPlex 2D Mesh Utilities
===================================

Run from the project root:
    python examples/09_2d_mesh_utilities.py
"""

import numpy as np
import torch

import voroplex


rng = np.random.default_rng(9)
points = rng.random((20, 2)) * 0.8 + 0.1
limits = [[0, 1], [0, 1]]

# The topology representation labels each vertex with the generators that
# define it, so that the copies of a vertex in different cells can be matched,
# and its position recomputed from the generators.
result = voroplex.compute_voronoi_2d(points, limits, representation="topology")

# Each vertex once, as a tensor; each cell's vertices, padded to equal length;
# each edge once; each cell's vertices in order; and the edges on the boundary.
vertex_xy = voroplex.compute_unique_vertices(result, points)
cell_xy, num_verts = voroplex.compute_padded_cell_vertices(result, points)
edges, edge_info = voroplex.unique_facets(result, facet_type="all")
polygons, cell_ids = voroplex.cell_polytopes(result)
boundary = voroplex.surface_mesh(result, points)

# Pack the polygons into a padded index array and a mask, the form that
# compute_polygon_geometry expects, to find the area and other properties
# of each cell.
max_len = max(len(poly) for poly in polygons)
cell_vertex_ids = torch.full((len(polygons), max_len), -1, dtype=torch.long)
cell_mask = torch.zeros((len(polygons), max_len), dtype=torch.bool)
for i, poly in enumerate(polygons):
    cell_vertex_ids[i, :len(poly)] = torch.tensor(poly, dtype=torch.long)
    cell_mask[i, :len(poly)] = True

geom = voroplex.analysis.compute_polygon_geometry(
    vertex_xy,
    cell_vertex_ids,
    cell_mask.sum(dim=1),
    cell_mask,
)

print("=" * 70)
print("VoroPlex 2D Mesh Utilities")
print("=" * 70)
print(f"Unique vertices:       {tuple(vertex_xy.shape)}")
print(f"Per-cell vertex tensor:{tuple(cell_xy.shape)}")
print(f"Num verts first 5:     {num_verts[:5].tolist()}")
print(f"Unique edges:          {len(edges)}")
print(f"Edge info shape:       {edge_info.shape}")
print(f"Polygonal cells:       {len(polygons)}")
print(f"Boundary edge IDs:     {tuple(boundary['facet_vertex_ids'].shape)}")
print(f"Cell areas sum:        {geom['cell_areas'].sum().item():.12f}")
print(f"Adjacency pairs:       {geom['cell_neighbor_pairs'].shape[0]}")
print(f"Cell IDs first 5:      {cell_ids[:5]}")
