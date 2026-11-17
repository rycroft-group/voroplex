"""Demonstrate the 3D mesh utilities:

1. compute_unique_vertices() - Compute explicit vertex positions
2. unique_facets() - Get deduplicated face connectivity
3. cell_polytopes() - Build cell→face→vertex hierarchy
4. surface_mesh() - Extract compact surface mesh

All functions require PyTorch. Vertex positions are differentiable tensors.
"""

import numpy as np
import voroplex


points = np.random.default_rng(42).random((20, 3)) * 10
limits = [[0, 10], [0, 10], [0, 10]]

# All of the functions below use the topology representation, in which each
# vertex is labeled by the generators that define it. The same result is used
# throughout.
result = voroplex.compute_voronoi_3d(points, limits, representation="topology")

print("=" * 70)
print("VoroPlex Mesh Utility Functions Examples")
print("=" * 70)

# Each vertex is listed once, and its position is found from the generators
# that define it, as the point equidistant from them, so it is differentiable
# with respect to the generator positions.
print("\n1. compute_unique_vertices()")
print("-" * 70)

vertices = voroplex.compute_unique_vertices(result, points, device="cpu")

print(f"Input: {len(points)} generator points")
print(f"Output: {len(vertices)} unique Voronoi vertices")
print(f"Shape: {vertices.shape}")
print(f"Type: {type(vertices)} (differentiable)")
print("\nFirst 3 vertices:")
for i in range(min(3, len(vertices))):
    print(f"  Vertex {i}: [{vertices[i, 0]:.4f}, {vertices[i, 1]:.4f}, {vertices[i, 2]:.4f}]")

# Each face is listed once, as a list of indices into the unique vertices.
# face_info gives the two cells on either side; a negative neighbor is the boundary of the domain.
print("\n2. unique_facets()")
print("-" * 70)

faces, face_info = voroplex.unique_facets(result, facet_type="surface")

print("Input: packed Voronoi topology (extracting surface faces)")
print(f"Output: {len(faces)} surface faces")
print("faces_vID: list of lists (vertex IDs)")
print(f"face_info: ndarray {face_info.shape} (cell_id, neighbor_id)")
print("\nFirst 3 faces:")
for i in range(min(3, len(faces))):
    owner_id = face_info[i, 0]
    neighbor_id = face_info[i, 1]
    neighbor_type = "boundary" if neighbor_id < 0 else f"cell {neighbor_id}"
    print(f"  Face {i}: {len(faces[i])} vertices, owner=cell {owner_id}, neighbor={neighbor_type}")
    print(f"           Vertex IDs: {faces[i]}")

# For every cell, the list of its faces, each a list of unique vertex indices.
print("\n3. cell_polytopes()")
print("-" * 70)

cells, cell_ids = voroplex.cell_polytopes(result)

print("Input: packed Voronoi topology")
print(f"Output: {len(cells)} polyhedral cells")
print(f"Cell IDs: {cell_ids[:5]}... (sorted)")
print(f"\nFirst cell (cell {cell_ids[0]}):")
print(f"  Number of faces: {len(cells[0])}")
for i in range(min(3, len(cells[0]))):
    print(f"  Face {i}: {len(cells[0][i])} vertices, IDs = {cells[0][i]}")

# The faces on the surface of the inner region, as a compact mesh with its own
# numbering of the vertices. The faces are padded to equal length with -100.
print("\n4. surface_mesh()")
print("-" * 70)

surf = voroplex.surface_mesh(result, points)

print("Input: packed Voronoi topology")
print("Output: compact surface mesh dictionary")
print(f"  face_vertex_ids: {surf['facet_vertex_ids'].shape} (compact indexing, padded with -100)")
print(f"  vertex_coords: {surf['vertex_coords'].shape} (surface vertices only)")
print(f"  n_vertices_per_face: {surf['n_vertices_per_facet'].shape}")
print(f"  face_mask: {surf['facet_mask'].shape} (validity mask)")
print(f"  face_neighbor_cell_ids: {surf['facet_neighbors'].shape}")

print("\nFirst 3 faces (compact representation):")
for i in range(min(3, len(surf["facet_vertex_ids"]))):
    mask = surf["facet_mask"][i]
    n_verts = surf["n_vertices_per_facet"][i].item()
    neighbor_id = surf["facet_neighbors"][i].item()
    neighbor_type = "boundary" if neighbor_id < 0 else f"cell {neighbor_id}"
    print(f"  Face {i}: {n_verts} vertices, neighbor={neighbor_type}")
    print(f"           Compact vertex IDs: {surf['facet_vertex_ids'][i][mask].tolist()}")

# Compose mesh utilities.
print("\n" + "=" * 70)
print("Composition Example: Vertices + Faces")
print("=" * 70)

# The vertex positions and the face lists together make a polygon mesh: the
# coordinates of a face are the rows of `vertices` given by its indices.
vertices = voroplex.compute_unique_vertices(result, points)
faces, face_info = voroplex.unique_facets(result, facet_type="surface")

print("\nComposed mesh:")
print(f"  Vertices: {vertices.shape} torch.Tensor (differentiable)")
print(f"  Faces: {len(faces)} polygons")
print(f"  Face info: {face_info.shape} ndarray (metadata)")
print("\nExample: Access face vertex coordinates:")
face_idx = 0
face_vertex_ids = faces[face_idx]
face_coords = vertices[face_vertex_ids]
print(f"  Face {face_idx} has {len(face_vertex_ids)} vertices:")
print(f"  Coordinates shape: {face_coords.shape}")

print("\n" + "=" * 70)
print("Mesh utility examples completed")
print("=" * 70)
