"""Demonstrate surface mesh analysis:

- Face areas, normals, centers, circularities, perimeters
- Triangle subdivision for integration/rendering
- Face adjacency and connectivity
- Vertex and face neighbor statistics

All outputs are differentiable PyTorch tensors, enabling gradient-based optimization.
"""

import numpy as np
import voroplex


points = np.random.default_rng(42).random((20, 3)) * 10
limits = [[0, 10], [0, 10], [0, 10]]

print("=" * 70)
print("VoroPlex Surface Analysis Demo")
print("=" * 70)

# Compute the tessellation and surface mesh.
print("\n1. Computing Voronoi tessellation and extracting surface mesh...")

result = voroplex.compute_voronoi_3d(points, limits)
surf = voroplex.surface_mesh(result, points)

print(f"   Surface mesh: {surf['vertex_coords'].shape[0]} vertices, "
      f"{surf['facet_vertex_ids'].shape[0]} faces")

# Analyze surface geometry.
print("\n2. Analyzing surface mesh geometry...")

geom = voroplex.analysis.compute_surface_geometry(
    surf["vertex_coords"],
    surf["facet_vertex_ids"],
    surf["n_vertices_per_facet"],
    surf["facet_mask"]
)

print("   Geometric properties computed:")
print(f"   - {len(geom)} properties available")

# Basic geometry properties.
print("\n3. Basic Geometry Properties:")
print("-" * 70)

print(f"   Face areas: min={geom['face_areas'].min():.4f}, "
      f"max={geom['face_areas'].max():.4f}, "
      f"mean={geom['face_areas'].mean():.4f}")

print(f"   Face perimeters: min={geom['face_perimeters'].min():.4f}, "
      f"max={geom['face_perimeters'].max():.4f}, "
      f"mean={geom['face_perimeters'].mean():.4f}")

print("   Face circularities (isoperimetric ratio): "
      f"min={geom['face_circularities'].min():.4f}, "
      f"max={geom['face_circularities'].max():.4f}, "
      f"mean={geom['face_circularities'].mean():.4f}")
print("      (Perfect circle = 1.0, lower values indicate elongated faces)")

# Triangle subdivision.
print("\n4. Triangle Subdivision:")
print("-" * 70)

print(f"   Original: {surf['facet_vertex_ids'].shape[0]} polygon faces")
print(f"   Subdivided: {geom['n_triangles']} triangles (fan triangulation)")
print(f"   Triangle vertices shape: {geom['tri_vertices'].shape}")
print(f"   Triangle IDs shape: {geom['tri_vertex_ids'].shape}")
print("   (Each polygon face split into triangles: [v0, v1, centroid])")

# Face adjacency and connectivity.
print("\n5. Face Adjacency and Connectivity:")
print("-" * 70)

print(f"   Unique edges: {geom['edges'].shape[0]}")
print(f"   Face neighbor pairs: {geom['face_neighbor_pairs'].shape[0]}")
print(f"   Max neighbors per face: {geom['adjacency_tensor'].shape[1]}")
print(f"   Average neighbors per face: {geom['n_adjacent'].float().mean():.2f}")

print("\n   First 5 faces and their neighbors:")
for i in range(min(5, surf["facet_vertex_ids"].shape[0])):
    num_neighbors = geom["n_adjacent"][i].item()
    neighbors = geom["adjacency_tensor"][i, :num_neighbors].tolist()
    print(f"   Face {i}: {num_neighbors} neighbors → {neighbors}")

# Vertex neighbor information.
print("\n6. Vertex Neighbor Information:")
print("-" * 70)

print(f"   vertex_neighbor_avg_position shape: {geom['vertex_neighbor_avg_position'].shape}")
print(f"   vertex_neighbor_avg_normal shape: {geom['vertex_neighbor_avg_normal'].shape}")
print("   (Used for Laplacian smoothing: move vertices toward neighbor average)")

vertices = surf["vertex_coords"]
laplacian_displacement = geom["vertex_neighbor_avg_position"] - vertices
print("\n   Laplacian displacement (first 3 vertices):")
for i in range(min(3, vertices.shape[0])):
    disp = laplacian_displacement[i]
    print(f"   Vertex {i}: [{disp[0]:7.4f}, {disp[1]:7.4f}, {disp[2]:7.4f}]")

# Face neighbor information.
print("\n7. Face Neighbor Information:")
print("-" * 70)

print(f"   face_neighbor_avg_center shape: {geom['face_neighbor_avg_center'].shape}")
print(f"   face_neighbor_avg_area shape: {geom['face_neighbor_avg_area'].shape}")
print(f"   face_neighbor_avg_normal shape: {geom['face_neighbor_avg_normal'].shape}")
print("   (Used for optimization objectives: area uniformity, normal consistency)")

area_diff = (geom["face_areas"] - geom["face_neighbor_avg_area"]).abs()
print("\n   Area uniformity metric (first 5 faces):")
for i in range(min(5, surf["facet_vertex_ids"].shape[0])):
    face_area = geom["face_areas"][i].item()
    neighbor_avg = geom["face_neighbor_avg_area"][i].item()
    diff = area_diff[i].item()
    print(f"   Face {i}: area={face_area:.4f}, neighbor_avg={neighbor_avg:.4f}, diff={diff:.4f}")

# Optimization objectives.
print("\n8. Example Optimization Objectives:")
print("-" * 70)

area_variance = geom["face_areas"].var()
print(f"   Area variance (minimize for uniform faces): {area_variance:.6f}")

mean_circularity = geom["face_circularities"].mean()
print(f"   Mean circularity (maximize for circular faces): {mean_circularity:.6f}")

normal_diff = geom["face_normals"] - geom["face_neighbor_avg_normal"]
normal_consistency = normal_diff.norm(dim=1).mean()
print(f"   Normal inconsistency (minimize for smooth surface): {normal_consistency:.6f}")

area_uniformity = (geom["face_areas"] - geom["face_neighbor_avg_area"]).abs().mean()
print(f"   Area non-uniformity (minimize for uniform areas): {area_uniformity:.6f}")

# Differentiability.
print("\n9. Differentiability Check:")
print("-" * 70)

surf["vertex_coords"].requires_grad_(True)

geom2 = voroplex.analysis.compute_surface_geometry(
    surf["vertex_coords"],
    surf["facet_vertex_ids"],
    surf["n_vertices_per_facet"],
    surf["facet_mask"]
)

loss = geom2["face_areas"].var() + (1 - geom2["face_circularities"]).mean()
loss.backward()

print(f"   Loss computed: {loss.item():.6f}")
print(f"   Gradient shape: {surf['vertex_coords'].grad.shape}")
print(f"   Gradient norm: {surf['vertex_coords'].grad.norm():.6f}")
print("   ✓ All outputs are differentiable!")

print("\n" + "=" * 70)
print("Summary: Surface Analysis Workflow")
print("=" * 70)
print("""
VoroPlex surface analysis utilities provide comprehensive mesh analysis:

1. Extract surface mesh from Voronoi result
   → voroplex.surface_mesh()

2. Analyze surface mesh geometry
   → voroplex.analysis.compute_surface_geometry()

3. Access 19 geometric properties:
   - Basic: areas, normals, centers, circularities, perimeters
   - Subdivision: triangulated mesh for integration
   - Connectivity: adjacency, neighbor pairs, unique edges
   - Statistics: vertex/face neighbor averages

4. All outputs are differentiable PyTorch tensors
   - Enables gradient-based optimization
   - Useful for rendering, analysis, and geometric queries
""")
print("=" * 70)
