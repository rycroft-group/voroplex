"""Write representative surface analysis outputs to surface_analysis_output.txt."""

from pathlib import Path

import numpy as np
import voroplex


points = np.random.default_rng(42).random((10, 3)) * 10
limits = [[0, 10], [0, 10], [0, 10]]

output_file = Path(__file__).with_name("surface_analysis_output.txt")
f = output_file.open("w")


def write_header(title, level=1):
    """Write a section header."""
    if level == 1:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n")
    else:
        f.write("\n" + "-" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("-" * 80 + "\n")


f.write("VoroPlex Surface Analysis Outputs\n")
f.write("=" * 80 + "\n")
f.write(f"Generated from {len(points)} random generator points in [0,10]^3\n")
f.write("Random seed: 42\n")

write_header("Setup: Extract Surface Mesh", level=1)

# The tessellation, and the mesh of the cell faces on the boundary of the domain.
# With several threads, the faces can be listed in a different order on each
# run, so one thread is used, to make the output file the same every time.
# For so few cells, one thread is as fast.
result = voroplex.compute_voronoi_3d(points, limits, threads=1)
surf = voroplex.surface_mesh(result, points)

f.write("\nStep 1: Compute packed Voronoi topology\n")
f.write("  result = voroplex.compute_voronoi_3d(points, limits, threads=1)\n")
f.write("\nStep 2: Extract surface mesh info\n")
f.write("  surf = voroplex.surface_mesh(result, points)\n")
f.write("\nSurface mesh summary:\n")
f.write(f'  - Surface vertices: {surf["vertex_coords"].shape[0]}\n')
f.write(f'  - Surface faces: {surf["facet_vertex_ids"].shape[0]}\n')
f.write(f'  - Max vertices per face: {surf["facet_vertex_ids"].shape[1]}\n')

write_header("compute_surface_geometry()", level=1)

# Every geometric property of the surface mesh, as a dictionary of tensors.
# The output file describes each one.
geom = voroplex.analysis.compute_surface_geometry(
    surf["vertex_coords"],
    surf["facet_vertex_ids"],
    surf["n_vertices_per_facet"],
    surf["facet_mask"]
)

f.write("\nFunction signature:\n")
f.write("  compute_surface_geometry(\n")
f.write("      vertex_coords, face_vertex_ids,\n")
f.write("      n_vertices_per_face, face_mask,\n")
f.write("      adjacency_tensor=None, n_adjacent=None,\n")
f.write("      face_neighbor_pairs=None, edges=None\n")
f.write("  )\n")

f.write("\nInput:\n")
f.write(
    f'  - vertex_coords: {surf["vertex_coords"].shape}, '
    f'{surf["vertex_coords"].dtype}\n'
)
f.write(f'  - face_vertex_ids: {surf["facet_vertex_ids"].shape}, {surf["facet_vertex_ids"].dtype}\n')
f.write(
    f'  - n_vertices_per_face: {surf["n_vertices_per_facet"].shape}, '
    f'{surf["n_vertices_per_facet"].dtype}\n'
)
f.write(f'  - face_mask: {surf["facet_mask"].shape}, {surf["facet_mask"].dtype}\n')

f.write("\nOutput: Dictionary with 19 properties in 5 categories\n")
f.write(f"  Total keys: {len(geom)}\n")
f.write("  Categories:\n")
f.write("    1. Basic geometry (5 properties)\n")
f.write("    2. Triangle subdivision (5 properties)\n")
f.write("    3. Face adjacency (4 properties)\n")
f.write("    4. Vertex neighbor statistics (2 properties)\n")
f.write("    5. Face neighbor statistics (3 properties)\n")
f.write("  Total: 5 + 5 + 4 + 2 + 3 = 19 properties\n")

write_header("Category 1: Basic Geometry (5 properties)", level=1)

f.write("\n1.1. face_areas\n")
f.write(f'  Shape: {geom["face_areas"].shape}\n')
f.write(f'  Dtype: {geom["face_areas"].dtype}\n')
f.write(f'  Device: {geom["face_areas"].device}\n')
f.write("  Description: Area of each polygon face\n")
f.write(
    f'  Statistics: min={geom["face_areas"].min():.4f}, '
    f'max={geom["face_areas"].max():.4f}, mean={geom["face_areas"].mean():.4f}\n'
)
f.write("  Sample values (first 8 faces):\n")
for i in range(min(8, len(geom["face_areas"]))):
    f.write(f'    Face {i}: area = {geom["face_areas"][i]:.6f}\n')

f.write("\n1.2. face_normals\n")
f.write(f'  Shape: {geom["face_normals"].shape}\n')
f.write(f'  Dtype: {geom["face_normals"].dtype}\n')
f.write("  Description: Outward-facing normal vector for each face\n")
f.write("  Sample values (first 5 faces):\n")
for i in range(min(5, len(geom["face_normals"]))):
    n = geom["face_normals"][i]
    f.write(f"    Face {i}: [{n[0]:8.5f}, {n[1]:8.5f}, {n[2]:8.5f}]\n")

f.write("\n1.3. face_centers\n")
f.write(f'  Shape: {geom["face_centers"].shape}\n')
f.write(f'  Dtype: {geom["face_centers"].dtype}\n')
f.write("  Description: Centroid (average vertex position) of each face\n")
f.write("  Sample values (first 5 faces):\n")
for i in range(min(5, len(geom["face_centers"]))):
    c = geom["face_centers"][i]
    f.write(f"    Face {i}: [{c[0]:8.4f}, {c[1]:8.4f}, {c[2]:8.4f}]\n")

f.write("\n1.4. face_circularities\n")
f.write(f'  Shape: {geom["face_circularities"].shape}\n')
f.write(f'  Dtype: {geom["face_circularities"].dtype}\n')
f.write("  Description: Isoperimetric ratio = 4π·Area/Perimeter² (1.0 = perfect circle)\n")
f.write(
    f'  Statistics: min={geom["face_circularities"].min():.4f}, '
    f'max={geom["face_circularities"].max():.4f}, '
    f'mean={geom["face_circularities"].mean():.4f}\n'
)
f.write("  Sample values (first 8 faces):\n")
for i in range(min(8, len(geom["face_circularities"]))):
    f.write(f'    Face {i}: circularity = {geom["face_circularities"][i]:.6f}\n')

f.write("\n1.5. face_perimeters\n")
f.write(f'  Shape: {geom["face_perimeters"].shape}\n')
f.write(f'  Dtype: {geom["face_perimeters"].dtype}\n')
f.write("  Description: Total edge length around each face\n")
f.write(
    f'  Statistics: min={geom["face_perimeters"].min():.4f}, '
    f'max={geom["face_perimeters"].max():.4f}, '
    f'mean={geom["face_perimeters"].mean():.4f}\n'
)
f.write("  Sample values (first 8 faces):\n")
for i in range(min(8, len(geom["face_perimeters"]))):
    f.write(f'    Face {i}: perimeter = {geom["face_perimeters"][i]:.6f}\n')

write_header("Category 2: Triangle Subdivision (5 properties)", level=1)

f.write("\nPolygon faces are subdivided using fan triangulation:\n")
f.write("  Each face → N triangles (one per edge)\n")
f.write("  Triangle structure: [v0, v1, face_centroid]\n")
f.write("  Purpose: Rendering, integration, area computation\n")

f.write("\n2.1. tri_vertices\n")
f.write(f'  Shape: {geom["tri_vertices"].shape}\n')
f.write(f'  Dtype: {geom["tri_vertices"].dtype}\n')
f.write("  Description: Combined vertex list [original vertices; face centroids]\n")
f.write("  Layout:\n")
f.write(f'    - Rows 0 to {surf["vertex_coords"].shape[0] - 1}: Original surface vertices\n')
f.write(
    f'    - Rows {surf["vertex_coords"].shape[0]} to '
    f'{geom["tri_vertices"].shape[0] - 1}: Face centroids (one per face)\n'
)

f.write("\n2.2. n_triangles\n")
f.write(f'  Value: {geom["n_triangles"]} (int)\n')
f.write("  Description: Total number of triangles in subdivision\n")
f.write(f'  Calculation: sum of vertices per face = {surf["n_vertices_per_facet"].sum().item()}\n')

f.write("\n2.3. tri_vertex_ids\n")
f.write(f'  Shape: {geom["tri_vertex_ids"].shape}\n')
f.write(f'  Dtype: {geom["tri_vertex_ids"].dtype}\n')
f.write("  Description: Triangle connectivity [v0, v1, centroid_id]\n")
f.write("  Sample triangles (first 8):\n")
for i in range(min(8, len(geom["tri_vertex_ids"]))):
    ids = geom["tri_vertex_ids"][i]
    f.write(f"    Triangle {i}: vertices [{ids[0]:3d}, {ids[1]:3d}, {ids[2]:3d}]\n")

f.write("\n2.4. tri_to_face_id\n")
f.write(f'  Shape: {geom["tri_to_face_id"].shape}\n')
f.write(f'  Dtype: {geom["tri_to_face_id"].dtype}\n')
f.write("  Description: Parent polygon face ID for each triangle\n")
f.write("  Sample mapping (first 8 triangles):\n")
for i in range(min(8, len(geom["tri_to_face_id"]))):
    face_id = geom["tri_to_face_id"][i].item()
    f.write(f"    Triangle {i} → Face {face_id}\n")

f.write("\n2.5. tri_centers\n")
f.write(f'  Shape: {geom["tri_centers"].shape}\n')
f.write(f'  Dtype: {geom["tri_centers"].dtype}\n')
f.write("  Description: Centroid of each subdivided triangle\n")
f.write("  Sample values (first 5 triangles):\n")
for i in range(min(5, len(geom["tri_centers"]))):
    c = geom["tri_centers"][i]
    f.write(f"    Triangle {i}: [{c[0]:8.4f}, {c[1]:8.4f}, {c[2]:8.4f}]\n")

write_header("Category 3: Face Adjacency (4 properties)", level=1)

f.write("\n3.1. adjacency_tensor\n")
f.write(f'  Shape: {geom["adjacency_tensor"].shape}\n')
f.write(f'  Dtype: {geom["adjacency_tensor"].dtype}\n')
f.write("  Description: Neighbor face IDs for each face, padded with -1\n")
f.write(f'  Max neighbors: {geom["adjacency_tensor"].shape[1]}\n')
f.write("  Sample adjacency (first 5 faces):\n")
for i in range(min(5, len(geom["adjacency_tensor"]))):
    n_neighbors = geom["n_adjacent"][i].item()
    neighbors = geom["adjacency_tensor"][i, :n_neighbors].tolist()
    f.write(f"    Face {i}: {n_neighbors} neighbors → {neighbors}\n")
    f.write(f'           Full array: {geom["adjacency_tensor"][i].tolist()}\n')

f.write("\n3.2. n_adjacent\n")
f.write(f'  Shape: {geom["n_adjacent"].shape}\n')
f.write(f'  Dtype: {geom["n_adjacent"].dtype}\n')
f.write("  Description: Number of neighboring faces for each face\n")
f.write(
    f'  Statistics: min={geom["n_adjacent"].min().item()}, '
    f'max={geom["n_adjacent"].max().item()}, '
    f'mean={geom["n_adjacent"].float().mean():.2f}\n'
)
f.write("  Sample values (first 10 faces):\n")
for i in range(min(10, len(geom["n_adjacent"]))):
    f.write(f'    Face {i}: {geom["n_adjacent"][i].item()} neighbors\n')

f.write("\n3.3. face_neighbor_pairs\n")
f.write(f'  Shape: {geom["face_neighbor_pairs"].shape}\n')
f.write(f'  Dtype: {geom["face_neighbor_pairs"].dtype}\n')
f.write("  Description: Pairs of neighboring face IDs [f1, f2] where f1 < f2\n")
f.write(f'  Total pairs: {len(geom["face_neighbor_pairs"])}\n')
f.write("  Sample pairs (first 10):\n")
for i in range(min(10, len(geom["face_neighbor_pairs"]))):
    pair = geom["face_neighbor_pairs"][i]
    f.write(f"    Pair {i}: Face {pair[0].item()} ↔ Face {pair[1].item()}\n")

f.write("\n3.4. edges\n")
f.write(f'  Shape: {geom["edges"].shape}\n')
f.write(f'  Dtype: {geom["edges"].dtype}\n')
f.write("  Description: Unique undirected edges [v1, v2] where v1 < v2\n")
f.write(f'  Total edges: {len(geom["edges"])}\n')
f.write("  Sample edges (first 10):\n")
for i in range(min(10, len(geom["edges"]))):
    edge = geom["edges"][i]
    f.write(f"    Edge {i}: Vertex {edge[0].item()} ↔ Vertex {edge[1].item()}\n")

write_header("Category 4: Vertex Neighbor Statistics (2 properties)", level=1)

f.write("\nUsed for Laplacian smoothing and regularization\n")

f.write("\n4.1. vertex_neighbor_avg_position\n")
f.write(f'  Shape: {geom["vertex_neighbor_avg_position"].shape}\n')
f.write(f'  Dtype: {geom["vertex_neighbor_avg_position"].dtype}\n')
f.write("  Description: Average position of neighboring vertices (connected by edges)\n")
f.write("  Use case: Laplacian smoothing (move vertex toward neighbor average)\n")
f.write("  Sample values (first 5 vertices):\n")
for i in range(min(5, len(geom["vertex_neighbor_avg_position"]))):
    pos = geom["vertex_neighbor_avg_position"][i]
    f.write(f"    Vertex {i}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]\n")

f.write("\n  Laplacian displacement (vertex_neighbor_avg_position - current_pos):\n")
for i in range(min(5, len(geom["vertex_neighbor_avg_position"]))):
    displacement = geom["vertex_neighbor_avg_position"][i] - surf["vertex_coords"][i]
    f.write(
        f"    Vertex {i}: [{displacement[0]:8.4f}, {displacement[1]:8.4f}, "
        f"{displacement[2]:8.4f}]\n"
    )

f.write("\n4.2. vertex_neighbor_avg_normal\n")
f.write(f'  Shape: {geom["vertex_neighbor_avg_normal"].shape}\n')
f.write(f'  Dtype: {geom["vertex_neighbor_avg_normal"].dtype}\n')
f.write("  Description: Average normal of faces incident to each vertex\n")
f.write("  Use case: Surface smoothness regularization\n")
f.write("  Sample values (first 5 vertices):\n")
for i in range(min(5, len(geom["vertex_neighbor_avg_normal"]))):
    n = geom["vertex_neighbor_avg_normal"][i]
    f.write(f"    Vertex {i}: [{n[0]:8.5f}, {n[1]:8.5f}, {n[2]:8.5f}]\n")

write_header("Category 5: Face Neighbor Statistics (3 properties)", level=1)

f.write("\nUsed for geometric analysis and regularization objectives\n")

f.write("\n5.1. face_neighbor_avg_center\n")
f.write(f'  Shape: {geom["face_neighbor_avg_center"].shape}\n')
f.write(f'  Dtype: {geom["face_neighbor_avg_center"].dtype}\n')
f.write("  Description: Average centroid of neighboring faces\n")
f.write("  Use case: Spatial uniformity regularization\n")
f.write("  Sample values (first 5 faces):\n")
for i in range(min(5, len(geom["face_neighbor_avg_center"]))):
    c = geom["face_neighbor_avg_center"][i]
    f.write(f"    Face {i}: [{c[0]:8.4f}, {c[1]:8.4f}, {c[2]:8.4f}]\n")

f.write("\n5.2. face_neighbor_avg_area\n")
f.write(f'  Shape: {geom["face_neighbor_avg_area"].shape}\n')
f.write(f'  Dtype: {geom["face_neighbor_avg_area"].dtype}\n')
f.write("  Description: Average area of neighboring faces\n")
f.write("  Use case: Area uniformity objective\n")
f.write("  Sample comparison (first 8 faces):\n")
for i in range(min(8, len(geom["face_neighbor_avg_area"]))):
    face_area = geom["face_areas"][i].item()
    neighbor_avg = geom["face_neighbor_avg_area"][i].item()
    diff = abs(face_area - neighbor_avg)
    f.write(
        f"    Face {i}: area={face_area:8.4f} | neighbor_avg={neighbor_avg:8.4f} | "
        f"diff={diff:8.4f}\n"
    )

f.write("\n5.3. face_neighbor_avg_normal\n")
f.write(f'  Shape: {geom["face_neighbor_avg_normal"].shape}\n')
f.write(f'  Dtype: {geom["face_neighbor_avg_normal"].dtype}\n')
f.write("  Description: Average normal of neighboring faces\n")
f.write("  Use case: Surface smoothness objective (normal consistency)\n")
f.write("  Sample values (first 5 faces):\n")
for i in range(min(5, len(geom["face_neighbor_avg_normal"]))):
    n = geom["face_neighbor_avg_normal"][i]
    f.write(f"    Face {i}: [{n[0]:8.5f}, {n[1]:8.5f}, {n[2]:8.5f}]\n")

write_header("Summary", level=1)

f.write("\nOutput dictionary structure:\n")
f.write("  geom = compute_surface_geometry(...)\n")
f.write("  type(geom) = dict\n")
f.write(f"  len(geom) = {len(geom)} keys\n")
f.write(f"\n  All {len(geom)} keys:\n")
for i, key in enumerate(geom.keys(), 1):
    f.write(f"    {i:2d}. {key}\n")

f.write("\nDifferentiability:\n")
f.write("  All outputs are PyTorch tensors\n")
f.write("  All are differentiable with respect to vertex_coords\n")
f.write("  Enables gradient-based optimization of surface properties\n")

f.write("\nTypical workflow:\n")
f.write("  1. Extract surface mesh: surface_mesh()\n")
f.write("  2. Analyze geometry: compute_surface_geometry()\n")
f.write('  3. Access properties: geom["face_areas"], geom["adjacency_tensor"], etc.\n')
f.write("  4. Use in analysis, rendering, or optimization\n")

f.write("\n" + "=" * 80 + "\n")
f.write("Output saved to: surface_analysis_output.txt\n")
f.write("=" * 80 + "\n")

f.close()

print(f"Output saved to: {output_file}")
print(f"Total properties computed: {len(geom)}")
print("  - Basic geometry: 5 properties")
print("  - Triangle subdivision: 5 properties")
print("  - Face adjacency: 4 properties")
print("  - Vertex neighbor statistics: 2 properties")
print("  - Face neighbor statistics: 3 properties")
