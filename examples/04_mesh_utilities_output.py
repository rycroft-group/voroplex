"""Write representative mesh utility outputs to mesh_utilities_output.txt."""

from pathlib import Path

import numpy as np
import voroplex


points = np.random.default_rng(42).random((10, 3)) * 10
limits = [[0, 10], [0, 10], [0, 10]]

output_file = Path(__file__).with_name("mesh_utilities_output.txt")
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


f.write("VoroPlex Mesh Utility Outputs\n")
f.write("=" * 80 + "\n")
f.write(f"Generated from {len(points)} random generator points in [0,10]^3\n")
f.write("Random seed: 42\n")

write_header("1. compute_unique_vertices(result, points)", level=1)

# One tessellation, in the topology representation, is used for every
# function below. With several threads, the faces of the tessellation can be
# listed in a different order on each run, so one thread is used, to make the
# output file the same every time. For so few cells, one thread is as fast.
result = voroplex.compute_voronoi_3d(points, limits, representation="topology", threads=1)
vertices = voroplex.compute_unique_vertices(result, points, device="cpu")

f.write("\nFunction signature:\n")
f.write('  compute_unique_vertices(result, points, device="cpu")\n')
f.write("\nInput:\n")
f.write("  - result: packed topology result from compute_voronoi_3d()\n")
f.write(f"  - points: {points.shape} ndarray (generator points)\n")
f.write("\nOutput:\n")
f.write(f"  - vertices: torch.Tensor, shape {vertices.shape}, dtype {vertices.dtype}\n")
f.write(f"  - Device: {vertices.device}\n")
f.write(f"  - Requires grad: {vertices.requires_grad}\n")
f.write("\nVertex positions (first 10 vertices):\n")
for i in range(min(10, len(vertices))):
    v = vertices[i]
    f.write(f"  Vertex {i:3d}: [{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}]\n")
if len(vertices) > 10:
    f.write(f"  ... ({len(vertices) - 10} more vertices)\n")

write_header("2. unique_facets(result)", level=1)

faces_all, face_info_all = voroplex.unique_facets(result, facet_type="all")
faces_surface, face_info_surface = voroplex.unique_facets(result, facet_type="surface")
faces_inner, face_info_inner = voroplex.unique_facets(result, facet_type="inner")

f.write("\nFunction signature:\n")
f.write('  unique_facets(result, facet_type="all")\n')

write_header('2a. facet_type="all"', level=2)
f.write("\nInput:\n")
f.write("  - result: packed topology result from compute_voronoi_3d(points, limits)\n")
f.write('  - facet_type: "all"\n')
f.write("\nOutput:\n")
f.write(f"  - faces_vID: list of {len(faces_all)} faces (vertex ID lists)\n")
f.write(f"  - face_info: ndarray {face_info_all.shape}, dtype {face_info_all.dtype}\n")
f.write("    - Column 0: owner cell ID\n")
f.write("    - Column 1: neighbor cell ID (negative = boundary of the domain)\n")
f.write("\nFace details (first 8 faces):\n")
for i in range(min(8, len(faces_all))):
    owner = face_info_all[i, 0]
    neighbor = face_info_all[i, 1]
    neighbor_str = f"cell {neighbor}" if neighbor >= 0 else f"boundary {neighbor}"
    f.write(
        f"  Face {i:3d}: {len(faces_all[i]):2d} vertices | owner=cell {owner} | "
        f"neighbor={neighbor_str}\n"
    )
    f.write(f"           Vertex IDs: {faces_all[i]}\n")
if len(faces_all) > 8:
    f.write(f"  ... ({len(faces_all) - 8} more faces)\n")

write_header('2b. facet_type="surface"', level=2)
f.write("\nOutput:\n")
f.write(f"  - faces_vID: list of {len(faces_surface)} faces\n")
f.write(f"  - face_info: ndarray {face_info_surface.shape}\n")
f.write("\nSurface face details (first 8 faces):\n")
for i in range(min(8, len(faces_surface))):
    owner = face_info_surface[i, 0]
    neighbor = face_info_surface[i, 1]
    f.write(
        f"  Face {i:3d}: {len(faces_surface[i]):2d} vertices | owner=cell {owner} | "
        f"boundary={neighbor}\n"
    )
    f.write(f"           Vertex IDs: {faces_surface[i]}\n")
if len(faces_surface) > 8:
    f.write(f"  ... ({len(faces_surface) - 8} more faces)\n")

write_header('2c. facet_type="inner"', level=2)
f.write("\nOutput:\n")
f.write(f"  - faces_vID: list of {len(faces_inner)} faces\n")
f.write(f"  - face_info: ndarray {face_info_inner.shape}\n")
f.write("\nInner face details (first 8 faces):\n")
for i in range(min(8, len(faces_inner))):
    owner = face_info_inner[i, 0]
    neighbor = face_info_inner[i, 1]
    f.write(
        f"  Face {i:3d}: {len(faces_inner[i]):2d} vertices | owner=cell {owner} | "
        f"neighbor=cell {neighbor}\n"
    )
    f.write(f"           Vertex IDs: {faces_inner[i]}\n")
if len(faces_inner) > 8:
    f.write(f"  ... ({len(faces_inner) - 8} more faces)\n")

write_header("3. cell_polytopes(result)", level=1)

cells, cell_ids = voroplex.cell_polytopes(result)

f.write("\nFunction signature:\n")
f.write("  cell_polytopes(result)\n")
f.write("\nInput:\n")
f.write("  - result: packed topology result from compute_voronoi_3d(points, limits)\n")
f.write("  - packed results contain complete cell facets\n")
f.write("\nOutput:\n")
f.write(f"  - cells: list of {len(cells)} polyhedral cells (each cell = list of faces)\n")
f.write(f"  - cell_ids: list of {len(cell_ids)} int, values: {cell_ids}\n")
f.write("\nCell topology (first 3 cells):\n")
for i in range(min(3, len(cells))):
    cell_id = cell_ids[i]
    f.write(f"\n  Cell {cell_id} ({len(cells[i])} faces):\n")
    for j, face in enumerate(cells[i][:5]):
        f.write(f"    Face {j}: {len(face)} vertices, IDs = {face}\n")
    if len(cells[i]) > 5:
        f.write(f"    ... ({len(cells[i]) - 5} more faces)\n")
if len(cells) > 3:
    f.write(f"\n  ... ({len(cells) - 3} more cells)\n")

write_header("4. surface_mesh(result, points)", level=1)

surf = voroplex.surface_mesh(result, points)

f.write("\nFunction signature:\n")
f.write('  surface_mesh(result, points, device="cpu")\n')
f.write("\nInput:\n")
f.write("  - result: packed topology result from compute_voronoi_3d(points, limits)\n")
f.write("  - surface facets are selected from packed neighbor IDs\n")
f.write("\nOutput: dictionary with 5 keys\n")
f.write(
    f'  - face_vertex_ids: torch.Tensor {surf["facet_vertex_ids"].shape}, dtype '
    f'{surf["facet_vertex_ids"].dtype}\n'
)
f.write("      Compact vertex indexing (surface vertices only), padded with -100\n")
f.write(
    f'  - vertex_coords: torch.Tensor {surf["vertex_coords"].shape}, dtype '
    f'{surf["vertex_coords"].dtype}\n'
)
f.write("      Surface vertex positions only (differentiable)\n")
f.write(
    f'  - n_vertices_per_face: torch.Tensor {surf["n_vertices_per_facet"].shape}, '
    f'dtype {surf["n_vertices_per_facet"].dtype}\n'
)
f.write("      Number of vertices for each face\n")
f.write(f'  - face_mask: torch.Tensor {surf["facet_mask"].shape}, dtype {surf["facet_mask"].dtype}\n')
f.write("      Boolean mask for valid vertex indices (filters out -100 padding)\n")
f.write(
    '  - face_neighbor_cell_ids: torch.Tensor '
    f'{surf["facet_neighbors"].shape}, dtype '
    f'{surf["facet_neighbors"].dtype}\n'
)
f.write("      Neighbor cell ID (owner is implicit) - negative values = boundary codes\n")

f.write("\nSurface mesh data (first 8 faces):\n")
for i in range(min(8, len(surf["facet_vertex_ids"]))):
    mask = surf["facet_mask"][i]
    n_verts = surf["n_vertices_per_facet"][i].item()
    compact_ids = surf["facet_vertex_ids"][i][mask].tolist()
    neighbor = surf["facet_neighbors"][i].item()
    neighbor_str = f"boundary {neighbor}" if neighbor < 0 else f"cell {neighbor}"

    f.write(f"\n  Face {i}: {n_verts} vertices, neighbor={neighbor_str}\n")
    f.write(f"    Compact vertex IDs: {compact_ids}\n")
    f.write(f'    Padded array: {surf["facet_vertex_ids"][i].tolist()}\n')
    f.write(f"    Validity mask: {mask.tolist()}\n")

    if i == 0:
        f.write("    Vertex coordinates:\n")
        for j, vid in enumerate(compact_ids):
            coord = surf["vertex_coords"][vid]
            f.write(f"      Vertex {vid}: [{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}]\n")

if len(surf["facet_vertex_ids"]) > 8:
    f.write(f'\n  ... ({len(surf["facet_vertex_ids"]) - 8} more faces)\n')

write_header("Summary of All Outputs", level=1)

f.write("\nFunction outputs summary:\n")
f.write("  1. compute_unique_vertices():\n")
f.write(f"     - {len(vertices)} unique vertices as torch.Tensor\n")
f.write("     - Global vertex indexing (all vertices)\n")
f.write("\n  2. unique_facets():\n")
f.write(f"     - all: {len(faces_all)} faces (surface + inner)\n")
f.write(f"     - surface: {len(faces_surface)} faces\n")
f.write(f"     - inner: {len(faces_inner)} faces\n")
f.write("     - Returns vertex IDs (list of lists) + metadata (ndarray)\n")
f.write("\n  3. cell_polytopes():\n")
f.write(f"     - {len(cells)} polyhedral cells\n")
f.write(f"     - Cell IDs: {cell_ids}\n")
f.write("     - Cell→Face→Vertex hierarchy\n")
f.write("\n  4. surface_mesh():\n")
f.write(f'     - {len(surf["facet_vertex_ids"])} surface faces\n')
f.write(f'     - {len(surf["vertex_coords"])} surface vertices (compact indexing)\n')
f.write("     - All data as torch.Tensor (differentiable)\n")

f.write("\n\n" + "=" * 80 + "\n")
f.write("End of mesh utility outputs\n")
f.write("=" * 80 + "\n")

f.close()

print(f"Mesh utility outputs saved to: {output_file}")
print(f"Generated from {len(points)} random points")
print("\nOutput file contains:")
print("  1. compute_unique_vertices() - vertex positions")
print("  2. unique_facets() - face connectivity (all/surface/inner)")
print("  3. cell_polytopes() - cell topology")
print("  4. surface_mesh() - compact surface mesh")
