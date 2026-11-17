"""Inspect the packed VoroPlex result schema."""

import numpy as np

import voroplex


points = np.random.default_rng(1).random((20, 3))
limits = [[0.0, 1.0]] * 3

# The cells are stored in a few flat arrays, rather than one object per cell.
# The *_offsets arrays mark where each cell's entries start: the vertices of
# cell i are rows cell_vertex_offsets[i] to cell_vertex_offsets[i + 1] - 1,
# and similarly for its facets (faces in 3D, edges in 2D), and for the vertices
# of each facet.
for representation in ("geometry", "per_cell", "topology"):
    result = voroplex.compute_voronoi_3d(
        points, limits, representation=representation, threads=2
    )
    print("\nrepresentation:", result.representation)
    print("dimension:", result.dimension)
    print("n_points / n_cells:", result.n_points, result.n_cells)
    print("measures:", result.measures.shape, result.measures.dtype)
    print("cell_vertex_offsets:", result.cell_vertex_offsets.shape)
    print("cell_facet_offsets:", result.cell_facet_offsets.shape)
    print("facet_neighbors:", result.facet_neighbors.shape)
    print("facet_vertex_offsets:", result.facet_vertex_offsets.shape)
    print("facet_vertices:", result.facet_vertices.shape)
    print("cell_surface_flags:", result.cell_surface_flags.shape)
    # The geometry and per-cell representations store the vertex coordinates;
    # the per-cell and topology representations store the signature of each
    # vertex, the IDs of the generators that define it.
    if result.has_vertices:
        print("vertices:", result.vertices.shape)
    if result.has_signatures:
        print(
            "vertex signatures:",
            result.vertex_signature_offsets.shape,
            result.vertex_signature_ids.shape,
        )

    # result.cell(i) gives a convenient view of one cell.
    cell = result.cell(0)
    print("cell 0 measure:", cell.measure)
    print("cell 0 neighbors:", cell.neighbors)
    print("cell 0 facets:", [facet.vertex_ids.tolist() for facet in cell.facets])
