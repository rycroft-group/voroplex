# Changelog

## 1.0.0 (not yet released)

Changes relative to the original VoroPlex code, before its first release.

### Changed

- **Names.** The public names were made consistent:
  - `compute_voronoi` is now `compute_voronoi_3d`, alongside
    `compute_voronoi_2d`, so that the dimension is always clear.
  - The mesh functions work in both 2D and 3D, and find the dimension from
    the result, replacing pairs of 3D and 2D functions:
    `compute_unique_vertices` (was `reconstruct_vertices` and
    `reconstruct_vertices_2d`), `compute_padded_cell_vertices`
    (`get_cell_vertex_positions`), `unique_facets` (`extract_unique_faces`,
    `extract_unique_edges_2d`), `cell_polytopes` (`build_polyhedral_cells`,
    `build_polygonal_cells_2d`), and `surface_mesh` (`get_surface_mesh_info`,
    `get_boundary_mesh_info_2d`). Functions that calculate begin with
    `compute_`; functions that extract data do not.
  - Their arguments are `result` (was `voro_result`) and `facet_type` (was
    `face_type` or `edge_type`), whose default is now `"all"` everywhere. The
    keys of `surface_mesh` are the same in 2D and 3D: `facet_vertex_ids`,
    `vertex_coords`, `n_vertices_per_facet`, `facet_mask`, and
    `facet_neighbors`.
  - The faces of the domain are its *boundary*, rather than its walls, which
    in Voro++ are extra constraints on the cells: `result.boundary_cells` (was
    `wall_cells`) are the cells that touch it, `result.surface_cells` (was
    `boundary_cells`) are the cells that touch it or an outer cell, and the
    flags are `cell_surface_flags`.
  - The cell properties `num_faces`, `num_vertices`, and `num_edges` are now
    `n_faces`, `n_vertices`, and `n_edges`, like `n_points` and `n_cells`.
  - `surface_analysis` and `cell_analysis_2d` are now one subpackage,
    `analysis`, with `compute_surface_geometry` and
    `compute_polygon_geometry`; their output keys use `n_` and `tri_`
    consistently. One flag, `TORCH_AVAILABLE`, replaces three.
  - The examples that compute CVTs are named `..._cvt_computation.py`, and
    example 15 is `15_2d_sdf_shape_meshing.py`.

  Archives saved with the earlier names can still be loaded.
- **The default representation is now `"geometry"`**, which gives each cell's
  vertices, faces, and neighbors as computed by Voro++. It was previously
  `"topology"`. Code that uses the topology functions (`compute_unique_vertices`,
  `compute_padded_cell_vertices`, and their 2D versions) without passing a
  representation must now pass `representation="topology"`.
- Generators on, or just below, the upper boundary of the domain (for example
  `x = 1` with limits `[0, 1]`) are now accepted: the domain is closed. Previously, Voro++
  silently dropped such points, and the computation failed with an unclear
  error.
- Coincident generators now raise a `ValueError` naming one such pair.
  Previously, they gave overlapping cells in 2D, and an unclear error in 3D.
- When Voro++ cannot compute some cells, the error message says how many.
- The vertex systems in the topology reconstruction are solved relative to
  the base generator of each vertex. This fixes a large loss of precision for
  small cells far from the origin (for example, cells of size 1e-6 near
  x = 1, which had errors of several percent of the cell size).
- A singular square vertex system in the topology reconstruction now raises
  an error, instead of being solved silently with a pseudoinverse. The test is
  relative, so it does not depend on the units of length.
- `surface_mesh` lists the vertices of each face counter-clockwise
  as seen from outside the inner region, so that the face normals from
  `analysis.compute_surface_geometry` point outward, as
  documented. They previously pointed inward. The normals are now computed
  with Newell's method from all of the triangles of each face, and the
  tolerances in the surface and 2D cell analysis are relative.
- `compute_polygon_geometry` measures coordinates relative to each cell,
  so that small cells far from the origin keep their precision.
- The meshes with shared vertices (`unique_facets`,
  `surface_mesh`, `export_vtu`, and the 2D versions) raise an error
  for all periodic results. Previously, per-cell results in periodic domains
  were not rejected, although their vertices cannot be matched between cells
  correctly.
- NPZ archives use schema version 2, which allows results without neighbor
  information, stores the radii of radical tessellations and the cell
  properties, and allows results without cells (`representation=None`). Version 1
  archives can still be read.
- The 3D tetrahedra from `cell_fan_simplices` are positively oriented, like
  the 2D triangles: their signed volumes are positive when the generator is
  inside its cell. Previously they were all negative. The examples and tests
  now use signed areas and volumes, which also give correct integrals when
  a generator lies outside its own cell (in a radical tessellation).
- The package metadata is defined only in `pyproject.toml`.
- VoroPlex requires Python 3.10 or later, and PyTorch 1.11 or later for the
  differentiable functions; the `export` extra also installs PyTorch, which
  `export_vtu` needs. All of the authors are listed in the package metadata,
  and the copyright is held by the VoroPlex authors.
- The extension is linked with the compiler from `config.mk`, and with that
  compiler's own OpenMP runtime, so that two OpenMP runtimes are not loaded
  into one process (which crashed on a Mac with Anaconda). A linker command
  set in `LDSHARED` or `LDCXXSHARED` is used as it is.
- Examples 13 and 14 (density-weighted CVTs in 2D and 3D, used for Sec. 5.1
  of the paper) use the per-cell representation and the current API. The 3D
  example splits each cell into tetrahedra from its generator through the
  centre of each face, so that the integrals do not depend on the numbering
  of the vertices, which can differ between runs with several threads. Its
  default quadrature is now the 4-point rule, since the 1-point rule gives
  gradient errors of tens of percent on these tetrahedra.

- **Voro++ is now a Git submodule** (`voroplex/_lib/voro`), pinned to a
  commit of the Voro++ `dev` branch, instead of a copy of its source in
  `voroplex/_lib/src_parallel`. Clone with `--recursive`. The copy had
  fallen behind Voro++, and the fix for generators on the upper boundary, which
  had been made only in the copy, is now in Voro++ itself. The Voro++ sources
  are compiled directly into the extension, so `make` is no longer needed.
  This also brings in an upstream fix for a rounding error in the radical
  tessellation.

### Added

- **Cell properties**, with the `properties` argument of `compute_voronoi_3d`
  and `compute_voronoi_2d`: the volume or area, surface area or perimeter,
  centroid, numbers of faces, vertices, and edges, total edge length, maximum
  radius, and second moment (scalar and tensor) of each cell. They are
  computed by Voro++ in the parallel loop over the cells, and are available as
  attributes of the result (`volumes`, `centroids`, `second_moments`, and so
  on) and in `result.properties`. By default, only the volumes (3D) or areas
  (2D) are computed, as before. The second moments use new functions in
  Voro++.
- **`representation=None`**, which stores only the requested properties, and
  nothing about the cells themselves. For $10^6$ generators in 3D, this uses
  about 24 bytes per cell for the centroids, instead of about 1.2 kB for the
  geometry representation, and it is about 25% faster. A new example,
  `examples/12_lloyd_algorithm.py`, uses it for Lloyd's algorithm.
- `compute_properties(result, points, properties)`, which computes further
  cell properties for an existing result, with the result's own settings, and
  checks that the points match it.
- `export_npz(..., points=points)` saves the generator positions with a
  result, and `load_npz(..., return_points=True)` returns them.
- **Radical (power) tessellations**, with the `radii` argument of
  `compute_voronoi_3d` and `compute_voronoi_2d`, using Voro++'s
  `container_poly_3d` and `container_poly_2d`. Empty cells are allowed, and
  have measure 0. The radii are stored in the result (`radii`,
  `is_radical`) and in NPZ archives. `cell_vertices`, `compute_unique_vertices`
  and `compute_padded_cell_vertices` (and their 2D versions) accept the radii
  as a tensor, and give gradients with respect to them.
- **The `"per_cell"` representation and `cell_vertices`**: each cell's
  vertices as a PyTorch tensor whose gradient flows to the generator
  positions, computed from the analytic derivative of each vertex in a
  parallel C++ loop. It does not need the vertices to be matched between
  cells, works in periodic domains, and stays accurate near lattices, where
  the topology reconstruction is ill-conditioned.
- `cell_fan_simplices`, which splits every cell into triangles or tetrahedra
  that share its generator, for integrals over the cells.
- Examples 15, 16, and 17, for the shape meshing and foam design applications
  in Secs. 5.2 and 5.3 of the paper. Examples 16 and 17 use the per-cell
  representation, which makes a 3D run several hundred times faster than with
  the topology representation, and their defaults are the paper's settings.
- `neighbors=False`, which uses Voro++'s simpler cell class, without
  neighbor information, for the geometry representation.
- `VoronoiResult.has_neighbors`, and a `repr` for `VoronoiResult`.
- A pytest test suite in `tests/`.
- Building with `VOROPLEX_PROFILE=1` times the Voro++ part of the cell loop
  separately, for benchmarks.

### Performance

- The tessellation wrapper no longer has large serial overheads: each thread
  stores its cells in its own buffers, which are reserved in advance, and the
  result is copied into uninitialized NumPy arrays in parallel. For
  5 million points on 28 threads, a 3D tessellation became about 3 times
  faster, and a 2D tessellation about 8 times faster.
- The 2D blocks hold 4.2 particles on average, as recommended by Voro++,
  instead of 8.

### Fixed

- `cell_vertices` raises a clear error if the points differ from the
  positions used for the tessellation, and for second derivatives
  (`create_graph=True`), which are not supported and were previously silently
  incomplete.
- Pickling a per-cell result no longer includes its cached PyTorch tensors.
- An empty surface mesh no longer crashes
  `compute_surface_geometry`.
