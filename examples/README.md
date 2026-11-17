# VoroPlex Examples

## Run

The examples import the `voroplex` package, so install it first, from the
project root, with `pip install -e .` (see the main README). Then run them from
the project root:

```bash
python examples/01_basic_usage.py
python examples/02_output_format.py
python examples/03_mesh_utilities.py
python examples/04_mesh_utilities_output.py
python examples/05_surface_analysis_demo.py
python examples/06_surface_analysis_output.py
python examples/07_differentiable_cvt_computation.py --density uniform
python examples/07_differentiable_cvt_computation.py --density adaptive
python examples/08_2d_basic_usage.py
python examples/09_2d_mesh_utilities.py
python examples/10_2d_differentiable_cvt_computation.py --density uniform
python examples/10_2d_differentiable_cvt_computation.py --density adaptive
python examples/11_2d_radical_tessellation.py
python examples/12_lloyd_algorithm.py
python examples/13_2d_density_weighted_cvt_computation.py
python examples/14_3d_density_weighted_cvt_computation.py
python examples/15_2d_sdf_shape_meshing.py
python examples/16_2d_foam_orientation_optimization.py
python examples/17_3d_foam_orientation_optimization.py --target=uniaxial
python examples/17_3d_foam_orientation_optimization.py --target=needle
python examples/17_3d_foam_orientation_optimization.py --target=circular
```

## Files

### `01_basic_usage.py`

Public API basics:

- The default geometry representation: each cell's vertices, faces, and
  neighbors, as computed by Voro++
- Inner/outer cells with `n_compute=n_inner`
- Differentiable vertices with the per-cell representation and
  `cell_vertices`
- Shared vertices recomputed in PyTorch with the topology representation

### `02_output_format.py`

Inspection of the packed arrays of all three representations, and the
arrays specific to each one

### `03_mesh_utilities.py`

Demonstration of the 3D mesh utilities: vertex reconstruction, unique faces,
polyhedral cells, and the surface mesh

### `04_mesh_utilities_output.py`

Mesh utility report writer → `mesh_utilities_output.txt`

### `05_surface_analysis_demo.py`

Surface analysis:

- Extract surface mesh from Voronoi
- Compute 19 geometric properties
- Show all 5 property categories with sample output
- Demonstrate differentiability

### `06_surface_analysis_output.py`

Surface analysis report writer → `surface_analysis_output.txt`

- All 19 properties explained with shapes, dtypes, and sample values
- Organized by 5 categories: basic geometry, triangle subdivision, adjacency,
  vertex neighbors, and face neighbors

### `07_differentiable_cvt_computation.py`

3D differentiable Centroidal Voronoi Tessellation (CVT) optimization, using
the per-cell representation: `cell_vertices` gives the vertices of every
cell as a differentiable tensor, and `cell_fan_simplices` splits every cell
into tetrahedra that share its generator. Two density modes:

- `--density uniform` (default): the exact energy, via a closed-form
  tetrahedral fan-triangulation decomposition -- no approximation.
- `--density adaptive`: a user-specified, non-uniform density function
  (a Gaussian bump), integrated with numerical quadrature (a 4-point,
  degree-2-exact rule) since no closed form exists for an arbitrary
  density. Verified to reduce to the exact formula (same energy value,
  same gradient) when the density is constant.

### `08_2d_basic_usage.py`

Basic 2D usage with the default geometry representation, including lazy cell
and edge views

### `09_2d_mesh_utilities.py`

2D reconstruction, unique edge extraction, polygonal cells, boundary mesh
information, and polygon analysis

### `10_2d_differentiable_cvt_computation.py`

2D counterpart of `07_differentiable_cvt_computation.py`, also using the
per-cell representation: the same `--density {uniform,adaptive}` modes
(exact triangular-fan energy vs. numerical quadrature), plus before/after
tessellation and energy plots.

### `11_2d_radical_tessellation.py`

Compares the standard Voronoi tessellation and the radical (power)
tessellation of a packing of disks with different radii, and shows the empty
cells that appear when all of the radii are increased by the same amount.

### `12_lloyd_algorithm.py`

Lloyd's algorithm in a periodic 2D domain, using `representation=None` to
compute only the centroid and second moment of each cell, with the energy
reported relative to that of a hexagonal lattice. `--plot` draws the final
tessellation.

### `13_2d_density_weighted_cvt_computation.py`

Density-weighted CVT in 2D with Adam, using the per-cell representation and
a 3-point quadrature rule on the triangles of each cell. `--density=NAME`
gives a density field, and `--sizing=NAME` a target cell size h, converted to
the density h^-4. Saves an animation, the final mesh, and the parameters.
The docstring lists the commands for the figures in Sec. 5.1 of the paper.

### `14_3d_density_weighted_cvt_computation.py`

3D counterpart of example 13, with density (`--density`) or sizing
(`--sizing`, density h^-5) fields, including fields that cluster cells near
the edges of the cube. Each cell is split into tetrahedra from its generator
through the centre of each face, integrated with a 4-point rule
(`--quad-points`) and optional refinement (`--quad-refine`). Optional
penalties act on cell volumes and on the faces on the boundary of the domain. Writes VTK files for ParaView
(faces, edges, generators, and the density field), optionally clipped by a
plane (`--clip-axis`), without needing the VTK package.

### `15_2d_sdf_shape_meshing.py`

Meshing a 2D shape given by a signed distance function (SDF), for Sec. 5.2 of
the paper. The generators inside the shape are listed first, and the interface
between the inner and outer cells is moved onto the boundary of the shape,
while the inner cells are kept regular by a density-weighted CVT energy, with
smaller cells near the boundary. Uses the per-cell representation. The
defaults are the paper's settings; `--shape` chooses one of the six shapes.
`--paper-figure` (with `--shape=flower`) also draws the flower figure of the
paper, and after the circle, star, rounded square, heart, and peanut have been
run into the same output directory, `--paper-gallery` draws the paper's row of
five shapes from their saved results. The docstring lists the commands. The
heart and the star take about ten times as long as the other shapes, because
their signed distance functions are computed from polygons.

### `16_2d_foam_orientation_optimization.py`

Anisotropic foam design in 2D, used for Sec. 5.3 of the paper. Optimizes the
generators so that every cell's length-weighted edge-normal tensor matches a
target built from a preferred axis and an anisotropy strength alpha, plus a
penalty on the variation in cell area, without which the first term is
satisfied just as well by cells collapsed into slivers. `--angle` and
`--anisotropy` set the target, `--field` chooses a fixed or a radial axis, and
`--movie` saves the evolution. Uses the per-cell representation. The defaults
are the paper's settings, and the script prints the alignment M of the paper
at the start and the end.

### `17_3d_foam_orientation_optimization.py`

3D counterpart of example 16, with the same per-cell objective. Walls are
polygonal faces weighted by area, and the cell measure is volume. `--target`
selects the axis field: `uniaxial` gives plate-like cells stacked along x,
`needle` gives columns along z, and `circular` gives cells arranged in curved
slats around the vertical centre line. `--vtu` writes files for ParaView. Uses
the per-cell representation. The defaults are the paper's settings, a run takes
a few minutes, and the script prints the alignment M of the paper, the mean of
u^T A_c u, and, for the columnar target, the median aspect ratio of the
cross-sections.

### Reproducing the numbers in Sec. 5.3

The runs reported in the paper were done on an Apple M4 Pro under macOS 15.7
with Python 3.12.

With one thread, which is the default, a run is exactly reproducible on a given
computer. With several threads, the contributions to the gradient are added in
an order that changes from run to run, so the results change slightly. In two
runs of the columnar target with eight threads, for example, M was 0.9919 and
0.9923, and the mean zz component was 0.1211 and 0.1198. Short runs, such as
300 generators for 100 iterations, can agree in every printed digit, because
the differences take many iterations to grow.

Values can also differ in the last digit from one computer to another, even
with one thread. The same calculation can be rounded differently on different
computers, by about one part in 10^16, for two reasons. First, compilers for
some processors, such as Apple's ARM processors, combine a multiplication and
an addition into one operation, a fused multiply-add, which is rounded once
instead of twice; this changes the vertex coordinates computed by Voro++.
Second, PyTorch adds up arrays with code that depends on the processor, and so
in a different order. Different versions of the libraries can also round
differently. The tessellation changes discontinuously as the generators move,
so such a small difference early in a run can decide a later topology change,
and then grow over a thousand iterations. For the columnar target, a Linux
computer with an Intel processor gives M = 0.9929, a mean zz component of
0.1200, and a median aspect ratio of 1.268, while the Apple M4 Pro gives 0.9920,
0.1208, and 1.253.

### Supporting modules for Sec. 5.3

Imported by examples 16 and 17, except where a script is noted as runnable.

- `foam_figure_style.py` — shared styling for the section's figures.
- `compose_figure_row.py` — composes separate panels into one labelled figure.
- `foam_schematic.py` — draws the schematic distinguishing a wall's normal, the
  wall's own direction, and the direction a cell is elongated in. Run it with
  an output directory.
- `render_foam_pyvista.py` — renders a saved 3D run as a whole cube and as a
  corner cutaway. Run it with a history archive.
- `make_foam_3d_grid.py` — composes those six renders into the 3x2 grid of the
  paper. Run it with the directory holding the runs.
- `replot_combined_2d.py`, `replot_foam_3d.py` — redraw the figures from a
  saved archive, so a layout can be changed without repeating the training.
