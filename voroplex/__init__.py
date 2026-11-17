"""
VoroPlex: parallel and differentiable Voronoi tessellations in 2D and 3D.

VoroPlex computes Voronoi tessellations with the multi-threaded version of the
C++ library Voro++, and returns the result in NumPy arrays. It can also return
the vertices of each cell as a PyTorch tensor, whose gradient flows back to the
generator positions.

Quick start
-----------
>>> import numpy as np
>>> import voroplex
>>>
>>> points = np.random.default_rng(0).random((100, 3))
>>> result = voroplex.compute_voronoi_3d(points, [[0, 1]] * 3)
>>> print(len(result), result.volumes.sum())

Computing tessellations
-----------------------
compute_voronoi_2d() : Tessellation of 2D points
compute_voronoi_3d() : Tessellation of 3D points
compute_properties() : Further cell properties for an existing result
VoronoiResult : The result of a tessellation

The representation argument chooses what is stored: "geometry" (the default)
stores the vertices, facets, and neighbors of each cell, as computed by
Voro++; "per_cell" also stores the signature of each vertex, for
cell_vertices(); "topology" stores the signatures without the coordinates, for
meshes in which the cells share their vertices; and None stores only the cell
properties.

The functions below work in 2D and 3D, and find the dimension from the
result. The facets of a cell are its edges in 2D, and its faces in 3D.

Differentiable vertices and meshes (require PyTorch)
-----------------------------------------------------
cell_vertices() : Each cell's own vertices, differentiable (per_cell)
cell_fan_simplices() : Triangles (2D) or tetrahedra (3D) of each cell
compute_unique_vertices() : Shared vertices, recomputed from the generators
compute_padded_cell_vertices() : Each cell's vertices, from the shared vertices
unique_facets() : Each facet once
cell_polytopes() : Each cell as a polygon (2D) or a polyhedron (3D)
surface_mesh() : Mesh of the surface of the inner region
TORCH_AVAILABLE : Whether PyTorch, and so these functions, are available

Analysis (requires PyTorch)
---------------------------
analysis.compute_surface_geometry() : Geometry of a polygon surface mesh (3D)
analysis.compute_polygon_geometry() : Geometry of polygon cells (2D)

Export
------
export_vtu() : Facets, as a VTU file for ParaView
export_npz() : Save a VoronoiResult to a compressed NumPy archive
load_npz() : Load a VoronoiResult from a NumPy archive

See the README and the examples/ directory for more.
"""

__version__ = "1.0.0"

from ._core import compute_properties, compute_voronoi_2d, compute_voronoi_3d
from ._result import VoronoiResult


try:
    from ._mesh import (
        cell_fan_simplices,
        cell_polytopes,
        cell_vertices,
        compute_padded_cell_vertices,
        compute_unique_vertices,
        surface_mesh,
        unique_facets,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    def _requires_torch(*args, **kwargs):
        raise ImportError(
            "This function requires PyTorch. Install it with "
            "'pip install torch'; it can run on the CPU, without a GPU."
        )

    cell_vertices = _requires_torch
    cell_fan_simplices = _requires_torch
    compute_unique_vertices = _requires_torch
    compute_padded_cell_vertices = _requires_torch
    unique_facets = _requires_torch
    cell_polytopes = _requires_torch
    surface_mesh = _requires_torch


from . import analysis
from .export import export_npz, export_vtu, load_npz

__all__ = [
    "compute_voronoi_2d",
    "compute_voronoi_3d",
    "compute_properties",
    "VoronoiResult",
    "cell_vertices",
    "cell_fan_simplices",
    "compute_unique_vertices",
    "compute_padded_cell_vertices",
    "unique_facets",
    "cell_polytopes",
    "surface_mesh",
    "TORCH_AVAILABLE",
    "analysis",
    "export_vtu",
    "export_npz",
    "load_npz",
    "__version__",
]
