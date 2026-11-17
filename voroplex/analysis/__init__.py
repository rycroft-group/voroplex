"""Differentiable geometry of polygon meshes.

compute_surface_geometry gives the areas, normals, perimeters, adjacency, and
neighbor averages of the faces of a polygon surface mesh in 3D, such as the
output of surface_mesh. compute_polygon_geometry gives the areas, centroids,
perimeters, edge lengths, and adjacency of a set of polygon cells in 2D. Both
work on any polygon mesh, not only on Voronoi output, and both require
PyTorch.
"""

try:
    from .polygon import compute_polygon_geometry
    from .surface import compute_surface_geometry
except ImportError as _error:
    _import_error = _error

    def _requires_torch(*args, **kwargs):
        raise ImportError(
            "This function requires PyTorch. Install it with "
            "'pip install torch'; it can run on the CPU, without a GPU. "
            f"(Original error: {_import_error})"
        )

    compute_surface_geometry = _requires_torch
    compute_polygon_geometry = _requires_torch


__all__ = ["compute_surface_geometry", "compute_polygon_geometry"]
