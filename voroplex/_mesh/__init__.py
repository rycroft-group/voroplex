"""Differentiable vertices, and meshes in which the cells share their vertices."""

from .meshes import (
    cell_polytopes,
    compute_padded_cell_vertices,
    compute_unique_vertices,
    surface_mesh,
    unique_facets,
)
from .per_cell import cell_fan_simplices, cell_vertices

__all__ = [
    "cell_vertices",
    "cell_fan_simplices",
    "compute_unique_vertices",
    "compute_padded_cell_vertices",
    "unique_facets",
    "cell_polytopes",
    "surface_mesh",
]
