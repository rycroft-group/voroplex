"""Meshes in which each vertex is shared between the cells that meet there.

Each function works in 2D and 3D, and finds the dimension from the result. In
2D, the facets of a cell are its edges, and in 3D, its faces.

The vertices are numbered in the same way by every function, for a given
result and device, so the outputs of several functions can be combined. Use
the same `device` for all of them.
"""

from . import reconstruct as _3d
from . import reconstruct_2d as _2d


def _dimension(result):
    return getattr(result, "dimension", None)


def _pick(result, function_2d, function_3d):
    return function_2d if _dimension(result) == 2 else function_3d


def compute_unique_vertices(result, points, device="cpu", radii=None):
    """Compute each vertex once, from the generators that define it.

    The result must use the topology representation. Each vertex is the point
    that is equidistant from its generators (or that lies on a boundary of the
    domain), so its position is found by solving a small linear system, and it
    is a differentiable function of the generator positions.

    Parameters
    ----------
    result : VoronoiResult
        A result computed with ``representation="topology"``.
    points : torch.Tensor or array_like, shape (N, dimension)
        The generator positions. Gradients flow back to them if they are a
        tensor with ``requires_grad=True``.
    device : str or torch.device
        The device for the returned tensor.
    radii : torch.Tensor, optional
        For a radical tessellation, the generator radii, so that gradients also
        flow back to them. By default, the radii stored in the result are used,
        as constants.

    Returns
    -------
    torch.Tensor, shape (n_unique_vertices, dimension)
    """
    function = _pick(result, _2d.unique_vertices_2d, _3d.unique_vertices_3d)
    return function(result, points, device=device, radii=radii)


def compute_padded_cell_vertices(result, points, device="cpu", radii=None):
    """Compute the vertices of each cell, padded to the same length.

    The positions are found as in :func:`compute_unique_vertices`, and then
    copied to each of the cells that share them.

    Returns
    -------
    vertices : torch.Tensor, shape (n_cells, max_vertices, dimension)
        The vertices of each cell, padded with zeros.
    counts : torch.Tensor, shape (n_cells,)
        The number of vertices of each cell.
    """
    function = _pick(result, _2d.padded_cell_vertices_2d, _3d.padded_cell_vertices_3d)
    return function(result, points, device=device, radii=radii)


def unique_facets(result, facet_type="all", device="cpu"):
    """List each facet (edge in 2D, face in 3D) once.

    Parameters
    ----------
    result : VoronoiResult
        A result with the geometry or topology representation.
    facet_type : {"all", "surface", "inner"}
        Which facets to list: every facet of every computed cell; the facets on
        the surface of the inner region, that is, between an inner and an outer
        cell, or on the boundary of the domain; or the facets between two inner
        cells.

    Returns
    -------
    facets : list of lists of int
        The vertices of each facet, as indices of the unique vertices.
    facet_info : numpy.ndarray, shape (n_facets, 2)
        The cell that owns each facet, and the cell on the other side, or a
        negative code for a boundary of the domain.
    """
    function = _pick(result, _2d.unique_facets_2d, _3d.unique_facets_3d)
    return function(result, facet_type=facet_type, device=device)


def cell_polytopes(result, device="cpu"):
    """Give each cell as a polygon (2D) or a polyhedron (3D).

    The vertices are indices of the unique vertices, so neighboring cells share
    them.

    Returns
    -------
    cells : list
        In 2D, the vertices of each cell, in order around it. In 3D, the faces
        of each cell, each a list of vertices.
    cell_ids : list of int
        The ID of each cell.
    """
    function = _pick(result, _2d.cell_polytopes_2d, _3d.cell_polytopes_3d)
    return function(result, device=device)


def surface_mesh(result, points=None, device="cpu"):
    """Give the surface of the inner region as a mesh with its own vertices.

    The surface is made of the facets between an inner and an outer cell, and
    the facets of inner cells on the boundary of the domain. In 3D, the
    vertices of each face are counter-clockwise as seen from outside the inner
    region, so that the right-hand rule gives the outward normal.

    Parameters
    ----------
    result : VoronoiResult
        A result computed with ``n_inner``. For the geometry and per-cell
        representations, the coordinates are those computed by Voro++. For the
        topology representation, they are recomputed from ``points``, and are
        differentiable with respect to them. Periodic domains, and results
        computed with ``neighbors=False``, are not supported.
    points : torch.Tensor or array_like, shape (N, dimension), optional
        The generator positions, required for the topology representation.
    device : str or torch.device
        The device for the returned tensors.

    Returns
    -------
    dict
        ``facet_vertex_ids`` (F, max_vertices), indices into ``vertex_coords``,
        padded with -100; ``vertex_coords`` (V, dimension);
        ``n_vertices_per_facet`` (F,); ``facet_mask`` (F, max_vertices), which
        is False for the padding; and ``facet_neighbors`` (F,), the outer cell,
        or the negative boundary code, on the other side of each facet. In 2D,
        every facet is an edge with two vertices.
    """
    function = _pick(result, _2d.surface_mesh_2d, _3d.surface_mesh_3d)
    return function(result, points=points, device=device)
