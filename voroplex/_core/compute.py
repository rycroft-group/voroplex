"""Public 3D Voronoi computation, and the code shared with 2D."""

import numpy as np

from voroplex._lib import voroplusplus_
from voroplex._result import VoronoiResult

from ._validate import (DEFAULT_PROPERTIES, PROPERTIES,
                        normalize_properties, normalize_representation,
                        validate_compute_inputs, validate_radii)


def compute_voronoi_3d(
    points,
    limits=None,
    n_compute=None,
    n_inner=None,
    threads="auto",
    periodic=False,
    representation="geometry",
    neighbors=True,
    radii=None,
    properties=None,
):
    """Compute a packed 3D Voronoi tessellation.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Generator coordinates. When inner and outer points are used, put all
        inner points first. No two generators may coincide (in a periodic
        domain, after wrapping into the box): coincident generators raise a
        ValueError, since their Voronoi cells are not defined.
    limits : array_like, shape (3, 2), optional
        Domain bounds. A 5% margin around the points is used when omitted.
        The domain is closed, so points exactly on its boundary are allowed.
    n_compute : int, optional
        Compute cells with IDs below this value while inserting all points.
    n_inner : int, optional
        Points below this value are inner. The interface and boundary categories apply
        only to those inner cells.
    threads : positive int or ``"auto"``
        OpenMP thread count.
    periodic : bool or sequence of 3 bools
        Periodic axes. Supported by the geometry and per_cell representations.
    representation : {``"geometry"``, ``"per_cell"``, ``"topology"``, None}
        ``"geometry"`` (the default) stores the Voro++ cell-local coordinates
        and facets. ``"per_cell"`` also stores the cell-local vertex
        signatures, for differentiation with :func:`cell_vertices`.
        ``"topology"`` stores the signatures without coordinates, for building
        meshes with shared vertices (for example with
        :func:`compute_unique_vertices`). ``None`` stores no information about
        the cells themselves, only the requested ``properties``: this uses
        far less memory, and is faster. It cannot be used with ``n_inner``.
    neighbors : bool
        Whether to record the neighbor of each face (the default). With
        ``neighbors=False``, Voro++ computes the cells with a simpler class that
        does not track neighbors, which is a few percent faster; the result
        then has no neighbor IDs or boundary flags. Only available with the
        geometry representation, and not with ``n_inner``.
    radii : array_like, shape (N,), optional
        Generator radii. If given, the result is the radical (power)
        tessellation: the cell of generator i is the region where
        ``|x - p_i|^2 - r_i^2`` is smaller than for any other generator. The
        cells are still convex polyhedra, and with equal radii they are the
        standard Voronoi cells. A generator can lie outside its own cell, and
        a cell can be empty (it then has measure 0, and no vertices or faces).
        Coincident generators raise a ValueError even if their radii differ.
        Voro++ searches further for neighbors when the largest radius is large
        compared with the spacing of the points, so a wide range of radii
        makes the computation slower.
    properties : sequence of str, optional
        The properties to compute for each cell, as arrays that are available
        from the result. The default, ``None``, computes only ``"volume"``,
        and ``()`` computes none. The available properties are:

        - ``"volume"``: the volume of the cell (``result.volumes``);
        - ``"surface_area"``: its surface area (``result.surface_areas``);
        - ``"centroid"``: its centroid, shape (N, 3) (``result.centroids``);
        - ``"n_faces"``: its number of faces, which is the number of
          neighbors (``result.n_faces``);
        - ``"n_vertices"``, ``"n_edges"``: its numbers of vertices and
          edges (``result.n_vertices``, ``result.n_edges``);
        - ``"total_edge_length"``: the total length of its edges
          (``result.total_edge_lengths``);
        - ``"max_radius"``: the largest distance from the generator to a vertex
          of its cell (``result.max_radii``);
        - ``"second_moment"``: the integral of ``|x - p_i|^2`` over the cell,
          which is its energy in a centroidal Voronoi tessellation
          (``result.second_moments``);
        - ``"second_moment_tensor"``: the integral of
          ``(x - p_i)(x - p_i)^T`` over the cell, shape (N, 3, 3)
          (``result.second_moment_tensors``).

        They are computed by Voro++, relative to each generator, during the
        parallel loop over the cells, and they are not differentiable. For an
        empty cell in a radical tessellation, the sizes and counts are 0, and
        the centroid and maximum radius are NaN. In a periodic domain, a
        centroid can lie outside the box.

    Returns
    -------
    VoronoiResult
        A read-only packed result with lazy cell and facet views.
    """
    return _compute(
        3, points, limits, n_compute, n_inner, threads, periodic,
        representation, neighbors, radii, properties,
    )


def _compute(dimension, points, limits, n_compute, n_inner, threads,
             periodic, representation, neighbors, radii, properties):
    """The computation shared by compute_voronoi_3d and compute_voronoi_2d."""
    representation = normalize_representation(representation)
    if properties is None:
        properties = DEFAULT_PROPERTIES[dimension]
    names = normalize_properties(properties, dimension)
    neighbors = bool(neighbors)
    if representation is None:
        if n_inner is not None:
            raise ValueError(
                "n_inner needs the cells' neighbors, so it cannot be used with "
                "representation=None"
            )
        # No neighbor information is stored, so the simpler cell class is used.
        neighbors = False
    elif not neighbors and representation != "geometry":
        raise ValueError(
            "neighbors=False is only available with representation='geometry'; "
            "the per_cell and topology representations need the neighbors "
            "to build vertex signatures"
        )
    if not neighbors and n_inner is not None:
        raise ValueError(
            "n_inner needs neighbor information, so it cannot be used with "
            "neighbors=False"
        )
    (
        pts,
        lims,
        n_compute_val,
        n_inner_val,
        threads,
        periodic_arr,
    ) = validate_compute_inputs(
        points,
        limits,
        dimension=dimension,
        n_compute=n_compute,
        n_inner=n_inner,
        threads=threads,
        periodic=periodic,
    )
    radii_arr = validate_radii(radii, len(pts))
    if representation == "topology" and periodic_arr.any():
        raise ValueError(
            "representation='topology' does not support periodic "
            "boundaries; use representation='geometry' or 'per_cell'"
        )

    compute = (voroplusplus_.compute_voronoi_packed_3d if dimension == 3
               else voroplusplus_.compute_voronoi_packed_2d)
    packed, arrays = compute(
        points=pts,
        limits=lims,
        nt=threads,
        optNumPtGrid=5.6 if dimension == 3 else 4.2,
        periodic=periodic_arr.tolist(),
        n_compute=n_compute_val,
        n_inner=n_inner_val,
        topology=representation in ("topology", "per_cell"),
        cell_vertices=representation == "per_cell",
        neighbors=neighbors,
        radii=radii_arr,
        properties=[PROPERTIES[dimension][name] for name in names],
        cells=representation is not None,
    )
    props = {name: arrays[PROPERTIES[dimension][name]] for name in names}
    cell_data = {}
    if packed is not None:
        cell_data = dict(
            vertices=packed[0] if representation != "topology" else None,
            cell_vertex_offsets=packed[1],
            cell_facet_offsets=packed[2],
            facet_neighbors=packed[3] if neighbors else None,
            facet_vertex_offsets=packed[4],
            facet_vertices=packed[5],
            cell_surface_flags=packed[6],
            vertex_signature_offsets=(
                packed[7] if representation != "geometry" else None
            ),
            vertex_signature_ids=(
                packed[8] if representation != "geometry" else None
            ),
        )
    return VoronoiResult(
        dimension=dimension,
        representation=representation,
        limits=lims,
        periodic=periodic_arr,
        n_points=len(pts),
        n_inner=n_inner_val,
        threads=threads,
        n_cells=n_compute_val,
        properties=props,
        has_neighbors=neighbors,
        radii=radii_arr,
        _trusted=True,
        **cell_data,
    )


def compute_properties(result, points, properties, threads=None):
    """Compute cell properties for an existing result.

    The tessellation is computed again, with only the requested properties,
    using the result's limits, periodicity, radii, and number of cells. This
    is fast, since no information about the cells is stored, and it gives the
    same values as computing the properties with the original tessellation.

    Parameters
    ----------
    result : VoronoiResult
        A result from :func:`compute_voronoi_3d` or :func:`compute_voronoi_2d`,
        with any representation, or from :func:`load_npz`.
    points : array_like, shape (N, d)
        The generator positions that the result was computed from. A result
        does not store them, but :func:`export_npz` can save them with it.
    properties : sequence of str
        The properties to compute, with the names used by
        :func:`compute_voronoi_3d` (3D) or :func:`compute_voronoi_2d` (2D).
    threads : positive int or ``"auto"``, optional
        OpenMP thread count. By default, the result's thread count.

    Returns
    -------
    dict
        A mapping from each property name to its read-only array, in the same
        form as ``result.properties``.

    Raises
    ------
    ValueError
        If the points do not match the result. This is checked against the
        result's stored areas or volumes if it has them, and otherwise against
        the number of vertices of each stored cell.
    """
    dimension = result.dimension
    names = normalize_properties(properties, dimension)
    shape = np.shape(points)
    if len(shape) != 2 or shape[1] != dimension:
        raise ValueError(
            "points must have shape (N, {}), got {}".format(dimension, shape)
        )
    if shape[0] != result.n_points:
        raise ValueError(
            "points must contain the same {} generators used for the "
            "Voronoi computation, got {}".format(result.n_points, shape[0])
        )
    measure = "volume" if dimension == 3 else "area"
    check = None
    if measure in result.properties:
        check = "measure"
    elif result.has_cells:
        check = "n_vertices"
    request = list(names)
    if check == "measure" and measure not in request:
        request.append(measure)
    if check == "n_vertices" and "n_vertices" not in request:
        request.append("n_vertices")
    new = _compute(
        dimension, points, result.limits, len(result), None,
        result.threads if threads is None else threads, result.periodic,
        None, False, result.radii, request,
    )
    if check == "measure":
        old, recomputed = result.properties[measure], new.properties[measure]
        scale = max(float(np.abs(old).max()), 1e-300)
        if not np.allclose(recomputed, old, rtol=1e-9, atol=1e-12 * scale):
            raise ValueError(
                "the points do not match the result: the recomputed cell "
                "{}s differ from the stored ones".format(measure)
            )
    elif check == "n_vertices":
        if not np.array_equal(new.properties["n_vertices"],
                              np.diff(result.cell_vertex_offsets)):
            raise ValueError(
                "the points do not match the result: the recomputed cells "
                "have different numbers of vertices"
            )
    return {name: new.properties[name] for name in names}
