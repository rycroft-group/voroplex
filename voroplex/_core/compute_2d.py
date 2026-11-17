"""Public 2D Voronoi computation."""

from .compute import _compute


def compute_voronoi_2d(
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
    """Compute a packed 2D Voronoi tessellation.

    The parameters and representations match :func:`compute_voronoi_3d`;
    points have shape ``(N, 2)`` and limits have shape ``(2, 2)``. With
    ``radii``, the result is the radical (power) tessellation.

    The available ``properties`` are ``"area"`` (the default),
    ``"perimeter"``, ``"centroid"``, ``"n_vertices"`` (which is also the
    number of edges and of neighbors), ``"max_radius"``, ``"second_moment"``,
    and ``"second_moment_tensor"`` (shape (N, 2, 2)). The arrays are available
    from the result as ``areas``, ``perimeters``, ``centroids``,
    ``n_vertices``, ``max_radii``, ``second_moments``, and
    ``second_moment_tensors``.
    """
    return _compute(
        2, points, limits, n_compute, n_inner, threads, periodic,
        representation, neighbors, radii, properties,
    )
