"""Shared validation helpers for VoroPlex compute entry points."""

import os
from numbers import Integral

import numpy as np


VALID_REPRESENTATIONS = ("geometry", "topology", "per_cell", None)

# The per-cell properties that can be computed, for each dimension: the public
# name, and the internal name used by the C++ layer.
PROPERTIES = {
    2: {
        "area": "measure",
        "perimeter": "boundary",
        "centroid": "centroid",
        "n_vertices": "num_vertices",
        "max_radius": "max_radius",
        "second_moment": "second_moment",
        "second_moment_tensor": "second_moment_tensor",
    },
    3: {
        "volume": "measure",
        "surface_area": "boundary",
        "centroid": "centroid",
        "n_faces": "num_faces",
        "n_vertices": "num_vertices",
        "n_edges": "num_edges",
        "total_edge_length": "edge_length",
        "max_radius": "max_radius",
        "second_moment": "second_moment",
        "second_moment_tensor": "second_moment_tensor",
    },
}
DEFAULT_PROPERTIES = {2: ("area",), 3: ("volume",)}


def validate_points(points, dimension):
    """Return a finite float64 point array with shape (N, dimension)."""
    try:
        pts = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be convertible to a float64 array") from exc

    if pts.ndim != 2 or pts.shape[1] != dimension:
        raise ValueError(
            f"points must have shape (N, {dimension}), got {pts.shape}"
        )
    if pts.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    return pts


def validate_radii(radii, n_points):
    """Return None, or a finite, non-negative float64 array of shape (N,)."""
    if radii is None:
        return None
    try:
        r = np.asarray(radii, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("radii must be convertible to a float64 array") from exc
    if r.shape != (n_points,):
        raise ValueError(
            f"radii must have shape ({n_points},), one per point, got {r.shape}"
        )
    if not np.isfinite(r).all():
        raise ValueError("radii must contain only finite values")
    if np.any(r < 0):
        raise ValueError("radii must be non-negative")
    return r


def resolve_limits(points, limits, dimension):
    """Return validated limits, auto-computing a 5% margin when absent.

    Also checks that the points are finite. The minimum and maximum of each
    coordinate are non-finite if any point is non-finite (NaN propagates
    through min and max), so two reductions check both finiteness and the
    limits without building any temporary arrays.
    """
    # One column at a time: NumPy's axis=0 reduction is slow for an (N, d)
    # array with small d.
    mins = np.array([points[:, k].min() for k in range(dimension)])
    maxs = np.array([points[:, k].max() for k in range(dimension)])
    if not (np.isfinite(mins).all() and np.isfinite(maxs).all()):
        raise ValueError("points must contain only finite values")
    if limits is None:
        ranges = maxs - mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        return np.column_stack([mins - 0.05 * ranges, maxs + 0.05 * ranges])

    try:
        lims = np.asarray(limits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("limits must be convertible to a float64 array") from exc

    expected_shape = (dimension, 2)
    if lims.shape != expected_shape:
        raise ValueError(f"limits must have shape {expected_shape}, got {lims.shape}")
    if not np.isfinite(lims).all():
        raise ValueError("limits must contain only finite values")
    if np.any(lims[:, 0] >= lims[:, 1]):
        raise ValueError("each limits row must satisfy min < max")

    if np.any(mins < lims[:, 0]) or np.any(maxs > lims[:, 1]):
        outside = np.any(points < lims[:, 0], axis=1) | np.any(points > lims[:, 1], axis=1)
        n_out = int(outside.sum())
        raise ValueError(
            f"{n_out} point(s) lie outside the provided limits. "
            "Widen your limits or use limits=None for auto-detection."
        )
    return lims


def normalize_periodic(periodic, dimension):
    """Return a bool ndarray of length dimension."""
    if isinstance(periodic, (bool, np.bool_)):
        return np.full(dimension, bool(periodic), dtype=bool)

    if isinstance(periodic, (str, bytes)):
        raise ValueError(
            f"periodic must be a bool or a length-{dimension} sequence of bools"
        )

    try:
        values = list(periodic)
    except TypeError as exc:
        raise ValueError(
            f"periodic must be a bool or a length-{dimension} sequence of bools"
        ) from exc

    if len(values) != dimension:
        raise ValueError(
            f"periodic must be a bool or a length-{dimension} sequence of bools, "
            f"got length {len(values)}"
        )
    if any(isinstance(v, (str, bytes)) for v in values):
        raise ValueError("periodic entries must be bool-like, not strings")
    return np.asarray([bool(v) for v in values], dtype=bool)


def normalize_threads(threads):
    """Return a positive thread count."""
    if threads == "auto":
        return os.cpu_count() or 1
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError(
            f"threads must be 'auto' or a positive integer, got {threads!r}"
        )
    return int(threads)


def normalize_n_compute(n_compute, n_total):
    """Return the number of cells to compute."""
    if n_compute is None:
        return n_total
    if isinstance(n_compute, bool) or not isinstance(n_compute, Integral):
        raise ValueError(
            f"n_compute must be a positive integer <= len(points) ({n_total}), "
            f"got {n_compute!r}"
        )
    value = int(n_compute)
    if value < 1 or value > n_total:
        raise ValueError(
            f"n_compute must be a positive integer <= len(points) ({n_total}), "
            f"got {n_compute!r}"
        )
    return value


def normalize_n_inner(n_inner, n_total):
    """Return the inner-cell split count."""
    if n_inner is None:
        return n_total
    if isinstance(n_inner, bool) or not isinstance(n_inner, Integral):
        raise ValueError(
            f"n_inner must be None or an integer in [0, {n_total}], "
            f"got {n_inner!r}"
        )
    value = int(n_inner)
    if value < 0 or value > n_total:
        raise ValueError(
            f"n_inner must be None or an integer in [0, {n_total}], "
            f"got {n_inner!r}"
        )
    return value


def normalize_representation(representation):
    """Validate the public result representation."""
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            "representation must be 'geometry', 'topology', 'per_cell', or "
            f"None, got {representation!r}"
        )
    return representation


def normalize_properties(properties, dimension):
    """Return the requested properties as a tuple of names, without repeats."""
    names = PROPERTIES[dimension]
    if isinstance(properties, str):
        properties = (properties,)
    try:
        requested = list(properties)
    except TypeError as exc:
        raise ValueError(
            "properties must be a sequence of property names"
        ) from exc
    out = []
    for name in requested:
        if name not in names:
            other = PROPERTIES[5 - dimension]
            hint = (
                f" ({name!r} is a {5 - dimension}D property)" if name in other else ""
            )
            raise ValueError(
                f"unknown {dimension}D property {name!r}{hint}; the available "
                f"properties are {', '.join(names)}"
            )
        if name not in out:
            out.append(name)
    return tuple(out)


def validate_compute_inputs(
    points,
    limits,
    *,
    dimension,
    n_compute,
    n_inner,
    threads,
    periodic,
):
    """Validate common compute arguments and return normalized values."""
    pts = validate_points(points, dimension)
    lims = resolve_limits(pts, limits, dimension)
    threads_val = normalize_threads(threads)
    n_total = pts.shape[0]
    n_compute_val = normalize_n_compute(n_compute, n_total)
    n_inner_val = normalize_n_inner(n_inner, n_total)
    periodic_val = normalize_periodic(periodic, dimension)
    return pts, lims, n_compute_val, n_inner_val, threads_val, periodic_val
