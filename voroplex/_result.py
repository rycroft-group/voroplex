"""Packed result objects returned by VoroPlex."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as integer_index
from types import MappingProxyType

import numpy as np


INTERFACE_FLAG = np.uint8(1)
BOUNDARY_FLAG = np.uint8(2)


def _readonly_array(value, dtype, name, ndim=1):
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError("{} must be a {}D array".format(name, ndim))
    array.setflags(write=False)
    return array


def _validate_offsets(offsets, final_size, name, full=True):
    if offsets.size == 0 or offsets[0] != 0:
        raise ValueError("{} must start at zero".format(name))
    # The monotonicity check reads the whole array, so it is skipped for
    # results built by VoroPlex itself, which are correct by construction.
    if full and np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("{} must be nondecreasing".format(name))
    if offsets[-1] != final_size:
        raise ValueError(
            "{} must end at {}, got {}".format(name, final_size, offsets[-1])
        )


# The per-cell properties: the name used in ``properties=`` and the plural name
# of the attribute that returns the array.
PROPERTY_ATTRIBUTES = {
    "volume": "volumes",
    "area": "areas",
    "surface_area": "surface_areas",
    "perimeter": "perimeters",
    "centroid": "centroids",
    "n_faces": "n_faces",
    "n_vertices": "n_vertices",
    "n_edges": "n_edges",
    "total_edge_length": "total_edge_lengths",
    "max_radius": "max_radii",
    "second_moment": "second_moments",
    "second_moment_tensor": "second_moment_tensors",
}
_PROPERTY_DIMENSIONS = {
    "volume": (3,), "area": (2,), "surface_area": (3,), "perimeter": (2,),
    "n_faces": (3,), "n_edges": (3,), "total_edge_length": (3,),
}
_INTEGER_PROPERTIES = ("n_faces", "n_vertices", "n_edges")


class VoronoiResult:
    """The result of a Voronoi tessellation.

    It holds the requested per-cell properties (by default, the area or
    volume of each cell), and, unless it was computed with
    ``representation=None``, the cells themselves. Geometry stores cell-local
    coordinates. Topology stores ragged integer vertex signatures. Both share
    the same packed cell/facet connectivity.
    """

    def __init__(
        self,
        *,
        dimension,
        representation,
        limits,
        periodic,
        n_points,
        n_inner,
        threads,
        n_cells=None,
        properties=None,
        cell_vertex_offsets=None,
        cell_facet_offsets=None,
        facet_neighbors=None,
        facet_vertex_offsets=None,
        facet_vertices=None,
        cell_surface_flags=None,
        vertices=None,
        vertex_signature_offsets=None,
        vertex_signature_ids=None,
        has_neighbors=True,
        radii=None,
        measures=None,
        _trusted=False,
    ):
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if representation not in ("geometry", "topology", "per_cell", None):
            raise ValueError(
                "representation must be 'geometry', 'topology', 'per_cell', "
                "or None"
            )

        self.dimension = dimension
        self.representation = representation
        self.limits = _readonly_array(limits, np.float64, "limits", ndim=2)
        if self.limits.shape != (dimension, 2):
            raise ValueError(
                "limits must have shape ({}, 2)".format(dimension)
            )
        if not np.isfinite(self.limits).all():
            raise ValueError("limits must contain finite values")
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("each limits row must satisfy min < max")
        if isinstance(periodic, (bool, np.bool_)):
            periodic = np.full(dimension, bool(periodic), dtype=np.bool_)
        self.periodic = _readonly_array(periodic, np.bool_, "periodic")
        if self.periodic.size != dimension:
            raise ValueError(
                "periodic must have {} entries".format(dimension)
            )
        if representation == "topology" and self.periodic.any():
            raise ValueError("topology results cannot be periodic")
        self.has_neighbors = bool(has_neighbors) and representation is not None
        if not self.has_neighbors and representation not in ("geometry", None):
            raise ValueError(
                "only geometry results can be computed without neighbors"
            )
        self.n_points = int(n_points)
        self.n_inner = int(n_inner)
        self.threads = int(threads)
        if self.n_points < 1:
            raise ValueError("n_points must be positive")
        if self.n_inner < 0 or self.n_inner > self.n_points:
            raise ValueError("n_inner must be in [0, n_points]")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if radii is None:
            self._radii = None
        else:
            self._radii = _readonly_array(radii, np.float64, "radii")
            if self._radii.size != self.n_points:
                raise ValueError("radii must have n_points entries")

        # The properties. For compatibility, measures can also be given
        # separately, as the area (2D) or volume (3D).
        properties = dict(properties or {})
        if measures is not None:
            properties.setdefault("volume" if dimension == 3 else "area", measures)
        if n_cells is None:
            if cell_facet_offsets is not None:
                n_cells = np.asarray(cell_facet_offsets).size - 1
            elif properties:
                n_cells = len(next(iter(properties.values())))
            else:
                raise ValueError("n_cells is required")
        n_cells = int(n_cells)
        if n_cells < 1:
            raise ValueError("results must contain at least one cell")
        if n_cells > self.n_points:
            raise ValueError("n_cells cannot exceed n_points")
        self._n_cells = n_cells
        self._properties = {}
        for name, value in properties.items():
            if (name not in PROPERTY_ATTRIBUTES
                    or dimension not in _PROPERTY_DIMENSIONS.get(name, (2, 3))):
                raise ValueError(
                    "unknown {}D property {!r}".format(dimension, name)
                )
            dtype = np.int32 if name in _INTEGER_PROPERTIES else np.float64
            shape = {"centroid": (n_cells, dimension),
                     "second_moment_tensor": (n_cells, dimension, dimension)
                     }.get(name, (n_cells,))
            array = _readonly_array(value, dtype, name, ndim=len(shape))
            if array.shape != shape:
                raise ValueError(
                    "{} must have shape {}, got {}".format(name, shape, array.shape)
                )
            self._properties[name] = array

        if representation is None:
            self._vertices = None
            self._cell_vertex_offsets = None
            self._cell_facet_offsets = None
            self._facet_neighbors = None
            self._facet_vertex_offsets = None
            self._facet_vertices = None
            self._cell_boundary_flags = None
            self._vertex_signature_offsets = None
            self._vertex_signature_ids = None
            return

        if not self.has_vertices:
            self._vertices = None
        else:
            if vertices is None:
                raise ValueError(
                    "vertices are required for {} results".format(representation)
                )
            self._vertices = _readonly_array(
                vertices, np.float64, "vertices", ndim=2
            )
            if self._vertices.shape[1] != dimension:
                raise ValueError(
                    "vertices must have shape (n, {})".format(dimension)
                )

        self._cell_vertex_offsets = _readonly_array(
            cell_vertex_offsets, np.int64, "cell_vertex_offsets"
        )
        self._cell_facet_offsets = _readonly_array(
            cell_facet_offsets, np.int64, "cell_facet_offsets"
        )
        self._facet_neighbors = (
            _readonly_array(facet_neighbors, np.int32, "facet_neighbors")
            if self.has_neighbors else None
        )
        self._facet_vertex_offsets = _readonly_array(
            facet_vertex_offsets, np.int64, "facet_vertex_offsets"
        )
        self._facet_vertices = _readonly_array(
            facet_vertices, np.int32, "facet_vertices"
        )
        self._cell_boundary_flags = _readonly_array(
            cell_surface_flags, np.uint8, "cell_surface_flags"
        )
        if self.has_signatures:
            self._vertex_signature_offsets = _readonly_array(
                vertex_signature_offsets,
                np.int64,
                "vertex_signature_offsets",
            )
            self._vertex_signature_ids = _readonly_array(
                vertex_signature_ids, np.int32, "vertex_signature_ids"
            )
        else:
            self._vertex_signature_offsets = None
            self._vertex_signature_ids = None

        if self._cell_vertex_offsets.size != n_cells + 1:
            raise ValueError("cell_vertex_offsets must have n_cells + 1 entries")
        if self._cell_facet_offsets.size != n_cells + 1:
            raise ValueError("cell_facet_offsets must have n_cells + 1 entries")
        if self._cell_boundary_flags.size != n_cells:
            raise ValueError("cell_surface_flags must have n_cells entries")

        _validate_offsets(
            self._cell_vertex_offsets,
            (
                self._vertices.shape[0]
                if self.has_vertices
                else self._vertex_signature_offsets.size - 1
            ),
            "cell_vertex_offsets",
            full=not _trusted,
        )
        n_facets = self._facet_vertex_offsets.size - 1
        if n_facets < 0:
            raise ValueError("facet_vertex_offsets must not be empty")
        if self.has_neighbors and self._facet_neighbors.size != n_facets:
            raise ValueError("facet_neighbors must have n_facets entries")
        _validate_offsets(
            self._cell_facet_offsets,
            n_facets,
            "cell_facet_offsets",
            full=not _trusted,
        )
        _validate_offsets(
            self._facet_vertex_offsets,
            self._facet_vertices.size,
            "facet_vertex_offsets",
            full=not _trusted,
        )
        if self.has_signatures:
            n_local_vertices = self._cell_vertex_offsets[-1]
            if self._vertex_signature_offsets.size != n_local_vertices + 1:
                raise ValueError(
                    "vertex_signature_offsets must have "
                    "n_local_vertices + 1 entries"
                )
            _validate_offsets(
                self._vertex_signature_offsets,
                self._vertex_signature_ids.size,
                "vertex_signature_offsets",
                full=not _trusted,
            )

        # Native packing validates local facet references.

    def __getstate__(self):
        """Exclude lazily-populated derived caches from pickling.

        Cache attributes (vertex/facet dedup maps, reconstruction plans,
        static device tensors -- all named with a `_cache` suffix by
        convention) may hold CUDA tensors and are cheap to rebuild on
        first use after unpickling. Including them would needlessly
        bloat the pickle and could make it fail to unpickle on a
        machine without a GPU, even though none of the actual result
        data is device-specific. This keeps plain `pickle`/`copy.deepcopy`
        consistent with the pickle-free design of `export_npz`/`load_npz`.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.endswith("_cache")
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def properties(self):
        """The computed per-cell properties, as a read-only mapping from
        property names, such as ``"volume"``, to arrays."""
        return MappingProxyType(self._properties)

    def _property(self, name):
        dims = _PROPERTY_DIMENSIONS.get(name, (2, 3))
        if self.dimension not in dims:
            other = {"volume": "area", "area": "volume",
                     "surface_area": "perimeter", "perimeter": "surface_area"}
            hint = (
                "; {}D results have {}".format(
                    self.dimension, PROPERTY_ATTRIBUTES[other[name]])
                if name in other else ""
            )
            raise AttributeError(
                "{} are only defined for {}D results{}".format(
                    PROPERTY_ATTRIBUTES[name], dims[0], hint)
            )
        if name not in self._properties:
            raise AttributeError(
                "the {} were not computed; pass properties=(..., {!r}) to "
                "compute them".format(PROPERTY_ATTRIBUTES[name], name)
            )
        return self._properties[name]

    @property
    def measures(self):
        """The area (2D) or volume (3D) of each cell."""
        return self._property("volume" if self.dimension == 3 else "area")

    areas = property(lambda self: self._property("area"),
                     doc="The area of each cell (2D).")
    volumes = property(lambda self: self._property("volume"),
                       doc="The volume of each cell (3D).")
    perimeters = property(lambda self: self._property("perimeter"),
                          doc="The perimeter of each cell (2D).")
    surface_areas = property(lambda self: self._property("surface_area"),
                             doc="The surface area of each cell (3D).")
    centroids = property(lambda self: self._property("centroid"),
                         doc="The centroid of each cell, shape (N, d).")
    n_faces = property(lambda self: self._property("n_faces"),
                         doc="The number of faces of each cell (3D).")
    n_vertices = property(lambda self: self._property("n_vertices"),
                            doc="The number of vertices of each cell.")
    n_edges = property(lambda self: self._property("n_edges"),
                         doc="The number of edges of each cell (3D).")
    total_edge_lengths = property(
        lambda self: self._property("total_edge_length"),
        doc="The total length of the edges of each cell (3D).")
    max_radii = property(
        lambda self: self._property("max_radius"),
        doc="The largest distance from each generator to a vertex of its "
            "cell.")
    second_moments = property(
        lambda self: self._property("second_moment"),
        doc="The integral of |x - p_i|^2 over each cell.")
    second_moment_tensors = property(
        lambda self: self._property("second_moment_tensor"),
        doc="The integral of (x - p_i)(x - p_i)^T over each cell, shape "
            "(N, d, d).")

    @property
    def radii(self):
        """The generator radii of a radical tessellation, or None."""
        return self._radii

    @property
    def is_radical(self):
        """True for a radical (power) tessellation, computed with radii."""
        return self._radii is not None

    @property
    def has_cells(self):
        """True if the result stores the cells, which is the case unless it
        was computed with representation=None."""
        return self.representation is not None

    @property
    def has_vertices(self):
        """True if the result stores cell-local vertex coordinates."""
        return self.representation in ("geometry", "per_cell")

    @property
    def has_signatures(self):
        """True if the result stores cell-local vertex signatures."""
        return self.representation in ("topology", "per_cell")

    @property
    def vertices(self):
        """The vertex coordinates of every cell; geometry and per-cell results only."""
        self._require_cells()
        if not self.has_vertices:
            raise AttributeError(
                "topology results store vertex signatures, not coordinates"
            )
        return self._vertices

    @property
    def cell_vertex_offsets(self):
        """Where each cell's vertices start: cell i has rows offsets[i] to offsets[i+1] - 1."""
        self._require_cells()
        return self._cell_vertex_offsets

    @property
    def cell_facet_offsets(self):
        """Where each cell's facets (faces in 3D, edges in 2D) start."""
        self._require_cells()
        return self._cell_facet_offsets

    @property
    def facet_neighbors(self):
        """The neighbor across each facet: a cell ID, or a negative boundary code."""
        self._require_neighbors()
        return self._facet_neighbors

    @property
    def facet_vertex_offsets(self):
        """Where each facet's vertex IDs start in facet_vertices."""
        self._require_cells()
        return self._facet_vertex_offsets

    @property
    def facet_vertices(self):
        """The vertices of each facet, as indices within its own cell."""
        self._require_cells()
        return self._facet_vertices

    @property
    def vertex_signature_offsets(self):
        """Where each vertex's signature starts; per-cell and topology only."""
        self._require_topology()
        return self._vertex_signature_offsets

    @property
    def vertex_signature_ids(self):
        """The generator IDs and boundary codes that define each vertex, sorted."""
        self._require_topology()
        return self._vertex_signature_ids

    @property
    def cell_surface_flags(self):
        """For each cell, bit flags for touching an outer cell or the domain boundary."""
        self._require_neighbors()
        return self._cell_boundary_flags

    @property
    def surface_cells(self):
        """The IDs of the cells on the surface: touching an outer cell or the boundary."""
        self._require_neighbors()
        return self._readonly_cell_ids(self._cell_boundary_flags != 0)

    @property
    def interface_cells(self):
        """The IDs of the inner cells that touch an outer cell."""
        self._require_neighbors()
        mask = (self._cell_boundary_flags & INTERFACE_FLAG) != 0
        return self._readonly_cell_ids(mask)

    @property
    def boundary_cells(self):
        """The IDs of the cells that touch the boundary of the domain."""
        self._require_neighbors()
        mask = (self._cell_boundary_flags & BOUNDARY_FLAG) != 0
        return self._readonly_cell_ids(mask)

    @staticmethod
    def _readonly_cell_ids(mask):
        ids = np.flatnonzero(mask).astype(np.int32)
        ids.setflags(write=False)
        return ids

    @property
    def cells(self):
        """All of the cells, as a sequence of VoronoiCell views."""
        return _CellSequence(self)

    def cell(self, index):
        """The cell with the given index, as a VoronoiCell view."""
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("cell index out of range")
        return VoronoiCell(self, index)

    def _require_cells(self):
        if not self.has_cells:
            raise AttributeError(
                "this result was computed with representation=None, so it "
                "only has the requested properties, and no information about "
                "the cells themselves"
            )

    def _require_geometry(self):
        self._require_cells()
        if not self.has_vertices:
            raise AttributeError(
                "cell-local geometry is only available for "
                "representation='geometry' or 'per_cell'"
            )

    def _require_neighbors(self):
        self._require_cells()
        if not self.has_neighbors:
            raise AttributeError(
                "this result was computed with neighbors=False, so it has no "
                "neighbor information or boundary flags; compute it with "
                "neighbors=True"
            )

    def _require_topology(self):
        self._require_cells()
        if not self.has_signatures:
            raise AttributeError(
                "vertex signatures are only available for "
                "representation='topology' or 'per_cell'"
            )

    def __len__(self):
        return self._n_cells

    @property
    def n_cells(self):
        """The number of computed cells."""
        return self._n_cells

    def __repr__(self):
        periodic = (
            "" if not self.periodic.any()
            else ", periodic={}".format(self.periodic.tolist())
        )
        neighbors = (
            "" if self.has_neighbors or not self.has_cells
            else ", neighbors=False"
        )
        radical = ", radical=True" if self.is_radical else ""
        return (
            "VoronoiResult(dimension={}, representation={!r}, n_cells={}, "
            "n_points={}{}{}{}, properties={})".format(
                self.dimension, self.representation, self.n_cells,
                self.n_points, periodic, neighbors, radical,
                tuple(self._properties),
            )
        )


class VoronoiCell:
    """A zero-copy view of one packed cell."""

    __slots__ = ("_result", "index")

    def __init__(self, result, index):
        self._result = result
        self.index = index

    @property
    def id(self):
        """The index of the cell, which is the index of its generator."""
        return self.index

    @property
    def measure(self):
        """The area (2D) or volume (3D) of the cell."""
        return self._result.measures[self.index]

    @property
    def area(self):
        """The area of the cell (2D only)."""
        if self._result.dimension != 2:
            raise AttributeError("3D cells have volume, not area")
        return self.measure

    @property
    def volume(self):
        """The volume of the cell (3D only)."""
        if self._result.dimension != 3:
            raise AttributeError("2D cells have area, not volume")
        return self.measure

    @property
    def vertices(self):
        """The coordinates of the cell's vertices; geometry and per-cell only."""
        self._result._require_geometry()
        start = self._result.cell_vertex_offsets[self.index]
        stop = self._result.cell_vertex_offsets[self.index + 1]
        return self._result.vertices[start:stop]

    @property
    def vertex_signatures(self):
        """The signature of each vertex; per-cell and topology only."""
        self._result._require_topology()
        start = self._result.cell_vertex_offsets[self.index]
        stop = self._result.cell_vertex_offsets[self.index + 1]
        return _VertexSignatureSequence(self._result, start, stop)

    @property
    def n_vertices(self):
        """The number of vertices of the cell."""
        return int(
            self._result.cell_vertex_offsets[self.index + 1]
            - self._result.cell_vertex_offsets[self.index]
        )

    @property
    def neighbors(self):
        """The neighbor across each facet: a cell ID, or a negative boundary code."""
        start = self._result.cell_facet_offsets[self.index]
        stop = self._result.cell_facet_offsets[self.index + 1]
        return self._result.facet_neighbors[start:stop]

    @property
    def facets(self):
        """The faces (3D) or edges (2D) of the cell, as VoronoiFacet views."""
        start = self._result.cell_facet_offsets[self.index]
        stop = self._result.cell_facet_offsets[self.index + 1]
        return _FacetSequence(self._result, start, stop)

    @property
    def edges(self):
        """The edges of the cell (2D only), as VoronoiFacet views."""
        if self._result.dimension != 2:
            raise AttributeError("3D cells have faces, not edges")
        return self.facets

    @property
    def faces(self):
        """The faces of the cell (3D only), as VoronoiFacet views."""
        if self._result.dimension != 3:
            raise AttributeError("2D cells have edges, not faces")
        return self.facets

    @property
    def surface_flags(self):
        """Bit flags for whether the cell touches an outer cell or the domain boundary."""
        return self._result.cell_surface_flags[self.index]


class VoronoiFacet:
    """A zero-copy view of one edge (2D) or face (3D)."""

    __slots__ = ("_result", "index")

    def __init__(self, result, index):
        self._result = result
        self.index = index

    @property
    def neighbor(self):
        """The cell on the other side of the facet, or a negative boundary code."""
        return self._result.facet_neighbors[self.index]

    @property
    def vertex_ids(self):
        """The facet's vertices, as indices within its cell."""
        start = self._result.facet_vertex_offsets[self.index]
        stop = self._result.facet_vertex_offsets[self.index + 1]
        return self._result.facet_vertices[start:stop]


class _CellSequence(Sequence):
    __slots__ = ("_result",)

    def __init__(self, result):
        self._result = result

    def __len__(self):
        return len(self._result)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._result.cell(i) for i in range(*index.indices(len(self)))]
        return self._result.cell(index)


class _FacetSequence(Sequence):
    __slots__ = ("_result", "_start", "_stop")

    def __init__(self, result, start, stop):
        self._result = result
        self._start = int(start)
        self._stop = int(stop)

    def __len__(self):
        return self._stop - self._start

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                VoronoiFacet(self._result, i)
                for i in range(self._start, self._stop)[index]
            ]
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("facet index out of range")
        return VoronoiFacet(self._result, self._start + index)


class _VertexSignatureSequence(Sequence):
    __slots__ = ("_result", "_start", "_stop")

    def __init__(self, result, start, stop):
        self._result = result
        self._start = int(start)
        self._stop = int(stop)

    def __len__(self):
        return self._stop - self._start

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("vertex signature index out of range")
        global_index = self._start + index
        start = self._result.vertex_signature_offsets[global_index]
        stop = self._result.vertex_signature_offsets[global_index + 1]
        return self._result.vertex_signature_ids[start:stop]
