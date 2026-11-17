# distutils: language = c++
#
# Cython interface for packed parallel Voro++ results.

# cython: language_level=3

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, int64_t, uint8_t
from libc.string cimport memcpy
cimport numpy as np
import numpy as _np
import time as _time
import math as _math

np.import_array()

# Benchmarking aid: split timing for the last packed call into the C++-side
# compute+pack phase (parallel tessellation plus internal buffer flattening,
# no Python objects yet) and the memcpy phase that copies the result into
# freshly-allocated numpy arrays. Not part of the public API. Not
# thread-safe across concurrent Python-level callers (distinct from the
# internal OpenMP `nt` parallelism within a single call, which is fine);
# only meaningful for single-threaded-Python benchmarking use.
_last_packed_timing = {}


def get_last_packed_timing():
    """Timing of the most recent compute_voronoi_packed_2d/3d call: container
    creation, insertion and computation, the parallel fill of the output
    arrays (transfer), freeing the result, and disposing of the container."""
    return dict(_last_packed_timing)

cdef extern from "vpp.h":
  cdef cppclass PackedVoronoiResult:
    pass

  cdef struct CellProperties:
    double* measure
    double* boundary
    double* centroid
    double* edge_length
    double* max_radius
    double* second_moment
    double* second_moment_tensor
    int32_t* num_faces
    int32_t* num_vertices
    int32_t* num_edges

  int compute_voronoi_properties_2d(void* container_2d_, bint poly, int n_cells,
                                    int nt, const CellProperties* props) except + nogil
  int compute_voronoi_properties_3d(void* container_3d_, bint poly, int n_cells,
                                    int nt, const CellProperties* props) except + nogil

  int64_t packed_result_size(PackedVoronoiResult* result, int which) nogil
  void packed_result_fill(
      PackedVoronoiResult* result, double* vertices,
      int64_t* cell_vertex_offsets, int64_t* cell_facet_offsets,
      int32_t* facet_neighbors, int64_t* facet_vertex_offsets,
      int32_t* facet_vertices, uint8_t* cell_boundary_flags,
      int64_t* vertex_signature_offsets, int32_t* vertex_signature_ids, int nt
  ) nogil

  void* container_3d_create(double ax_, double bx_, double ay_, double by_,
    double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_,
    int init_mem_, int nt_) except +
  void put_particles_interleaved_3d(
      void* container_3d_, int n_, const double* points_, int nt_
  ) except + nogil
  void* container_poly_3d_create(double ax_, double bx_, double ay_, double by_,
    double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_,
    int init_mem_, int nt_) except +
  void put_particles_radii_3d(
      void* container_3d_, int n_, const double* points_, const double* radii_,
      int nt_
  ) except + nogil

  PackedVoronoiResult* compute_voronoi_geometry_3d_packed(
      void* container_3d_, bint poly, int n_cells, int nt, int n_inner,
      bint with_neighbors, const CellProperties* props
  ) except + nogil
  PackedVoronoiResult* compute_voronoi_topology_3d_packed(
      void* container_3d_, bint poly, int n_cells, int nt, int n_inner,
      bint with_vertices, const CellProperties* props
  ) except + nogil

  void dispose_container_3d(void* container_3d_, bint poly) nogil
  int64_t count_coincident_3d(void* container_3d_, bint poly, int nt, int* first,
                              int* second) nogil
  int64_t count_coincident_2d(void* container_2d_, bint poly, int nt, int* first,
                              int* second) nogil

  void* container_2d_create(double ax_, double bx_, double ay_, double by_,
    int nx_, int ny_, int px_, int py_, int init_mem_, int nt_) except +

  void put_particles_interleaved_2d(
      void* container_2d_, int n_, const double* points_, int nt_
  ) except + nogil
  void* container_poly_2d_create(double ax_, double bx_, double ay_, double by_,
    int nx_, int ny_, int px_, int py_, int init_mem_, int nt_) except +
  void put_particles_radii_2d(
      void* container_2d_, int n_, const double* points_, const double* radii_,
      int nt_
  ) except + nogil

  PackedVoronoiResult* compute_voronoi_geometry_2d_packed(
      void* container_2d_, bint poly, int n_cells, int nt, int n_inner,
      bint with_neighbors, const CellProperties* props
  ) except + nogil
  PackedVoronoiResult* compute_voronoi_topology_2d_packed(
      void* container_2d_, bint poly, int n_cells, int nt, int n_inner,
      bint with_vertices, const CellProperties* props
  ) except + nogil

  void dispose_container_2d(void* container_2d_, bint poly) nogil
  int per_cell_backward(
      int dimension, int64_t n_vertices, const double* points,
      const double* radii, const double* vertices, const double* grad_v,
      const int64_t* sig_offsets, const int32_t* sig_ids, const double* box,
      double* grad_p, double* grad_r, int nt, int* n_mismatched
  ) nogil
  int64_t per_cell_simplex_offsets(
      int dimension, int n_cells, const int64_t* cell_facet_offsets,
      const int64_t* facet_vertex_offsets, int64_t* simplex_offsets, int nt
  ) nogil
  void per_cell_fill_simplices(
      int dimension, int n_cells, const int64_t* cell_vertex_offsets,
      const int64_t* cell_facet_offsets, const int64_t* facet_vertex_offsets,
      const int32_t* facet_vertices, const int64_t* simplex_offsets,
      int64_t* owners, int64_t* simplices, int nt
  ) nogil
  void dispose_packed_result(PackedVoronoiResult* result) nogil
  void packed_result_profile(PackedVoronoiResult* result, double* voro_seconds,
                             double* extract_seconds) nogil

def _coincident_error(count, first, second):
  return ValueError(
      f"{count} pair(s) of generators have identical positions, for example "
      f"generators {first} and {second} (in a periodic domain, after wrapping "
      "the positions into the box). Voronoi cells are not defined for "
      "coincident generators, so remove the duplicates first."
  )


# The per-cell properties: the name used here (and in CellProperties), whether
# it is an integer, and the number of values per cell (0 for one value, 1 for
# dim values, 2 for dim x dim values).
_PROPERTY_LAYOUT = {
    "measure": (False, 0), "boundary": (False, 0), "centroid": (False, 1),
    "edge_length": (False, 0), "max_radius": (False, 0),
    "second_moment": (False, 0), "second_moment_tensor": (False, 2),
    "num_faces": (True, 0), "num_vertices": (True, 0), "num_edges": (True, 0),
}


def _allocate_properties(names, n, dim):
  """Uninitialized arrays for the requested properties; every entry is
  written by the cell loop."""
  arrays = {}
  for name in names:
    is_int, shape = _PROPERTY_LAYOUT[name]
    size = (n,) if shape == 0 else (n, dim) if shape == 1 else (n, dim, dim)
    arrays[name] = _np.empty(size, dtype=_np.int32 if is_int else _np.float64)
  return arrays


cdef double* _dptr(dict arrays, str name):
  if name not in arrays:
    return NULL
  return <double*>np.PyArray_DATA(arrays[name])


cdef int32_t* _iptr(dict arrays, str name):
  if name not in arrays:
    return NULL
  return <int32_t*>np.PyArray_DATA(arrays[name])


cdef CellProperties _cell_properties(dict arrays):
  cdef CellProperties p
  p.measure = _dptr(arrays, "measure")
  p.boundary = _dptr(arrays, "boundary")
  p.centroid = _dptr(arrays, "centroid")
  p.edge_length = _dptr(arrays, "edge_length")
  p.max_radius = _dptr(arrays, "max_radius")
  p.second_moment = _dptr(arrays, "second_moment")
  p.second_moment_tensor = _dptr(arrays, "second_moment_tensor")
  p.num_faces = _iptr(arrays, "num_faces")
  p.num_vertices = _iptr(arrays, "num_vertices")
  p.num_edges = _iptr(arrays, "num_edges")
  return p


cdef tuple _fill_packed(PackedVoronoiResult* result, int nt):
  """Allocate the output arrays (uninitialized) and fill them in parallel."""
  cdef int64_t nc = packed_result_size(result, 0)
  cdef int64_t nv = packed_result_size(result, 1)
  cdef int64_t nf = packed_result_size(result, 2)
  cdef int64_t nr = packed_result_size(result, 3)
  cdef int64_t ns = packed_result_size(result, 4)
  cdef bint hv = packed_result_size(result, 5) != 0
  cdef bint hs = packed_result_size(result, 6) != 0
  cdef bint hn = packed_result_size(result, 8) != 0
  cdef int dim = <int>packed_result_size(result, 7)
  vertices = _np.empty((nv if hv else 0) * dim, dtype=_np.float64)
  cell_vertex_offsets = _np.empty(nc + 1, dtype=_np.int64)
  cell_facet_offsets = _np.empty(nc + 1, dtype=_np.int64)
  facet_neighbors = _np.empty(nf if hn else 0, dtype=_np.int32)
  facet_vertex_offsets = _np.empty(nf + 1, dtype=_np.int64)
  facet_vertices = _np.empty(nr, dtype=_np.int32)
  cell_boundary_flags = _np.empty(nc, dtype=_np.uint8)
  vertex_signature_offsets = _np.empty(nv + 1 if hs else 0, dtype=_np.int64)
  vertex_signature_ids = _np.empty(ns if hs else 0, dtype=_np.int32)
  cdef double* p_vert = <double*>np.PyArray_DATA(vertices)
  cdef int64_t* p_cvo = <int64_t*>np.PyArray_DATA(cell_vertex_offsets)
  cdef int64_t* p_cfo = <int64_t*>np.PyArray_DATA(cell_facet_offsets)
  cdef int32_t* p_fn = <int32_t*>np.PyArray_DATA(facet_neighbors)
  cdef int64_t* p_fvo = <int64_t*>np.PyArray_DATA(facet_vertex_offsets)
  cdef int32_t* p_fv = <int32_t*>np.PyArray_DATA(facet_vertices)
  cdef uint8_t* p_flags = <uint8_t*>np.PyArray_DATA(cell_boundary_flags)
  cdef int64_t* p_so = <int64_t*>np.PyArray_DATA(vertex_signature_offsets)
  cdef int32_t* p_si = <int32_t*>np.PyArray_DATA(vertex_signature_ids)
  with nogil:
    packed_result_fill(result, p_vert, p_cvo, p_cfo, p_fn, p_fvo,
                       p_fv, p_flags, p_so, p_si, nt)
  return (vertices, cell_vertex_offsets, cell_facet_offsets,
          facet_neighbors, facet_vertex_offsets, facet_vertices,
          cell_boundary_flags, vertex_signature_offsets, vertex_signature_ids)


def compute_voronoi_packed_3d(
    points,
    limits,
    n_compute,
    n_inner,
    nt=1,
    optNumPtGrid=5.6,
    periodic=(False, False, False),
    topology=False,
    cell_vertices=False,
    init_mem=None,
    neighbors=True,
    radii=None,
    properties=("measure",),
    cells=True,
):
  """Compute packed 3D geometry or topology."""
  cdef np.ndarray[np.float64_t, ndim=2, mode="c"] points_array
  cdef int n
  cdef int val
  cdef int nt_c
  cdef int n_compute_c
  cdef int n_inner_c
  cdef bint topology_c
  cdef bint cell_vertices_c
  cdef bint neighbors_c = bool(neighbors)
  cdef double voro_seconds = 0.
  cdef double extract_seconds = 0.
  cdef int64_t n_coincident = 0
  cdef int first_coincident = -1
  cdef int second_coincident = -1
  cdef const double* points_ptr
  cdef const double* radii_ptr = NULL
  cdef np.ndarray[np.float64_t, ndim=1, mode="c"] radii_array
  cdef bint poly = radii is not None
  cdef bint cells_c = bool(cells)
  cdef CellProperties props
  cdef int missing = 0
  cdef void* container = NULL
  cdef PackedVoronoiResult* result = NULL
  cdef np.ndarray[np.float64_t, ndim=1] vertices
  cdef np.ndarray[np.int64_t, ndim=1] cell_vertex_offsets
  cdef np.ndarray[np.int64_t, ndim=1] cell_facet_offsets
  cdef np.ndarray[np.int32_t, ndim=1] facet_neighbors
  cdef np.ndarray[np.int64_t, ndim=1] facet_vertex_offsets
  cdef np.ndarray[np.int32_t, ndim=1] facet_vertices
  cdef np.ndarray[np.uint8_t, ndim=1] cell_boundary_flags
  cdef np.ndarray[np.int64_t, ndim=1] vertex_signature_offsets
  cdef np.ndarray[np.int32_t, ndim=1] vertex_signature_ids

  points_array = _np.ascontiguousarray(points, dtype=_np.float64)
  if points_array.ndim != 2 or points_array.shape[1] != 3:
      raise ValueError("points must have shape (n, 3)")
  property_arrays = _allocate_properties(properties, n_compute, 3)
  props = _cell_properties(property_arrays)
  n = points_array.shape[0]
  if n == 0:
      raise ValueError("points must not be empty")
  if poly:
      radii_array = _np.ascontiguousarray(radii, dtype=_np.float64)
      if radii_array.ndim != 1 or radii_array.shape[0] != n:
          raise ValueError("radii must have shape (n,)")
      radii_ptr = <const double*>&radii_array[0]
  nt_c = nt
  n_compute_c = n_compute
  n_inner_c = n_inner
  topology_c = bool(topology)
  cell_vertices_c = bool(cell_vertices)
  points_ptr = <const double*>&points_array[0, 0]
  periodic = [1 if value else 0 for value in periodic]
  val = int((n / optNumPtGrid) ** (1.0 / 3.0))
  blocks = [max(1, val), max(1, val), max(1, val)]
  # Initial memory per block: twice the expected number of particles, so that
  # few blocks overflow during parallel insertion.
  if init_mem is None:
    init_mem = max(8, int(_math.ceil(2.0 * n / (blocks[0] * blocks[1] * blocks[2]))))

  try:
      _tc = _time.perf_counter()
      if poly:
          container = container_poly_3d_create(
              <double>limits[0][0], <double>limits[0][1],
              <double>limits[1][0], <double>limits[1][1],
              <double>limits[2][0], <double>limits[2][1],
              <int>blocks[0], <int>blocks[1], <int>blocks[2],
              <int>periodic[0], <int>periodic[1], <int>periodic[2],
              <int>init_mem, <int>nt
          )
      else:
          container = container_3d_create(
              <double>limits[0][0], <double>limits[0][1],
              <double>limits[1][0], <double>limits[1][1],
              <double>limits[2][0], <double>limits[2][1],
              <int>blocks[0], <int>blocks[1], <int>blocks[2],
              <int>periodic[0], <int>periodic[1], <int>periodic[2],
              <int>init_mem, <int>nt
          )
      t0 = _time.perf_counter()
      _create_seconds = t0 - _tc
      with nogil:
          if poly:
              put_particles_radii_3d(
                  container, n, points_ptr, radii_ptr, nt_c
              )
          else:
              put_particles_interleaved_3d(
                  container, n, points_ptr, nt_c
              )
          n_coincident = count_coincident_3d(
              container, poly, nt_c, &first_coincident, &second_coincident
          )
      if n_coincident:
          raise _coincident_error(n_coincident, first_coincident, second_coincident)
      t_insert = _time.perf_counter()
      with nogil:
          if not cells_c:
              missing = compute_voronoi_properties_3d(
                  container, poly, n_compute_c, nt_c, &props
              )
          elif topology_c:
              result = compute_voronoi_topology_3d_packed(
                  container, poly, n_compute_c, nt_c, n_inner_c, cell_vertices_c,
                  &props
              )
          else:
              result = compute_voronoi_geometry_3d_packed(
                  container, poly, n_compute_c, nt_c, n_inner_c, neighbors_c,
                  &props
              )
      if missing:
          raise RuntimeError(
              f"{missing} of {n_compute_c} cells could not be computed by Voro++"
          )
      t1 = _time.perf_counter()
      packed = None
      if result != NULL:
          packed_result_profile(result, &voro_seconds, &extract_seconds)
          packed = _fill_packed(result, nt_c)
      t2 = _time.perf_counter()
      _last_packed_timing.clear()
      _last_packed_timing.update({
          "dimension": 3,
          "topology": bool(topology_c),
          "create_seconds": _create_seconds,
          "insert_seconds": t_insert - t0,
          "compute_seconds": t1 - t0,
          "transfer_seconds": t2 - t1,
          # Summed over threads; zero unless built with VOROPLEX_PROFILE.
          "voro_thread_seconds": voro_seconds,
          "extract_thread_seconds": extract_seconds,
      })

      if packed is not None:
          packed = (packed[0].reshape((-1, 3)),) + packed[1:]
      return packed, property_arrays
  finally:
      if result != NULL:
          _tf = _time.perf_counter()
          with nogil:
              dispose_packed_result(result)
          _last_packed_timing["free_result_seconds"] = _time.perf_counter() - _tf
      if container != NULL:
          _td = _time.perf_counter()
          with nogil:
              dispose_container_3d(container, poly)
          _last_packed_timing["dispose_seconds"] = _time.perf_counter() - _td


def compute_voronoi_packed_2d(
    points,
    limits,
    n_compute,
    n_inner,
    nt=1,
    optNumPtGrid=4.2,
    periodic=(False, False),
    topology=False,
    cell_vertices=False,
    init_mem=None,
    neighbors=True,
    radii=None,
    properties=("measure",),
    cells=True,
):
  """Compute packed 2D geometry or topology."""
  cdef np.ndarray[np.float64_t, ndim=2, mode="c"] points_array
  cdef int n
  cdef int val
  cdef int nt_c
  cdef int n_compute_c
  cdef int n_inner_c
  cdef bint topology_c
  cdef bint cell_vertices_c
  cdef bint neighbors_c = bool(neighbors)
  cdef double voro_seconds = 0.
  cdef double extract_seconds = 0.
  cdef int64_t n_coincident = 0
  cdef int first_coincident = -1
  cdef int second_coincident = -1
  cdef const double* points_ptr
  cdef const double* radii_ptr = NULL
  cdef np.ndarray[np.float64_t, ndim=1, mode="c"] radii_array
  cdef bint poly = radii is not None
  cdef bint cells_c = bool(cells)
  cdef CellProperties props
  cdef int missing = 0
  cdef void* container = NULL
  cdef PackedVoronoiResult* result = NULL
  cdef np.ndarray[np.float64_t, ndim=1] vertices
  cdef np.ndarray[np.int64_t, ndim=1] cell_vertex_offsets
  cdef np.ndarray[np.int64_t, ndim=1] cell_facet_offsets
  cdef np.ndarray[np.int32_t, ndim=1] facet_neighbors
  cdef np.ndarray[np.int64_t, ndim=1] facet_vertex_offsets
  cdef np.ndarray[np.int32_t, ndim=1] facet_vertices
  cdef np.ndarray[np.uint8_t, ndim=1] cell_boundary_flags
  cdef np.ndarray[np.int64_t, ndim=1] vertex_signature_offsets
  cdef np.ndarray[np.int32_t, ndim=1] vertex_signature_ids

  points_array = _np.ascontiguousarray(points, dtype=_np.float64)
  if points_array.ndim != 2 or points_array.shape[1] != 2:
      raise ValueError("points must have shape (n, 2)")
  property_arrays = _allocate_properties(properties, n_compute, 2)
  props = _cell_properties(property_arrays)
  n = points_array.shape[0]
  if n == 0:
      raise ValueError("points must not be empty")
  if poly:
      radii_array = _np.ascontiguousarray(radii, dtype=_np.float64)
      if radii_array.ndim != 1 or radii_array.shape[0] != n:
          raise ValueError("radii must have shape (n,)")
      radii_ptr = <const double*>&radii_array[0]
  nt_c = nt
  n_compute_c = n_compute
  n_inner_c = n_inner
  topology_c = bool(topology)
  cell_vertices_c = bool(cell_vertices)
  points_ptr = <const double*>&points_array[0, 0]
  periodic = [1 if value else 0 for value in periodic]
  val = int((n / optNumPtGrid) ** 0.5)
  blocks = [max(1, val), max(1, val)]
  # Initial memory per block: twice the expected number of particles, so that
  # few blocks overflow during parallel insertion.
  if init_mem is None:
    init_mem = max(8, int(_math.ceil(2.0 * n / (blocks[0] * blocks[1]))))

  try:
      _tc = _time.perf_counter()
      if poly:
          container = container_poly_2d_create(
              <double>limits[0][0], <double>limits[0][1],
              <double>limits[1][0], <double>limits[1][1],
              <int>blocks[0], <int>blocks[1],
              <int>periodic[0], <int>periodic[1], <int>init_mem, <int>nt
          )
      else:
          container = container_2d_create(
              <double>limits[0][0], <double>limits[0][1],
              <double>limits[1][0], <double>limits[1][1],
              <int>blocks[0], <int>blocks[1],
              <int>periodic[0], <int>periodic[1], <int>init_mem, <int>nt
          )
      t0 = _time.perf_counter()
      _create_seconds = t0 - _tc
      with nogil:
          if poly:
              put_particles_radii_2d(
                  container, n, points_ptr, radii_ptr, nt_c
              )
          else:
              put_particles_interleaved_2d(
                  container, n, points_ptr, nt_c
              )
          n_coincident = count_coincident_2d(
              container, poly, nt_c, &first_coincident, &second_coincident
          )
      if n_coincident:
          raise _coincident_error(n_coincident, first_coincident, second_coincident)
      t_insert = _time.perf_counter()
      with nogil:
          if not cells_c:
              missing = compute_voronoi_properties_2d(
                  container, poly, n_compute_c, nt_c, &props
              )
          elif topology_c:
              result = compute_voronoi_topology_2d_packed(
                  container, poly, n_compute_c, nt_c, n_inner_c, cell_vertices_c,
                  &props
              )
          else:
              result = compute_voronoi_geometry_2d_packed(
                  container, poly, n_compute_c, nt_c, n_inner_c, neighbors_c,
                  &props
              )
      if missing:
          raise RuntimeError(
              f"{missing} of {n_compute_c} cells could not be computed by Voro++"
          )
      t1 = _time.perf_counter()
      packed = None
      if result != NULL:
          packed_result_profile(result, &voro_seconds, &extract_seconds)
          packed = _fill_packed(result, nt_c)
      t2 = _time.perf_counter()
      _last_packed_timing.clear()
      _last_packed_timing.update({
          "dimension": 2,
          "topology": bool(topology_c),
          "create_seconds": _create_seconds,
          "insert_seconds": t_insert - t0,
          "compute_seconds": t1 - t0,
          "transfer_seconds": t2 - t1,
          # Summed over threads; zero unless built with VOROPLEX_PROFILE.
          "voro_thread_seconds": voro_seconds,
          "extract_thread_seconds": extract_seconds,
      })

      if packed is not None:
          packed = (packed[0].reshape((-1, 2)),) + packed[1:]
      return packed, property_arrays
  finally:
      if result != NULL:
          _tf = _time.perf_counter()
          with nogil:
              dispose_packed_result(result)
          _last_packed_timing["free_result_seconds"] = _time.perf_counter() - _tf
      if container != NULL:
          _td = _time.perf_counter()
          with nogil:
              dispose_container_2d(container, poly)
          _last_packed_timing["dispose_seconds"] = _time.perf_counter() - _td


def per_cell_backward_cpu(points, vertices, grad_vertices, sig_offsets, sig_ids,
                          box=None, nt=1, radii=None, radii_grad=False):
  """Gradient with respect to the generators from the per-cell vertex gradient.

  All arrays must be C-contiguous; points, vertices and grad_vertices are
  float64 with shape (n, d). box gives the box length along each periodic
  axis and 0 along the others (None for no periodic axes). radii gives the
  generator radii of a radical tessellation (None otherwise). Returns
  (grad_points, grad_radii, n_singular, n_mismatched), where grad_radii is
  None unless radii_grad is true.
  """
  cdef np.ndarray[np.float64_t, ndim=2, mode="c"] P = _np.ascontiguousarray(points, dtype=_np.float64)
  cdef np.ndarray[np.float64_t, ndim=2, mode="c"] V = _np.ascontiguousarray(vertices, dtype=_np.float64)
  cdef np.ndarray[np.float64_t, ndim=2, mode="c"] G = _np.ascontiguousarray(grad_vertices, dtype=_np.float64)
  cdef np.ndarray[np.int64_t, ndim=1, mode="c"] so = _np.ascontiguousarray(sig_offsets, dtype=_np.int64)
  cdef np.ndarray[np.int32_t, ndim=1, mode="c"] si = _np.ascontiguousarray(sig_ids, dtype=_np.int32)
  cdef np.ndarray[np.float64_t, ndim=2, mode="c"] out = _np.zeros_like(P)
  cdef int dim = P.shape[1]
  cdef np.ndarray[np.float64_t, ndim=1, mode="c"] B = (
      _np.zeros(dim) if box is None else _np.ascontiguousarray(box, dtype=_np.float64))
  cdef int64_t nv = V.shape[0]
  cdef int nt_c = nt
  cdef int bad
  cdef int mismatched = 0
  cdef np.ndarray[np.float64_t, ndim=1, mode="c"] Rad
  cdef np.ndarray[np.float64_t, ndim=1, mode="c"] grad_r
  cdef const double* rad_ptr = NULL
  cdef double* grad_r_ptr = NULL
  if radii is not None:
    Rad = _np.ascontiguousarray(radii, dtype=_np.float64)
    if Rad.shape[0] != P.shape[0]:
      raise ValueError("inconsistent array shapes")
    rad_ptr = &Rad[0]
    if radii_grad:
      grad_r = _np.zeros(P.shape[0])
      grad_r_ptr = &grad_r[0]
  elif radii_grad:
    raise ValueError("radii_grad requires radii")
  if V.shape[1] != dim or G.shape[0] != nv or G.shape[1] != dim or so.shape[0] != nv + 1 or B.shape[0] != dim:
    raise ValueError("inconsistent array shapes")
  if nv == 0:
    return out, (grad_r if radii_grad else None), 0, 0
  with nogil:
    bad = per_cell_backward(dim, nv, &P[0, 0], rad_ptr, &V[0, 0], &G[0, 0],
                            <const int64_t*>&so[0], <const int32_t*>&si[0],
                            &B[0], &out[0, 0], grad_r_ptr, nt_c, &mismatched)
  return out, (grad_r if radii_grad else None), bad, mismatched


def per_cell_fan_simplices_cpu(dimension, cell_vertex_offsets, cell_facet_offsets,
                               facet_vertex_offsets, facet_vertices, nt=1):
  """Fan triangles (2D) or tetrahedra (3D) for per-cell vertices.

  Returns (owners, simplices): int64 arrays of shape (S,) and (S, dimension),
  where the simplex vertex indices refer to the packed per-cell vertices.
  """
  cdef np.ndarray[np.int64_t, ndim=1, mode="c"] cvo = _np.ascontiguousarray(cell_vertex_offsets, dtype=_np.int64)
  cdef np.ndarray[np.int64_t, ndim=1, mode="c"] cfo = _np.ascontiguousarray(cell_facet_offsets, dtype=_np.int64)
  cdef np.ndarray[np.int64_t, ndim=1, mode="c"] fvo = _np.ascontiguousarray(facet_vertex_offsets, dtype=_np.int64)
  cdef np.ndarray[np.int32_t, ndim=1, mode="c"] fv = _np.ascontiguousarray(facet_vertices, dtype=_np.int32)
  cdef int dim = dimension
  cdef int n_cells = cfo.shape[0] - 1
  cdef int nt_c = nt
  cdef int64_t total
  if dim not in (2, 3):
    raise ValueError("dimension must be 2 or 3")
  if cvo.shape[0] != n_cells + 1:
    raise ValueError("inconsistent offsets")
  cdef np.ndarray[np.int64_t, ndim=1, mode="c"] so = _np.empty(n_cells + 1, dtype=_np.int64)
  if fv.shape[0] == 0:
    return _np.empty(0, dtype=_np.int64), _np.empty((0, dim), dtype=_np.int64)
  with nogil:
    total = per_cell_simplex_offsets(dim, n_cells, <const int64_t*>&cfo[0],
                                     <const int64_t*>&fvo[0], <int64_t*>&so[0], nt_c)
  # Outputs are filled in parallel by the C++ code, so they are not zeroed.
  cdef np.ndarray[np.int64_t, ndim=1, mode="c"] owners = _np.empty(total, dtype=_np.int64)
  cdef np.ndarray[np.int64_t, ndim=2, mode="c"] simplices = _np.empty((total, dim), dtype=_np.int64)
  if total == 0:
    return owners, simplices
  with nogil:
    per_cell_fill_simplices(dim, n_cells, <const int64_t*>&cvo[0], <const int64_t*>&cfo[0],
                            <const int64_t*>&fvo[0], <const int32_t*>&fv[0],
                            <const int64_t*>&so[0], <int64_t*>&owners[0],
                            <int64_t*>&simplices[0, 0], nt_c)
  return owners, simplices
