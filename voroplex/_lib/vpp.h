// C++ interface for parallel 2D and 3D Voronoi tessellations.

#ifndef VOROPLEX_VPP_H
#define VOROPLEX_VPP_H

#include <vector>
#include <stdint.h>

// The result of a tessellation, held internally until it is copied into
// caller-provided arrays by packed_result_fill. The arrays use CSR-style
// offsets; facet_vertices contains local vertex IDs within each owning cell.
// Geometry results have vertices; topology results have vertex signatures;
// topology results with with_vertices=true (the per-cell representation)
// have both. In a radical tessellation, a cell can be empty: it then has
// measure 0, and no vertices or facets.
struct PackedVoronoiResult;

// Arrays for the requested per-cell properties, indexed by particle ID, which
// the cell loops fill directly. A NULL pointer means that the property was not
// requested. centroid has dim entries per cell, and second_moment_tensor has
// dim*dim. For an empty cell in a radical tessellation, the sizes and counts
// are zero, and the centroid and maximum radius are NaN.
struct CellProperties {
    double* measure;              // area (2D) or volume (3D)
    double* boundary;             // perimeter (2D) or surface area (3D)
    double* centroid;
    double* edge_length;          // total edge length (3D)
    double* max_radius;           // largest distance from the particle to a vertex
    double* second_moment;        // integral of |x-p|^2 over the cell
    double* second_moment_tensor; // integral of (x-p)(x-p)^T over the cell
    int32_t* num_faces;           // (3D)
    int32_t* num_vertices;
    int32_t* num_edges;           // (3D)
};

void* container_3d_create(double ax_, double bx_, double ay_, double by_,
  double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_,
  int init_mem_, int nt_);

void put_particles_interleaved_3d(
    void* container_3d_, int n_, const double* points_, int nt_
);

// The containers for radical (power) tessellations, which store a radius for
// each particle. Functions below that take a container also take poly, which
// is true for these containers.
void* container_poly_3d_create(double ax_, double bx_, double ay_, double by_,
  double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_,
  int init_mem_, int nt_);

void put_particles_radii_3d(
    void* container_3d_, int n_, const double* points_, const double* radii_,
    int nt_
);

PackedVoronoiResult* compute_voronoi_geometry_3d_packed(
    void* container_3d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_neighbors,
    const CellProperties* props
);

PackedVoronoiResult* compute_voronoi_topology_3d_packed(
    void* container_3d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_vertices,
    const CellProperties* props
);

// Computes only the requested properties of each cell, with no cell geometry
// (representation=None). Returns the number of cells that could not be
// computed.
int compute_voronoi_properties_3d(void* container_3d_, bool poly, int n_cells,
                                  int nt, const CellProperties* props);

// The number of pairs of particles with identical positions; the IDs of one
// such pair are returned in (first, second), or -1 if there are none.
int64_t count_coincident_3d(void* container_3d_, bool poly, int nt, int* first,
                            int* second);

void dispose_container_3d(void* container_3d_, bool poly);

void* container_2d_create(double ax_, double bx_, double ay_, double by_,
  int nx_, int ny_, int px_, int py_, int init_mem_, int nt_);

void put_particles_interleaved_2d(
    void* container_2d_, int n_, const double* points_, int nt_
);

void* container_poly_2d_create(double ax_, double bx_, double ay_, double by_,
  int nx_, int ny_, int px_, int py_, int init_mem_, int nt_);

void put_particles_radii_2d(
    void* container_2d_, int n_, const double* points_, const double* radii_,
    int nt_
);

PackedVoronoiResult* compute_voronoi_geometry_2d_packed(
    void* container_2d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_neighbors,
    const CellProperties* props
);

PackedVoronoiResult* compute_voronoi_topology_2d_packed(
    void* container_2d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_vertices,
    const CellProperties* props
);

// Computes only the requested properties of each cell, with no cell geometry
// (representation=None). Returns the number of cells that could not be
// computed.
int compute_voronoi_properties_2d(void* container_2d_, bool poly, int n_cells,
                                  int nt, const CellProperties* props);

int64_t count_coincident_2d(void* container_2d_, bool poly, int nt, int* first,
                            int* second);

void dispose_container_2d(void* container_2d_, bool poly);
void dispose_packed_result(PackedVoronoiResult* result);

// The time spent in Voro++'s compute_cell and in the rest of the cell loop,
// summed over threads. Both are zero unless compiled with VOROPLEX_PROFILE.
void packed_result_profile(PackedVoronoiResult* result, double* voro_seconds,
                           double* extract_seconds);

int per_cell_backward(
    int dimension, int64_t n_vertices, const double* points,
    const double* radii, const double* vertices, const double* grad_v,
    const int64_t* sig_offsets, const int32_t* sig_ids, const double* box,
    double* grad_p, double* grad_r, int nt, int* n_mismatched
);

int64_t per_cell_simplex_offsets(
    int dimension, int n_cells, const int64_t* cell_facet_offsets,
    const int64_t* facet_vertex_offsets, int64_t* simplex_offsets, int nt
);

void per_cell_fill_simplices(
    int dimension, int n_cells, const int64_t* cell_vertex_offsets,
    const int64_t* cell_facet_offsets, const int64_t* facet_vertex_offsets,
    const int32_t* facet_vertices, const int64_t* simplex_offsets,
    int64_t* owners, int64_t* simplices, int nt
);

// which: 0 n_cells, 1 total vertices, 2 total facets, 3 total facet vertex
// references, 4 total signature IDs, 5 has vertices, 6 has signatures,
// otherwise the dimension.
int64_t packed_result_size(PackedVoronoiResult* result, int which);

void packed_result_fill(
    PackedVoronoiResult* result, double* vertices,
    int64_t* cell_vertex_offsets, int64_t* cell_facet_offsets,
    int32_t* facet_neighbors, int64_t* facet_vertex_offsets,
    int32_t* facet_vertices, uint8_t* cell_boundary_flags,
    int64_t* vertex_signature_offsets, int32_t* vertex_signature_ids, int nt
);

#endif
