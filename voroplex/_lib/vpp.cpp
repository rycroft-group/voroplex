// Packed parallel Voro++ implementation.

#include "vpp.h"
#include "voro++.hh"
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <omp.h>
using namespace voro;
using namespace std;

// With VOROPLEX_PROFILE defined, each thread measures the time spent in
// Voro++'s compute_cell and in the rest of the loop body. This adds two clock
// reads per cell, so it is only used in a separate build for benchmarks.
#ifdef VOROPLEX_PROFILE
#define PROFILE_START double t_prof0_ = omp_get_wtime(), t_prof1_ = t_prof0_;
#define PROFILE_VORO_DONE t_prof1_ = omp_get_wtime(); \
    result->voro_seconds[tid] += t_prof1_ - t_prof0_;
#define PROFILE_CELL_DONE result->extract_seconds[tid] += omp_get_wtime() - t_prof1_;
#else
#define PROFILE_START
#define PROFILE_VORO_DONE
#define PROFILE_CELL_DONE
#endif

namespace {

const uint8_t INTERFACE_BOUNDARY_FLAG = 1;
const uint8_t WALL_BOUNDARY_FLAG = 2;

}  // namespace

// Scratch space for one cell. Each thread reuses a single CellScratch for all
// of the cells that it computes, so no memory is allocated per cell.
struct CellScratch {
    int64_t n_vertices;
    vector<double> vertices;
    vector<int32_t> neighbors;
    vector<int64_t> facet_offsets;
    vector<int32_t> facet_vertices;
    vector<int64_t> signature_offsets;
    vector<int32_t> signature_ids;

    CellScratch() : n_vertices(0) {}
};

// Where one cell's data is stored, in the buffers of the thread that computed
// it. Plain data, so arrays of these are allocated without initialization.
struct CellRecord {
    int32_t thread;
    int32_t n_vertices;
    int32_t n_facets;
    int32_t n_refs;
    int32_t n_sig_ids;
    int64_t vertex_start;
    int64_t facet_start;
    int64_t ref_start;
    int64_t sig_start;
};

// The data of all of the cells computed by one thread, stored contiguously.
struct ThreadStore {
    int64_t n_vertices;
    vector<double> vertices;
    vector<int32_t> neighbors;
    vector<int32_t> facet_sizes;
    vector<int32_t> facet_vertices;
    vector<int32_t> sig_sizes;
    vector<int32_t> sig_ids;

    ThreadStore() : n_vertices(0) {}
};

// The result of a tessellation, held in per-thread buffers until it is copied
// into arrays provided by the caller (packed_result_fill).
struct PackedVoronoiResult {
    int dimension;
    int n_cells;
    bool has_vertices;
    bool has_signatures;
    bool has_neighbors;
    uint8_t* flags;
    CellRecord* records;
    vector<ThreadStore> stores;
    // Exclusive prefix sums over cells, and the totals.
    int64_t* vertex_offsets;
    int64_t* facet_offsets;
    int64_t* ref_offsets;
    int64_t* sig_offsets;
    int64_t total_vertices, total_facets, total_refs, total_sig_ids;
    // Per-thread time in Voro++'s compute_cell and in the rest of the loop
    // body. Only measured when compiled with VOROPLEX_PROFILE.
    vector<double> voro_seconds, extract_seconds;

    PackedVoronoiResult(int n_cells_, int dimension_, bool has_vertices_,
                        bool has_signatures_, int nt,
                        bool has_neighbors_ = true)
        : dimension(dimension_), n_cells(n_cells_),
          has_vertices(has_vertices_), has_signatures(has_signatures_),
          has_neighbors(has_neighbors_),
          flags(new uint8_t[n_cells_]),
          records(new CellRecord[n_cells_]), stores(nt < 1 ? 1 : nt),
          vertex_offsets(NULL), facet_offsets(NULL), ref_offsets(NULL),
          sig_offsets(NULL), total_vertices(0), total_facets(0),
          total_refs(0), total_sig_ids(0),
          voro_seconds(nt < 1 ? 1 : nt, 0.), extract_seconds(nt < 1 ? 1 : nt, 0.) {}

    // Reserve this thread's buffers for its expected share of the cells, so
    // that they rarely need to grow. Growing a buffer copies it and touches
    // every page twice, which is slow for large tessellations. The sizes are
    // a little above the averages for random points; pages that are reserved
    // but never written are not given physical memory.
    void reserve(int tid, int nt) {
        ThreadStore& st = stores[tid];
        const size_t c = static_cast<size_t>(1.25 * n_cells / (nt < 1 ? 1 : nt)) + 64;
        const bool three = dimension == 3;
        if(has_vertices) st.vertices.reserve(c * (three ? 3 * 29 : 2 * 7));
        if(has_neighbors) st.neighbors.reserve(c * (three ? 17 : 7));
        st.facet_sizes.reserve(c * (three ? 17 : 7));
        st.facet_vertices.reserve(c * (three ? 87 : 14));
        if(has_signatures) {
            st.sig_sizes.reserve(c * (three ? 29 : 7));
            st.sig_ids.reserve(c * (three ? 116 : 21));
        }
    }

    ~PackedVoronoiResult() {
        delete [] flags;
        delete [] records;
        delete [] vertex_offsets;
        delete [] facet_offsets;
        delete [] ref_offsets;
        delete [] sig_offsets;
    }

    // Copy one cell's scratch data into this thread's buffers.
    void append(int tid, int id, const CellScratch& out) {
        ThreadStore& st = stores[tid];
        CellRecord& rec = records[id];
        const int64_t nv = out.n_vertices;
        rec.thread = tid;
        rec.n_vertices = static_cast<int32_t>(nv);
        rec.vertex_start = st.n_vertices;
        st.n_vertices += nv;
        if(has_vertices) {
            st.vertices.insert(st.vertices.end(), out.vertices.begin(),
                               out.vertices.begin() + nv * dimension);
        }
        rec.n_facets = static_cast<int32_t>(out.facet_offsets.size() - 1);
        rec.facet_start = static_cast<int64_t>(st.facet_sizes.size());
        if(has_neighbors) {
            st.neighbors.insert(st.neighbors.end(), out.neighbors.begin(),
                                out.neighbors.end());
        }
        for(size_t j = 0; j + 1 < out.facet_offsets.size(); ++j) {
            st.facet_sizes.push_back(static_cast<int32_t>(
                out.facet_offsets[j + 1] - out.facet_offsets[j]));
        }
        rec.n_refs = static_cast<int32_t>(out.facet_vertices.size());
        rec.ref_start = static_cast<int64_t>(st.facet_vertices.size());
        st.facet_vertices.insert(st.facet_vertices.end(),
                                 out.facet_vertices.begin(),
                                 out.facet_vertices.end());
        if(has_signatures) {
            for(size_t k = 0; k + 1 < out.signature_offsets.size(); ++k) {
                st.sig_sizes.push_back(static_cast<int32_t>(
                    out.signature_offsets[k + 1] - out.signature_offsets[k]));
            }
            rec.n_sig_ids = static_cast<int32_t>(out.signature_ids.size());
            rec.sig_start = static_cast<int64_t>(st.sig_ids.size());
            st.sig_ids.insert(st.sig_ids.end(), out.signature_ids.begin(),
                              out.signature_ids.end());
        } else {
            rec.n_sig_ids = 0;
            rec.sig_start = 0;
        }
    }

    // Compute the per-cell offsets with a chunked parallel prefix sum.
    void finalize(int nt) {
        const int n = n_cells;
        vertex_offsets = new int64_t[n];
        facet_offsets = new int64_t[n];
        ref_offsets = new int64_t[n];
        sig_offsets = new int64_t[n];
        int chunks = nt < 1 ? 1 : nt;
        if(chunks > n) chunks = n < 1 ? 1 : n;
        vector<int64_t> sums(4 * chunks, 0);
        #pragma omp parallel for num_threads(chunks) schedule(static, 1)
        for(int t = 0; t < chunks; ++t) {
            const int lo = (int)((int64_t)n * t / chunks);
            const int hi = (int)((int64_t)n * (t + 1) / chunks);
            int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for(int i = lo; i < hi; ++i) {
                s0 += records[i].n_vertices;
                s1 += records[i].n_facets;
                s2 += records[i].n_refs;
                s3 += records[i].n_sig_ids;
            }
            sums[4 * t] = s0; sums[4 * t + 1] = s1;
            sums[4 * t + 2] = s2; sums[4 * t + 3] = s3;
        }
        int64_t run[4] = {0, 0, 0, 0};
        for(int t = 0; t < chunks; ++t) {
            for(int k = 0; k < 4; ++k) {
                const int64_t v = sums[4 * t + k];
                sums[4 * t + k] = run[k];
                run[k] += v;
            }
        }
        total_vertices = run[0]; total_facets = run[1];
        total_refs = run[2]; total_sig_ids = run[3];
        #pragma omp parallel for num_threads(chunks) schedule(static, 1)
        for(int t = 0; t < chunks; ++t) {
            const int lo = (int)((int64_t)n * t / chunks);
            const int hi = (int)((int64_t)n * (t + 1) / chunks);
            int64_t s0 = sums[4 * t], s1 = sums[4 * t + 1];
            int64_t s2 = sums[4 * t + 2], s3 = sums[4 * t + 3];
            for(int i = lo; i < hi; ++i) {
                vertex_offsets[i] = s0; s0 += records[i].n_vertices;
                facet_offsets[i] = s1; s1 += records[i].n_facets;
                ref_offsets[i] = s2; s2 += records[i].n_refs;
                sig_offsets[i] = s3; s3 += records[i].n_sig_ids;
            }
        }
    }
};

namespace {

void classify_boundary(
    PackedVoronoiResult* result,
    int owner,
    int neighbor,
    int n_inner
) {
    if(owner < n_inner && neighbor >= n_inner) {
        result->flags[owner] |= INTERFACE_BOUNDARY_FLAG;
    }
    if(owner < n_inner && neighbor < 0) {
        result->flags[owner] |= WALL_BOUNDARY_FLAG;
    }
}

}  // namespace

void* container_3d_create(double ax_, double bx_, double ay_, double by_,
  double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_, int init_mem_, int nt_) {
  
  return (void*)new container_3d(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_,
      (bool)py_, (bool)pz_, init_mem_, nt_);
}

void put_particles_interleaved_3d(
    void* container_3d_, int n_, const double* points_, int nt_
) {
  container_3d* c = (container_3d*)container_3d_;
  #pragma omp parallel for num_threads(nt_)
  for(int i = 0; i < n_; ++i) {
    const double* point = points_ + 3 * i;
    c->put_parallel(i, point[0], point[1], point[2]);
  }
  c->put_reconcile_overflow();
}

void* container_poly_3d_create(double ax_, double bx_, double ay_, double by_,
  double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_, int init_mem_, int nt_) {

  return (void*)new container_poly_3d(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_,
      (bool)px_, (bool)py_, (bool)pz_, init_mem_, nt_);
}

void put_particles_radii_3d(
    void* container_3d_, int n_, const double* points_, const double* radii_,
    int nt_
) {
  container_poly_3d* c = (container_poly_3d*)container_3d_;
  #pragma omp parallel for num_threads(nt_)
  for(int i = 0; i < n_; ++i) {
    const double* point = points_ + 3 * i;
    c->put_parallel(i, point[0], point[1], point[2], radii_[i]);
  }
  c->put_reconcile_overflow();
}

void* container_2d_create(double ax_, double bx_, double ay_, double by_,
  int nx_, int ny_, int px_, int py_, int init_mem_, int nt_) {

  return (void*)new container_2d(ax_, bx_, ay_, by_, nx_, ny_, (bool)px_,
      (bool)py_, init_mem_, nt_);
}

void put_particles_interleaved_2d(
    void* container_2d_, int n_, const double* points_, int nt_
) {
  container_2d* c = (container_2d*)container_2d_;
  #pragma omp parallel for num_threads(nt_)
  for(int i = 0; i < n_; ++i) {
    const double* point = points_ + 2 * i;
    c->put_parallel(i, point[0], point[1]);
  }
  c->put_reconcile_overflow();
}

void* container_poly_2d_create(double ax_, double bx_, double ay_, double by_,
  int nx_, int ny_, int px_, int py_, int init_mem_, int nt_) {

  return (void*)new container_poly_2d(ax_, bx_, ay_, by_, nx_, ny_, (bool)px_,
      (bool)py_, init_mem_, nt_);
}

void put_particles_radii_2d(
    void* container_2d_, int n_, const double* points_, const double* radii_,
    int nt_
) {
  container_poly_2d* c = (container_poly_2d*)container_2d_;
  #pragma omp parallel for num_threads(nt_)
  for(int i = 0; i < n_; ++i) {
    const double* point = points_ + 2 * i;
    c->put_parallel(i, point[0], point[1], radii_[i]);
  }
  c->put_reconcile_overflow();
}

// Counts the pairs of particles with identical positions. Coincident particles
// are always in the same block, so it is enough to compare the particles in
// each block with each other; there are only a few per block, so this is
// cheap. Voronoi cells are not defined for coincident generators: Voro++ either
// fails to compute them or returns overlapping cells. The IDs of one such pair
// are returned in (first, second).
template<class C>
static int64_t count_coincident(C* con, int n_blocks, int dim, int nt,
                                int* first, int* second) {
    int64_t count = 0;
    *first = *second = -1;
    #pragma omp parallel for num_threads(nt) schedule(static) reduction(+:count)
    for(int b = 0; b < n_blocks; ++b) {
        const double* pp = con->p[b];
        for(int i = 0; i < con->co[b]; ++i) {
            for(int j = i + 1; j < con->co[b]; ++j) {
                bool same = true;
                for(int k = 0; k < dim; ++k) {
                    if(pp[con->ps * i + k] != pp[con->ps * j + k]) { same = false; break; }
                }
                if(same) {
                    count++;
                    #pragma omp critical(voroplex_coincident)
                    if(*first < 0) {
                        *first = min(con->id[b][i], con->id[b][j]);
                        *second = max(con->id[b][i], con->id[b][j]);
                    }
                }
            }
        }
    }
    return count;
}

int64_t count_coincident_2d(void* container_2d_, bool poly, int nt, int* first,
                            int* second) {
    if(poly) {
        container_poly_2d* con = (container_poly_2d*)container_2d_;
        return count_coincident(con, con->nxy, 2, nt, first, second);
    }
    container_2d* con = (container_2d*)container_2d_;
    return count_coincident(con, con->nxy, 2, nt, first, second);
}

int64_t count_coincident_3d(void* container_3d_, bool poly, int nt, int* first,
                            int* second) {
    if(poly) {
        container_poly_3d* con = (container_poly_3d*)container_3d_;
        return count_coincident(con, con->nxyz, 3, nt, first, second);
    }
    container_3d* con = (container_3d*)container_3d_;
    return count_coincident(con, con->nxyz, 3, nt, first, second);
}

// Whether a container class computes a radical (power) tessellation.
template<class Con> struct is_radical { static const bool value = false; };
template<> struct is_radical<container_poly_2d> { static const bool value = true; };
template<> struct is_radical<container_poly_3d> { static const bool value = true; };

// Stores the requested properties of a computed 3D cell, given the particle
// position (x,y,z). Voro++ computes the centroid relative to the particle, and
// its maximum radius squared at four times its true value.
template<class Cell>
static void store_properties_3d(Cell& c, int id, double x, double y, double z,
                                const CellProperties& P) {
    if(P.measure) P.measure[id] = c.volume();
    if(P.boundary) P.boundary[id] = c.surface_area();
    if(P.centroid) {
        double cx, cy, cz;
        c.centroid(cx, cy, cz);
        double* q = P.centroid + 3 * (int64_t)id;
        q[0] = x + cx; q[1] = y + cy; q[2] = z + cz;
    }
    if(P.num_faces) P.num_faces[id] = c.number_of_faces();
    if(P.num_vertices) P.num_vertices[id] = c.p;
    if(P.num_edges) P.num_edges[id] = c.number_of_edges();
    if(P.edge_length) P.edge_length[id] = c.total_edge_distance();
    if(P.max_radius) P.max_radius[id] = 0.5 * sqrt(c.max_radius_squared());
    if(P.second_moment || P.second_moment_tensor) {
        double m[6];
        c.second_moment(m);
        if(P.second_moment) P.second_moment[id] = m[0] + m[3] + m[5];
        if(P.second_moment_tensor) {
            double* t = P.second_moment_tensor + 9 * (int64_t)id;
            t[0] = m[0]; t[1] = t[3] = m[1]; t[2] = t[6] = m[2];
            t[4] = m[3]; t[5] = t[7] = m[4]; t[8] = m[5];
        }
    }
}

// Stores the requested properties of a computed 2D cell, given the particle
// position (x,y).
template<class Cell>
static void store_properties_2d(Cell& c, int id, double x, double y,
                                const CellProperties& P) {
    if(P.measure) P.measure[id] = c.area();
    if(P.boundary) P.boundary[id] = c.perimeter();
    if(P.centroid) {
        double cx, cy;
        c.centroid(cx, cy);
        double* q = P.centroid + 2 * (int64_t)id;
        q[0] = x + cx; q[1] = y + cy;
    }
    if(P.num_vertices) P.num_vertices[id] = c.p;
    if(P.max_radius) P.max_radius[id] = 0.5 * sqrt(c.max_radius_squared());
    if(P.second_moment || P.second_moment_tensor) {
        double m[3];
        c.second_moment(m);
        if(P.second_moment) P.second_moment[id] = m[0] + m[2];
        if(P.second_moment_tensor) {
            double* t = P.second_moment_tensor + 4 * (int64_t)id;
            t[0] = m[0]; t[1] = t[2] = m[1]; t[3] = m[2];
        }
    }
}

// Stores the properties of an empty cell: zero for its size and its counts,
// and NaN for its centroid and maximum radius, which are not defined.
static void store_empty_properties(int id, int dim, const CellProperties& P) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if(P.measure) P.measure[id] = 0.;
    if(P.boundary) P.boundary[id] = 0.;
    if(P.centroid) for(int k = 0; k < dim; ++k) P.centroid[(int64_t)dim * id + k] = nan;
    if(P.num_faces) P.num_faces[id] = 0;
    if(P.num_vertices) P.num_vertices[id] = 0;
    if(P.num_edges) P.num_edges[id] = 0;
    if(P.edge_length) P.edge_length[id] = 0.;
    if(P.max_radius) P.max_radius[id] = nan;
    if(P.second_moment) P.second_moment[id] = 0.;
    if(P.second_moment_tensor) {
        for(int k = 0; k < dim * dim; ++k)
            P.second_moment_tensor[(int64_t)dim * dim * id + k] = 0.;
    }
}

// Stores an empty cell. In a radical tessellation, a generator can have an
// empty cell, when the cells of larger neighbors cover it; Voro++'s
// compute_cell then returns false.
static void append_empty(PackedVoronoiResult* result, int tid, int id,
                         CellScratch& output, const CellProperties& props) {
    result->flags[id] = 0;
    store_empty_properties(id, result->dimension, props);
    output.n_vertices = 0;
    output.vertices.clear();
    output.neighbors.clear();
    output.facet_offsets.assign(1, 0);
    output.facet_vertices.clear();
    output.signature_offsets.assign(1, 0);
    output.signature_ids.clear();
    result->append(tid, id, output);
}

// The error raised when Voro++ could not compute some of the cells.
static runtime_error cells_failed(const char* what, int found, int n_cells,
                                  int invalid) {
    string msg = string("Failed to construct packed ") + what + ": ";
    if(found != n_cells) {
        msg += to_string(n_cells - found) + " of " + to_string(n_cells)
            + " cells could not be computed by Voro++";
    } else {
        msg += to_string(invalid) + " cells had inconsistent connectivity";
    }
    return runtime_error(msg);
}

// The geometry loop, for either cell class: voronoicell_neighbor_2d records
// the neighbor of each facet, and voronoicell_2d does not, which is a little
// faster.
template<class Con, class Cell>
static PackedVoronoiResult* geometry_2d(
    Con* con,
    int n_cells,
    int nt,
    int n_inner,
    bool with_neighbors,
    const CellProperties& props
) {
    PackedVoronoiResult* result = new PackedVoronoiResult(
        n_cells, 2, true, false, nt, with_neighbors
    );
    int found = 0;
    int invalid = 0;
    typename Con::iterator cli;

    #pragma omp parallel num_threads(nt)
    {
        Cell cell(*(con));
        CellScratch output;
        const int tid = omp_get_thread_num();
        result->reserve(tid, omp_get_num_threads());
        vector<int> local_neighbors;

        #pragma omp for schedule(dynamic) reduction(+:found,invalid)
        for(cli=con->begin(); cli<con->end(); cli++) {
            const int ijk = (*cli).ijk;
            const int q = (*cli).q;
            const int id = con->id[ijk][q];
            if(id >= n_cells) continue;
            PROFILE_START
            if(!con->compute_cell(cell, cli)) {
                if(is_radical<Con>::value) { append_empty(result, tid, id, output, props); found++; }
                continue;
            }
            PROFILE_VORO_DONE

            result->flags[id] = 0;
            double* pp = con->p[ijk] + con->ps * q;
            const double x = *(pp++);
            const double y = *pp;
            store_properties_2d(cell, id, x, y, props);
            cell.vertices(x, y, output.vertices);
            output.n_vertices = static_cast<int64_t>(
                output.vertices.size() / 2
            );
            if(with_neighbors) cell.neighbors(local_neighbors);
            output.neighbors.clear();
            output.facet_offsets.clear();
            output.facet_vertices.clear();
            output.facet_offsets.push_back(0);

            if(cell.p > 0) {
                int vertex = 0;
                int traversed = 0;
                do {
                    const int next = cell.ed[2 * vertex];
                    if(next < 0 || next >= cell.p || traversed >= cell.p) {
                        invalid++;
                        break;
                    }
                    output.facet_vertices.push_back(vertex);
                    output.facet_vertices.push_back(next);
                    output.facet_offsets.push_back(
                        static_cast<int64_t>(output.facet_vertices.size())
                    );

                    if(with_neighbors) {
                        const int neighbor = local_neighbors[vertex];
                        output.neighbors.push_back(neighbor);
                        classify_boundary(result, id, neighbor, n_inner);
                    }
                    vertex = next;
                    traversed++;
                } while(vertex != 0);
                if(traversed != cell.p) {
                    invalid++;
                }
            }

            if(with_neighbors
               && output.neighbors.size() + 1 != output.facet_offsets.size()) {
                invalid++;
            }
            result->append(tid, id, output);
            found++;
            PROFILE_CELL_DONE
        }
    }

    if(found != n_cells || invalid != 0) {
        delete result;
        throw cells_failed("2D Voronoi geometry", found, n_cells, invalid);
    }
    result->finalize(nt);
    return result;
}

PackedVoronoiResult* compute_voronoi_geometry_2d_packed(
    void* container_2d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_neighbors,
    const CellProperties* props
) {
    if(poly) {
        container_poly_2d* con = (container_poly_2d*)container_2d_;
        if(with_neighbors) {
            return geometry_2d<container_poly_2d, voronoicell_neighbor_2d>(
                con, n_cells, nt, n_inner, true, *props);
        }
        return geometry_2d<container_poly_2d, voronoicell_2d>(
            con, n_cells, nt, n_inner, false, *props);
    }
    container_2d* con = (container_2d*)container_2d_;
    if(with_neighbors) {
        return geometry_2d<container_2d, voronoicell_neighbor_2d>(
            con, n_cells, nt, n_inner, true, *props);
    }
    return geometry_2d<container_2d, voronoicell_2d>(con, n_cells, nt, n_inner, false, *props);
}

// The geometry loop, for either cell class: voronoicell_neighbor_3d records
// the neighbor of each facet, and voronoicell_3d does not, which is a little
// faster.
template<class Con, class Cell>
static PackedVoronoiResult* geometry_3d(
    Con* con,
    int n_cells,
    int nt,
    int n_inner,
    bool with_neighbors,
    const CellProperties& props
) {
    PackedVoronoiResult* result = new PackedVoronoiResult(
        n_cells, 3, true, false, nt, with_neighbors
    );
    int found = 0;
    int invalid = 0;
    typename Con::iterator cli;

    #pragma omp parallel num_threads(nt)
    {
        Cell cell(*(con));
        CellScratch output;
        const int tid = omp_get_thread_num();
        result->reserve(tid, omp_get_num_threads());
        vector<int> encoded_faces;

        #pragma omp for schedule(dynamic) reduction(+:found,invalid)
        for(cli=con->begin(); cli<con->end(); cli++) {
            const int ijk = (*cli).ijk;
            const int q = (*cli).q;
            const int id = con->id[ijk][q];
            if(id >= n_cells) continue;
            PROFILE_START
            if(!con->compute_cell(cell, cli)) {
                if(is_radical<Con>::value) { append_empty(result, tid, id, output, props); found++; }
                continue;
            }
            PROFILE_VORO_DONE

            result->flags[id] = 0;
            double* pp = con->p[ijk] + con->ps * q;
            const double x = *(pp++);
            const double y = *(pp++);
            const double z = *pp;
            store_properties_3d(cell, id, x, y, z, props);
            cell.vertices(x, y, z, output.vertices);
            output.n_vertices = static_cast<int64_t>(
                output.vertices.size() / 3
            );
            if(with_neighbors) cell.neighbors(output.neighbors);
            else output.neighbors.clear();

            cell.face_vertices(encoded_faces);
            output.facet_offsets.clear();
            output.facet_vertices.clear();
            output.facet_offsets.push_back(0);
            size_t cursor = 0;
            const size_t n_faces = with_neighbors
                ? output.neighbors.size() : (size_t)cell.number_of_faces();
            for(size_t face = 0; face < n_faces; ++face) {
                if(cursor >= encoded_faces.size()) {
                    invalid++;
                    break;
                }
                const int order = encoded_faces[cursor++];
                if(order < 0 || cursor + order > encoded_faces.size()) {
                    invalid++;
                    break;
                }
                for(int j = 0; j < order; ++j) {
                    const int vertex = encoded_faces[cursor++];
                    if(vertex < 0 || vertex >= output.n_vertices) {
                        invalid++;
                        continue;
                    }
                    output.facet_vertices.push_back(vertex);
                }
                output.facet_offsets.push_back(
                    static_cast<int64_t>(output.facet_vertices.size())
                );

                if(with_neighbors) {
                    classify_boundary(result, id, output.neighbors[face], n_inner);
                }
            }

            if(
                cursor != encoded_faces.size()
                || n_faces + 1 != output.facet_offsets.size()
            ) {
                invalid++;
            }
            result->append(tid, id, output);
            found++;
            PROFILE_CELL_DONE
        }
    }

    if(found != n_cells || invalid != 0) {
        delete result;
        throw cells_failed("3D Voronoi geometry", found, n_cells, invalid);
    }
    result->finalize(nt);
    return result;
}

PackedVoronoiResult* compute_voronoi_geometry_3d_packed(
    void* container_3d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_neighbors,
    const CellProperties* props
) {
    if(poly) {
        container_poly_3d* con = (container_poly_3d*)container_3d_;
        if(with_neighbors) {
            return geometry_3d<container_poly_3d, voronoicell_neighbor_3d>(
                con, n_cells, nt, n_inner, true, *props);
        }
        return geometry_3d<container_poly_3d, voronoicell_3d>(
            con, n_cells, nt, n_inner, false, *props);
    }
    container_3d* con = (container_3d*)container_3d_;
    if(with_neighbors) {
        return geometry_3d<container_3d, voronoicell_neighbor_3d>(
            con, n_cells, nt, n_inner, true, *props);
    }
    return geometry_3d<container_3d, voronoicell_3d>(con, n_cells, nt, n_inner, false, *props);
}

template<class Con>
static PackedVoronoiResult* topology_2d(
    Con* con,
    int n_cells,
    int nt,
    int n_inner,
    bool with_vertices,
    const CellProperties& props
) {
    PackedVoronoiResult* result = new PackedVoronoiResult(
        n_cells, 2, with_vertices, true, nt
    );
    int found = 0;
    int invalid = 0;
    typename Con::iterator cli;

    #pragma omp parallel num_threads(nt)
    {
        voronoicell_neighbor_2d cell(*(con));
        CellScratch output;
        const int tid = omp_get_thread_num();
        result->reserve(tid, omp_get_num_threads());
        vector<int> local_neighbors;
        vector<int32_t> signature;

        #pragma omp for schedule(dynamic) reduction(+:found,invalid)
        for(cli=con->begin(); cli<con->end(); cli++) {
            const int ijk = (*cli).ijk;
            const int q = (*cli).q;
            const int id = con->id[ijk][q];
            if(id >= n_cells) continue;
            PROFILE_START
            if(!con->compute_cell(cell, cli)) {
                if(is_radical<Con>::value) { append_empty(result, tid, id, output, props); found++; }
                continue;
            }
            PROFILE_VORO_DONE

            result->flags[id] = 0;
            output.n_vertices = cell.p;
            const double* pp = con->p[ijk] + con->ps * q;
            const double x = pp[0], y = pp[1];
            store_properties_2d(cell, id, x, y, props);
            if(with_vertices) cell.vertices(x, y, output.vertices);
            cell.neighbors(local_neighbors);
            output.neighbors.clear();
            output.facet_offsets.assign(1, 0);
            output.facet_vertices.clear();

            if(cell.p > 0) {
                int vertex = 0;
                int traversed = 0;
                do {
                    const int next = cell.ed[2 * vertex];
                    if(next < 0 || next >= cell.p || traversed >= cell.p) {
                        invalid++;
                        break;
                    }
                    output.facet_vertices.push_back(vertex);
                    output.facet_vertices.push_back(next);
                    output.facet_offsets.push_back(
                        static_cast<int64_t>(output.facet_vertices.size())
                    );
                    const int neighbor = local_neighbors[vertex];
                    output.neighbors.push_back(neighbor);
                    classify_boundary(
                        result, id, neighbor, n_inner
                    );
                    vertex = next;
                    traversed++;
                } while(vertex != 0);
                if(traversed != cell.p) {
                    invalid++;
                }
            }

            output.signature_offsets.assign(1, 0);
            output.signature_ids.clear();
            for(int vertex = 0; vertex < cell.p; ++vertex) {
                const int previous = cell.ed[2 * vertex + 1];
                if(previous < 0 || previous >= cell.p) {
                    invalid++;
                    continue;
                }
                signature.clear();
                signature.push_back(id);
                signature.push_back(local_neighbors[vertex]);
                signature.push_back(local_neighbors[previous]);
                sort(signature.begin(), signature.end());
                signature.erase(
                    unique(signature.begin(), signature.end()),
                    signature.end()
                );
                output.signature_ids.insert(
                    output.signature_ids.end(),
                    signature.begin(),
                    signature.end()
                );
                output.signature_offsets.push_back(
                    static_cast<int64_t>(output.signature_ids.size())
                );
            }

            if(
                output.neighbors.size() + 1 != output.facet_offsets.size()
                || output.signature_offsets.size()
                    != static_cast<size_t>(cell.p + 1)
            ) {
                invalid++;
            }
            result->append(tid, id, output);
            found++;
            PROFILE_CELL_DONE
        }
    }

    if(found != n_cells || invalid != 0) {
        delete result;
        throw cells_failed("2D Voronoi topology", found, n_cells, invalid);
    }
    result->finalize(nt);
    return result;
}

PackedVoronoiResult* compute_voronoi_topology_2d_packed(
    void* container_2d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_vertices,
    const CellProperties* props
) {
    if(poly) {
        return topology_2d((container_poly_2d*)container_2d_, n_cells, nt,
                           n_inner, with_vertices, *props);
    }
    return topology_2d((container_2d*)container_2d_, n_cells, nt, n_inner,
                       with_vertices, *props);
}

template<class Con>
static PackedVoronoiResult* topology_3d(
    Con* con,
    int n_cells,
    int nt,
    int n_inner,
    bool with_vertices,
    const CellProperties& props
) {
    PackedVoronoiResult* result = new PackedVoronoiResult(
        n_cells, 3, with_vertices, true, nt
    );
    int found = 0;
    int invalid = 0;
    typename Con::iterator cli;

    #pragma omp parallel num_threads(nt)
    {
        voronoicell_neighbor_3d cell(*(con));
        CellScratch output;
        const int tid = omp_get_thread_num();
        result->reserve(tid, omp_get_num_threads());
        vector<int> encoded_faces;
        vector<vector<int32_t> > signatures;

        #pragma omp for schedule(dynamic) reduction(+:found,invalid)
        for(cli=con->begin(); cli<con->end(); cli++) {
            const int ijk = (*cli).ijk;
            const int q = (*cli).q;
            const int id = con->id[ijk][q];
            if(id >= n_cells) continue;
            PROFILE_START
            if(!con->compute_cell(cell, cli)) {
                if(is_radical<Con>::value) { append_empty(result, tid, id, output, props); found++; }
                continue;
            }
            PROFILE_VORO_DONE

            result->flags[id] = 0;
            output.n_vertices = cell.p;
            const double* pp = con->p[ijk] + con->ps * q;
            const double x = pp[0], y = pp[1], z = pp[2];
            store_properties_3d(cell, id, x, y, z, props);
            if(with_vertices) cell.vertices(x, y, z, output.vertices);
            cell.neighbors(output.neighbors);
            if((int)signatures.size() < cell.p) signatures.resize(cell.p);
            for(int vertex = 0; vertex < cell.p; ++vertex) signatures[vertex].clear();
            for(int vertex = 0; vertex < cell.p; ++vertex) {
                signatures[vertex].push_back(id);
            }

            cell.face_vertices(encoded_faces);
            output.facet_offsets.assign(1, 0);
            output.facet_vertices.clear();
            size_t cursor = 0;
            for(size_t face = 0; face < output.neighbors.size(); ++face) {
                if(cursor >= encoded_faces.size()) {
                    invalid++;
                    break;
                }
                const int order = encoded_faces[cursor++];
                if(order < 0 || cursor + order > encoded_faces.size()) {
                    invalid++;
                    break;
                }
                const int neighbor = output.neighbors[face];
                for(int j = 0; j < order; ++j) {
                    const int vertex = encoded_faces[cursor++];
                    if(vertex < 0 || vertex >= cell.p) {
                        invalid++;
                        continue;
                    }
                    output.facet_vertices.push_back(vertex);
                    signatures[vertex].push_back(neighbor);
                }
                output.facet_offsets.push_back(
                    static_cast<int64_t>(output.facet_vertices.size())
                );
                classify_boundary(result, id, neighbor, n_inner);
            }

            output.signature_offsets.assign(1, 0);
            output.signature_ids.clear();
            for(int vertex = 0; vertex < cell.p; ++vertex) {
                vector<int32_t>& signature = signatures[vertex];
                sort(signature.begin(), signature.end());
                signature.erase(
                    unique(signature.begin(), signature.end()),
                    signature.end()
                );
                output.signature_ids.insert(
                    output.signature_ids.end(),
                    signature.begin(),
                    signature.end()
                );
                output.signature_offsets.push_back(
                    static_cast<int64_t>(output.signature_ids.size())
                );
            }

            if(
                cursor != encoded_faces.size()
                || output.neighbors.size() + 1 != output.facet_offsets.size()
                || output.signature_offsets.size()
                    != static_cast<size_t>(cell.p + 1)
            ) {
                invalid++;
            }
            result->append(tid, id, output);
            found++;
            PROFILE_CELL_DONE
        }
    }

    if(found != n_cells || invalid != 0) {
        delete result;
        throw cells_failed("3D Voronoi topology", found, n_cells, invalid);
    }
    result->finalize(nt);
    return result;
}

PackedVoronoiResult* compute_voronoi_topology_3d_packed(
    void* container_3d_,
    bool poly,
    int n_cells,
    int nt,
    int n_inner,
    bool with_vertices,
    const CellProperties* props
) {
    if(poly) {
        return topology_3d((container_poly_3d*)container_3d_, n_cells, nt,
                           n_inner, with_vertices, *props);
    }
    return topology_3d((container_3d*)container_3d_, n_cells, nt, n_inner,
                       with_vertices, *props);
}

// The loops for representation=None: each cell is computed with Voro++'s
// simpler cell class, which does not record neighbors, and only the requested
// properties are stored, straight into the output arrays. Returns the number
// of cells that Voro++ could not compute.
template<class Con, class Cell>
static int properties_2d(Con* con, int n_cells, int nt, const CellProperties& props) {
    int found = 0;
    typename Con::iterator cli;
    #pragma omp parallel num_threads(nt)
    {
        Cell cell(*(con));
        #pragma omp for schedule(dynamic) reduction(+:found)
        for(cli=con->begin(); cli<con->end(); cli++) {
            const int ijk = (*cli).ijk, q = (*cli).q;
            const int id = con->id[ijk][q];
            if(id >= n_cells) continue;
            if(!con->compute_cell(cell, cli)) {
                if(is_radical<Con>::value) { store_empty_properties(id, 2, props); found++; }
                continue;
            }
            const double* pp = con->p[ijk] + con->ps * q;
            store_properties_2d(cell, id, pp[0], pp[1], props);
            found++;
        }
    }
    return n_cells - found;
}

template<class Con, class Cell>
static int properties_3d(Con* con, int n_cells, int nt, const CellProperties& props) {
    int found = 0;
    typename Con::iterator cli;
    #pragma omp parallel num_threads(nt)
    {
        Cell cell(*(con));
        #pragma omp for schedule(dynamic) reduction(+:found)
        for(cli=con->begin(); cli<con->end(); cli++) {
            const int ijk = (*cli).ijk, q = (*cli).q;
            const int id = con->id[ijk][q];
            if(id >= n_cells) continue;
            if(!con->compute_cell(cell, cli)) {
                if(is_radical<Con>::value) { store_empty_properties(id, 3, props); found++; }
                continue;
            }
            const double* pp = con->p[ijk] + con->ps * q;
            store_properties_3d(cell, id, pp[0], pp[1], pp[2], props);
            found++;
        }
    }
    return n_cells - found;
}

int compute_voronoi_properties_2d(void* container_2d_, bool poly, int n_cells,
                                  int nt, const CellProperties* props) {
    if(poly) return properties_2d<container_poly_2d, voronoicell_2d>(
        (container_poly_2d*)container_2d_, n_cells, nt, *props);
    return properties_2d<container_2d, voronoicell_2d>(
        (container_2d*)container_2d_, n_cells, nt, *props);
}

int compute_voronoi_properties_3d(void* container_3d_, bool poly, int n_cells,
                                  int nt, const CellProperties* props) {
    if(poly) return properties_3d<container_poly_3d, voronoicell_3d>(
        (container_poly_3d*)container_3d_, n_cells, nt, *props);
    return properties_3d<container_3d, voronoicell_3d>(
        (container_3d*)container_3d_, n_cells, nt, *props);
}

// Backward pass for per-cell vertices. For each local vertex v (one copy of a
// vertex in one cell) with a sorted signature [walls..., p_1, p_2, ...], form
// the equation rows: 2 (p_j - p_1) for each generator p_j after the base p_1,
// and the unit vector e_axis for each wall. Then y = A^{-T} g (or the
// least-squares solution if there are more than dim rows), and
//     grad[p_j] += 2 y_j (p_j - v),   grad[p_1] -= 2 (sum_j y_j) (p_1 - v).
// In a radical tessellation, with radii r, the equations have the extra
// right-hand side -(r_j^2 - r_1^2). The matrix A is the same, so the gradient
// with respect to the positions is unchanged, and
//     grad[r_j] -= 2 y_j r_j,   grad[r_1] += 2 (sum_j y_j) r_1.
// radii and grad_r are NULL for a standard Voronoi tessellation; grad_r may
// also be NULL if the gradient with respect to the radii is not needed.
// In a periodic domain, each generator is the periodic image nearest the
// vertex. Returns the number of vertices whose system was singular, and sets
// n_mismatched to the number whose nearest images were not at the same power
// distance |v - p|^2 - r^2 from the vertex (which happens if the periodic
// box is too small); both are skipped.
template<int D>
static int per_cell_backward_impl(
    int64_t n_vertices, const double* points, const double* radii,
    const double* vertices, const double* grad_v, const int64_t* sig_offsets,
    const int32_t* sig_ids, const double* box, double* grad_p, double* grad_r,
    int nt, int* n_mismatched
) {
    int singular = 0, mismatched = 0;
    #pragma omp parallel for num_threads(nt) schedule(dynamic, 1024) reduction(+:singular,mismatched)
    for(int64_t v = 0; v < n_vertices; ++v) {
        const int64_t s0 = sig_offsets[v], s1 = sig_offsets[v + 1];
        int n_walls = 0;
        while(s0 + n_walls < s1 && sig_ids[s0 + n_walls] < 0) n_walls++;
        if(s0 + n_walls >= s1) { singular++; continue; }
        const int base = sig_ids[s0 + n_walls];
        const double* vv = vertices + (int64_t)D * v;
        const double* g = grad_v + (int64_t)D * v;
        // In a periodic domain, each generator that defines the vertex is the
        // periodic image nearest to the vertex: all defining generators are at
        // the same distance R from it, and no generator is closer. box[k] is
        // the box length along a periodic axis, and 0 otherwise.
        double pb[D];
        for(int k = 0; k < D; ++k) {
            pb[k] = points[(int64_t)D * base + k];
            if(box[k] > 0.0) pb[k] -= box[k] * nearbyint((pb[k] - vv[k]) / box[k]);
        }
        double rb2 = 0.0;
        for(int k = 0; k < D; ++k) rb2 += (pb[k] - vv[k]) * (pb[k] - vv[k]);
        const double sb2 = radii == NULL ? 0.0 : radii[base] * radii[base];
        bool bad_image = false;

        // Rows of A; at most 16 rows (a vertex with more is skipped).
        const int n_rows = (int)(s1 - s0) - 1;
        // Too few rows means that two periodic images of the same generator
        // define this vertex, and were merged in its signature: the periodic
        // box is too small compared with the cells.
        if(n_rows < D) { mismatched++; continue; }
        if(n_rows > 16) { singular++; continue; }
        double A[16][D];
        double pg[16][D];
        int gen[16];
        int r = 0;
        for(int64_t e = s0; e < s1; ++e) {
            if(e == s0 + n_walls) continue;           // the base generator
            const int id = sig_ids[e];
            if(id < 0) {
                const int axis = (-id - 1) / 2;
                for(int k = 0; k < D; ++k) A[r][k] = (k == axis) ? 1.0 : 0.0;
                gen[r] = -1;
            } else {
                double rj2 = 0.0;
                for(int k = 0; k < D; ++k) {
                    double x = points[(int64_t)D * id + k];
                    if(box[k] > 0.0) x -= box[k] * nearbyint((x - vv[k]) / box[k]);
                    pg[r][k] = x;
                    A[r][k] = 2.0 * (x - pb[k]);
                    rj2 += (x - vv[k]) * (x - vv[k]);
                }
                // The nearest images must be at the same power distance from
                // the vertex. If not, the box is too small for the
                // nearest-image rule.
                const double sj2 = radii == NULL ? 0.0 : radii[id] * radii[id];
                if(fabs((rj2 - sj2) - (rb2 - sb2))
                   > 1e-6 * max(rb2 + sb2, rj2 + sj2)) bad_image = true;
                gen[r] = id;
            }
            r++;
        }
        if(bad_image) { mismatched++; continue; }

        double y[16];
        if(n_rows == D) {
            // y = A^{-T} g by explicit formulas.
            if(D == 2) {
                const double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
                if(det == 0.0) { singular++; continue; }
                y[0] = (A[1][1] * g[0] - A[1][0] * g[1]) / det;
                y[1] = (-A[0][1] * g[0] + A[0][0] * g[1]) / det;
            } else {
                double c[3][3];   // c[i] = row_{i+1} x row_{i+2}
                for(int i = 0; i < 3; ++i) {
                    const double* a = A[(i + 1) % 3];
                    const double* b = A[(i + 2) % 3];
                    c[i][0] = a[1] * b[2] - a[2] * b[1];
                    c[i][1] = a[2] * b[0] - a[0] * b[2];
                    c[i][2] = a[0] * b[1] - a[1] * b[0];
                }
                const double det = A[0][0] * c[0][0] + A[0][1] * c[0][1] + A[0][2] * c[0][2];
                if(det == 0.0) { singular++; continue; }
                for(int i = 0; i < 3; ++i)
                    y[i] = (c[i][0] * g[0] + c[i][1] * g[1] + c[i][2] * g[2]) / det;
            }
        } else {
            // Least squares: z = (A^T A)^{-1} g, y = A z.
            double M[D][D];
            for(int i = 0; i < D; ++i) for(int j = 0; j < D; ++j) {
                double sum = 0.0;
                for(int q = 0; q < n_rows; ++q) sum += A[q][i] * A[q][j];
                M[i][j] = sum;
            }
            double z[D];
            if(D == 2) {
                const double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
                if(det == 0.0) { singular++; continue; }
                z[0] = (M[1][1] * g[0] - M[0][1] * g[1]) / det;
                z[1] = (-M[1][0] * g[0] + M[0][0] * g[1]) / det;
            } else {
                double inv[3][3];
                inv[0][0] = M[1][1] * M[2][2] - M[1][2] * M[2][1];
                inv[0][1] = M[0][2] * M[2][1] - M[0][1] * M[2][2];
                inv[0][2] = M[0][1] * M[1][2] - M[0][2] * M[1][1];
                inv[1][0] = M[1][2] * M[2][0] - M[1][0] * M[2][2];
                inv[1][1] = M[0][0] * M[2][2] - M[0][2] * M[2][0];
                inv[1][2] = M[0][2] * M[1][0] - M[0][0] * M[1][2];
                inv[2][0] = M[1][0] * M[2][1] - M[1][1] * M[2][0];
                inv[2][1] = M[0][1] * M[2][0] - M[0][0] * M[2][1];
                inv[2][2] = M[0][0] * M[1][1] - M[0][1] * M[1][0];
                const double det = M[0][0] * inv[0][0] + M[0][1] * inv[1][0] + M[0][2] * inv[2][0];
                if(det == 0.0) { singular++; continue; }
                for(int i = 0; i < 3; ++i)
                    z[i] = (inv[i][0] * g[0] + inv[i][1] * g[1] + inv[i][2] * g[2]) / det;
            }
            for(int q = 0; q < n_rows; ++q) {
                double sum = 0.0;
                for(int k = 0; k < D; ++k) sum += A[q][k] * z[k];
                y[q] = sum;
            }
        }

        // Scatter the contributions into the generator gradients.
        double y_sum = 0.0;
        for(int q = 0; q < n_rows; ++q) {
            if(gen[q] < 0) continue;
            y_sum += y[q];
            const double* pj = pg[q];
            double* gp = grad_p + (int64_t)D * gen[q];
            for(int k = 0; k < D; ++k) {
                const double val = 2.0 * y[q] * (pj[k] - vv[k]);
                #pragma omp atomic
                gp[k] += val;
            }
            if(grad_r != NULL) {
                const double val = -2.0 * y[q] * radii[gen[q]];
                #pragma omp atomic
                grad_r[gen[q]] += val;
            }
        }
        double* gb = grad_p + (int64_t)D * base;
        for(int k = 0; k < D; ++k) {
            const double val = -2.0 * y_sum * (pb[k] - vv[k]);
            #pragma omp atomic
            gb[k] += val;
        }
        if(grad_r != NULL) {
            const double val = 2.0 * y_sum * radii[base];
            #pragma omp atomic
            grad_r[base] += val;
        }
    }
    *n_mismatched = mismatched;
    return singular;
}

int per_cell_backward(
    int dimension, int64_t n_vertices, const double* points,
    const double* radii, const double* vertices, const double* grad_v,
    const int64_t* sig_offsets, const int32_t* sig_ids, const double* box,
    double* grad_p, double* grad_r, int nt, int* n_mismatched
) {
    if(dimension == 2)
        return per_cell_backward_impl<2>(n_vertices, points, radii, vertices,
                                         grad_v, sig_offsets, sig_ids, box,
                                         grad_p, grad_r, nt, n_mismatched);
    return per_cell_backward_impl<3>(n_vertices, points, radii, vertices,
                                     grad_v, sig_offsets, sig_ids, box, grad_p,
                                     grad_r, nt, n_mismatched);
}

// Fan simplices for per-cell vertices. In 2D, each edge of a cell gives one
// triangle (generator, v0, v1); in 3D, each face is split into triangles from
// its first vertex, each giving a tetrahedron (generator, v0, v_{k+2},
// v_{k+1}). Voro++ lists the edges of a 2D cell counter-clockwise, and the
// vertices of each face of a 3D cell clockwise as seen from outside, so with
// this order every simplex is positively oriented: its signed volume is
// positive if the generator is inside the cell's facet planes. Vertex
// indices refer to the packed per-cell vertices: cell_vertex_offsets[i] plus
// the cell-local index. The order matches the reference NumPy version.

// Fills simplex_offsets (n_cells + 1 entries) and returns the total count.
int64_t per_cell_simplex_offsets(
    int dimension, int n_cells, const int64_t* cell_facet_offsets,
    const int64_t* facet_vertex_offsets, int64_t* simplex_offsets, int nt
) {
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int i = 0; i < n_cells; ++i) {
        int64_t count = 0;
        if(dimension == 2) {
            count = cell_facet_offsets[i + 1] - cell_facet_offsets[i];
        } else {
            for(int64_t f = cell_facet_offsets[i]; f < cell_facet_offsets[i + 1]; ++f) {
                const int64_t len = facet_vertex_offsets[f + 1] - facet_vertex_offsets[f];
                if(len > 2) count += len - 2;
            }
        }
        simplex_offsets[i + 1] = count;
    }
    simplex_offsets[0] = 0;
    for(int i = 0; i < n_cells; ++i) simplex_offsets[i + 1] += simplex_offsets[i];
    return simplex_offsets[n_cells];
}

void per_cell_fill_simplices(
    int dimension, int n_cells, const int64_t* cell_vertex_offsets,
    const int64_t* cell_facet_offsets, const int64_t* facet_vertex_offsets,
    const int32_t* facet_vertices, const int64_t* simplex_offsets,
    int64_t* owners, int64_t* simplices, int nt
) {
    #pragma omp parallel for num_threads(nt) schedule(dynamic, 256)
    for(int i = 0; i < n_cells; ++i) {
        const int64_t base = cell_vertex_offsets[i];
        int64_t s = simplex_offsets[i];
        for(int64_t f = cell_facet_offsets[i]; f < cell_facet_offsets[i + 1]; ++f) {
            const int64_t a = facet_vertex_offsets[f];
            const int64_t len = facet_vertex_offsets[f + 1] - a;
            if(dimension == 2) {
                owners[s] = i;
                simplices[2 * s] = base + facet_vertices[a];
                simplices[2 * s + 1] = base + facet_vertices[a + 1];
                s++;
            } else {
                for(int64_t k = 0; k + 2 < len; ++k) {
                    owners[s] = i;
                    simplices[3 * s] = base + facet_vertices[a];
                    simplices[3 * s + 1] = base + facet_vertices[a + 2 + k];
                    simplices[3 * s + 2] = base + facet_vertices[a + 1 + k];
                    s++;
                }
            }
        }
    }
}

// Sizes of the arrays that packed_result_fill writes.
int64_t packed_result_size(PackedVoronoiResult* r, int which) {
    switch(which) {
        case 0: return r->n_cells;
        case 1: return r->total_vertices;
        case 2: return r->total_facets;
        case 3: return r->total_refs;
        case 4: return r->total_sig_ids;
        case 5: return r->has_vertices ? 1 : 0;
        case 6: return r->has_signatures ? 1 : 0;
        case 8: return r->has_neighbors ? 1 : 0;
        default: return r->dimension;
    }
}

// Copy the result into caller-provided arrays, in parallel over cells. The
// arrays need not be initialized. vertices is used only if has_vertices, and
// the signature arrays only if has_signatures.
void packed_result_fill(
    PackedVoronoiResult* r, double* vertices,
    int64_t* cell_vertex_offsets, int64_t* cell_facet_offsets,
    int32_t* facet_neighbors, int64_t* facet_vertex_offsets,
    int32_t* facet_vertices, uint8_t* cell_boundary_flags,
    int64_t* vertex_signature_offsets, int32_t* vertex_signature_ids, int nt
) {
    const int n = r->n_cells;
    const int dim = r->dimension;
    #pragma omp parallel for num_threads(nt) schedule(dynamic, 256)
    for(int i = 0; i < n; ++i) {
        const CellRecord& rec = r->records[i];
        const ThreadStore& st = r->stores[rec.thread];
        cell_boundary_flags[i] = r->flags[i];
        const int64_t v0 = r->vertex_offsets[i];
        const int64_t f0 = r->facet_offsets[i];
        int64_t ref = r->ref_offsets[i];
        cell_vertex_offsets[i] = v0;
        cell_facet_offsets[i] = f0;
        if(r->has_vertices) {
            copy(st.vertices.begin() + rec.vertex_start * dim,
                 st.vertices.begin() + (rec.vertex_start + rec.n_vertices) * dim,
                 vertices + v0 * dim);
        }
        for(int j = 0; j < rec.n_facets; ++j) {
            if(r->has_neighbors) {
                facet_neighbors[f0 + j] = st.neighbors[rec.facet_start + j];
            }
            facet_vertex_offsets[f0 + j] = ref;
            ref += st.facet_sizes[rec.facet_start + j];
        }
        copy(st.facet_vertices.begin() + rec.ref_start,
             st.facet_vertices.begin() + rec.ref_start + rec.n_refs,
             facet_vertices + r->ref_offsets[i]);
        if(r->has_signatures) {
            int64_t sig = r->sig_offsets[i];
            for(int k = 0; k < rec.n_vertices; ++k) {
                vertex_signature_offsets[v0 + k] = sig;
                sig += st.sig_sizes[rec.vertex_start + k];
            }
            copy(st.sig_ids.begin() + rec.sig_start,
                 st.sig_ids.begin() + rec.sig_start + rec.n_sig_ids,
                 vertex_signature_ids + r->sig_offsets[i]);
        }
    }
    cell_vertex_offsets[n] = r->total_vertices;
    cell_facet_offsets[n] = r->total_facets;
    facet_vertex_offsets[r->total_facets] = r->total_refs;
    if(r->has_signatures) {
        vertex_signature_offsets[r->total_vertices] = r->total_sig_ids;
    }
}

void packed_result_profile(PackedVoronoiResult* r, double* voro_seconds,
                           double* extract_seconds) {
    *voro_seconds = 0.;
    *extract_seconds = 0.;
    for(size_t t = 0; t < r->voro_seconds.size(); ++t) {
        *voro_seconds += r->voro_seconds[t];
        *extract_seconds += r->extract_seconds[t];
    }
}

void dispose_container_3d(void* container_3d_, bool poly) {
  if(poly) delete (container_poly_3d*)container_3d_;
  else delete (container_3d*)container_3d_;
}

void dispose_container_2d(void* container_2d_, bool poly) {
  if(poly) delete (container_poly_2d*)container_2d_;
  else delete (container_2d*)container_2d_;
}

void dispose_packed_result(PackedVoronoiResult* result) {
  delete result;
}
