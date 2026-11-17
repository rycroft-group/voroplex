"""Tests of the tessellation itself: compute_voronoi_3d and compute_voronoi_2d."""
import numpy as np
import pytest

from conftest import compute, lattice, random_points

REPRESENTATIONS = ("geometry", "per_cell", "topology")
ARRAYS = ("measures", "cell_vertex_offsets", "cell_facet_offsets",
          "facet_neighbors", "facet_vertex_offsets", "facet_vertices",
          "cell_surface_flags")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("threads", [1, 4])
def test_cells_fill_the_box(dim, representation, threads):
    res = compute(dim, random_points(dim, 2000), representation, threads=threads)
    assert len(res) == 2000
    assert res.measures.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(res.measures > 0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [True, [True, False, False]])
def test_periodic_cells_fill_the_box(dim, periodic):
    if isinstance(periodic, list):
        periodic = periodic[:dim]
    for representation in ("geometry", "per_cell"):
        res = compute(dim, random_points(dim, 1000), representation,
                      periodic=periodic)
        assert res.measures.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_representations_agree(dim):
    """With one thread, all representations come from the same Voro++ cells."""
    p = random_points(dim, 1500)
    geo, per, top = (compute(dim, p, r, threads=1) for r in REPRESENTATIONS)
    for name in ARRAYS:
        assert np.array_equal(getattr(geo, name), getattr(per, name)), name
        assert np.array_equal(getattr(geo, name), getattr(top, name)), name
    assert np.array_equal(geo.vertices, per.vertices)
    assert np.array_equal(per.vertex_signature_ids, top.vertex_signature_ids)
    assert np.array_equal(per.vertex_signature_offsets, top.vertex_signature_offsets)


@pytest.mark.parametrize("dim", [2, 3])
def test_thread_count_does_not_change_cells(dim):
    p = random_points(dim, 3000)
    one, many = (compute(dim, p, threads=t) for t in (1, 8))
    assert np.allclose(one.measures, many.measures, rtol=0, atol=1e-14)
    # The order of the vertices within a cell may differ, but not their number.
    assert np.array_equal(np.diff(one.cell_vertex_offsets),
                          np.diff(many.cell_vertex_offsets))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_neighbors_false(dim, periodic):
    p = random_points(dim, 1500)
    with_nb = compute(dim, p, threads=1, periodic=periodic)
    without = compute(dim, p, threads=1, periodic=periodic, neighbors=False)
    assert not without.has_neighbors
    for name in ("measures", "vertices", "cell_vertex_offsets",
                 "cell_facet_offsets", "facet_vertex_offsets", "facet_vertices"):
        assert np.array_equal(getattr(with_nb, name), getattr(without, name)), name
    for get in (lambda r: r.facet_neighbors, lambda r: r.cell_surface_flags,
                lambda r: r.surface_cells, lambda r: r.boundary_cells,
                lambda r: r.interface_cells, lambda r: r.cell(0).neighbors,
                lambda r: r.cell(0).facets[0].neighbor):
        with pytest.raises(AttributeError, match="neighbors=False"):
            get(without)


@pytest.mark.parametrize("kwargs", [dict(representation="per_cell"),
                                    dict(representation="topology"),
                                    dict(n_inner=10)])
def test_neighbors_false_rejects_options_that_need_neighbors(kwargs):
    with pytest.raises(ValueError):
        compute(2, random_points(2, 100), neighbors=False, **kwargs)


@pytest.mark.parametrize("dim", [2, 3])
def test_generators_on_the_boundary(dim):
    """A lattice that includes the boundary: the cells on the boundary are cut."""
    m = 6
    g = np.linspace(0.0, 1.0, m)
    p = np.stack(np.meshgrid(*([g] * dim), indexing="ij"), -1).reshape(-1, dim)
    for representation in REPRESENTATIONS:
        res = compute(dim, p, representation)
        assert res.measures.sum() == pytest.approx(1.0, abs=1e-12)
        # Each cell is a box of side 1/(m-1), halved for each coordinate on the boundary.
        on_boundary = ((p == 0.0) | (p == 1.0)).sum(axis=1)
        expected = (1.0 / (m - 1)) ** dim / 2.0 ** on_boundary
        assert np.allclose(res.measures, expected, rtol=1e-12)


@pytest.mark.parametrize("dim,n", [(2, 620), (3, 10000)])
def test_generator_just_below_the_upper_boundary(dim, n):
    """A generator just below the upper boundary can be mapped by rounding to the
    block past the last one. It must still be put in the last block."""
    ax, bx = -7.955456837799035, 7.916707341387999
    nx = 12   # the number of blocks along x for these values of n
    x = np.nextafter(bx, -np.inf)
    assert np.floor((x - ax) * (nx / (bx - ax))) == nx   # the rounding occurs
    p = random_points(dim, n)
    p[:, 0] = ax + (bx - ax) * p[:, 0]
    p[0, 0] = x
    res = compute(dim, p, limits=[[ax, bx]] + [[0.0, 1.0]] * (dim - 1))
    assert res.measures.sum() == pytest.approx(bx - ax, rel=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_coincident_generators_raise(dim):
    p = random_points(dim, 200)
    p[7] = p[3]
    with pytest.raises(ValueError, match="identical positions.*3 and 7"):
        compute(dim, p)


def test_coincident_periodic_images_raise():
    """In a periodic box, (0, 0) and (1, 1) are the same point."""
    p = random_points(2, 100)
    p[0], p[1] = [0.0, 0.0], [1.0, 1.0]
    compute(2, p)                       # different points in a non-periodic box
    with pytest.raises(ValueError, match="identical positions"):
        compute(2, p, periodic=True)


def test_points_outside_the_limits_raise():
    p = random_points(2, 50)
    p[4] = [1.5, 0.5]
    with pytest.raises(ValueError, match="outside the provided limits"):
        compute(2, p)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_points_raise(bad):
    p = random_points(3, 50)
    p[2, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        compute(3, p)


def test_automatic_limits_contain_the_points():
    import voroplex
    p = random_points(3, 500) * 4.0 - 1.0
    res = voroplex.compute_voronoi_3d(p)
    assert np.all(res.limits[:, 0] < p.min(axis=0))
    assert np.all(res.limits[:, 1] > p.max(axis=0))
    assert res.measures.sum() == pytest.approx(np.prod(np.diff(res.limits, axis=1)))


@pytest.mark.parametrize("dim", [2, 3])
def test_n_compute_and_n_inner(dim):
    p = random_points(dim, 1000)
    full = compute(dim, p, threads=1)
    part = compute(dim, p, threads=1, n_compute=300, n_inner=200)
    assert len(part) == 300
    assert np.array_equal(part.measures, full.measures[:300])
    # Interface cells: inner cells with a neighbor that is an outer generator.
    expected = [
        i for i in range(200)
        if np.any(part.cell(i).neighbors >= 200)
    ]
    assert list(part.interface_cells) == expected
    on_boundary = [i for i in range(200) if np.any(part.cell(i).neighbors < 0)]
    assert list(part.boundary_cells) == on_boundary


def test_topology_rejects_periodic():
    with pytest.raises(ValueError, match="periodic"):
        compute(2, random_points(2, 100), "topology", periodic=True)


def test_repr():
    res = compute(3, random_points(3, 20), "per_cell", periodic=True)
    text = repr(res)
    assert "representation='per_cell'" in text and "n_cells=20" in text
    assert "periodic=[True, True, True]" in text
