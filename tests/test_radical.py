"""Tests of radical (power) tessellations, computed with generator radii.

The cell of generator i is the region where |x - p_i|^2 - r_i^2 is smallest.
A useful check of the gradients is the power energy

    E = sum_i int_{V_i} (|x - p_i|^2 - r_i^2) dx.

Its integrand is continuous across the cell faces, so moving the faces does
not change E to first order, and

    dE/dp_i = 2 m_i (p_i - c_i),    dE/dr_i = -2 r_i m_i,

where m_i and c_i are the measure and centroid of cell i. (The usual CVT
energy, without the r_i^2 term, has a discontinuous integrand in a radical
tessellation, so its gradient has no such simple form.)
"""
import numpy as np
import pytest

import voroplex
from conftest import (closed_form_cvt_gradient, compute, cvt_energy,
                      random_points, rel_err, requires_cuda)

REPRESENTATIONS = ("geometry", "per_cell", "topology")


def random_radii(n, scale=0.02, seed=99):
    return scale * np.random.default_rng(seed).random(n)


def cell_measures(points, vertices, owners, simplices, n):
    """Signed measure of each cell, from its fan simplices."""
    import torch
    q = vertices[simplices] - points[owners].unsqueeze(1)
    if q.shape[2] == 2:
        m = 0.5 * (q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0])
    else:
        m = (q[:, 0] * torch.cross(q[:, 1], q[:, 2], dim=1)).sum(1) / 6.0
    return torch.zeros(n, dtype=points.dtype).index_add_(0, owners, m)


def power_energy(points, radii, vertices, owners, simplices):
    m = cell_measures(points, vertices, owners, simplices, len(points))
    return cvt_energy(points, vertices, owners, simplices) - (radii ** 2 * m).sum()


def power_energy_and_gradients(dim, p, r, periodic=False):
    import torch
    res = compute(dim, p, "per_cell", periodic=periodic, radii=r)
    pts = torch.tensor(p, requires_grad=True)
    rad = torch.tensor(r, requires_grad=True)
    v = voroplex.cell_vertices(res, pts, rad)
    owners, simplices = voroplex.cell_fan_simplices(res)
    e = power_energy(pts, rad, v, owners, simplices)
    e.backward()
    return e.item(), pts.grad.numpy(), rad.grad.numpy(), res


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_equal_radii_give_the_voronoi_cells(dim, representation):
    p = random_points(dim, 1000)
    a = compute(dim, p, representation, threads=1)
    b = compute(dim, p, representation, threads=1, radii=np.full(1000, 0.05))
    assert b.is_radical and not a.is_radical
    assert np.allclose(a.measures, b.measures, rtol=0, atol=1e-15)
    assert np.array_equal(a.cell_vertex_offsets, b.cell_vertex_offsets)
    assert np.array_equal(a.facet_neighbors, b.facet_neighbors)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_cells_fill_the_box(dim, periodic):
    p = random_points(dim, 2000)
    r = random_radii(2000, scale=0.05)
    res = compute(dim, p, radii=r, periodic=periodic)
    assert res.measures.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(res.measures >= 0)
    # With these radii, some cells are empty.
    assert np.any(res.measures == 0)


@pytest.mark.parametrize("dim", [2, 3])
def test_adding_a_constant_to_the_squared_radii_changes_nothing(dim):
    p = random_points(dim, 1500)
    r = random_radii(1500)
    a = compute(dim, p, threads=1, radii=r)
    b = compute(dim, p, threads=1, radii=np.sqrt(r ** 2 + 0.03 ** 2))
    assert np.allclose(a.measures, b.measures, rtol=0, atol=1e-15)
    assert np.array_equal(a.facet_neighbors, b.facet_neighbors)


@pytest.mark.parametrize("dim", [2, 3])
def test_thread_count_does_not_change_cells(dim):
    p = random_points(dim, 3000)
    r = random_radii(3000, scale=0.05)
    one, many = (compute(dim, p, threads=t, radii=r) for t in (1, 8))
    assert np.allclose(one.measures, many.measures, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_empty_cell(dim, representation):
    """A small particle between two large ones has an empty cell."""
    p = np.full((3, dim), 0.5)
    p[0, 0], p[1, 0] = 0.3, 0.7
    r = np.array([0.25, 0.25, 0.01])
    res = compute(dim, p, representation, radii=r)
    assert res.measures[2] == 0
    assert res.cell(2).n_vertices == 0 and len(res.cell(2).facets) == 0
    assert res.measures.sum() == pytest.approx(1.0, abs=1e-14)
    for i in (0, 1):
        assert 2 not in res.cell(i).neighbors


def test_largest_particle_eventually_fills_the_domain():
    """With radii r_i + t, the power distance contains -2 t r_i, so for large
    t the particle with the largest radius takes over the whole domain."""
    p = random_points(2, 30)
    r = 0.01 * np.random.default_rng(5).permutation(30)
    empty = []
    for t in (0.0, 1.0, 10.0, 1000.0):
        res = compute(2, p, radii=r + t)
        empty.append(int(np.sum(res.measures == 0)))
    assert empty == sorted(empty)
    assert res.measures[np.argmax(r)] == pytest.approx(1.0, abs=1e-12)
    assert empty[-1] == 29


@pytest.mark.parametrize("dim", [2, 3])
def test_generator_outside_its_cell(dim):
    """A generator inside the large neighbor's disk lies outside its own cell;
    the signed fan simplices still give the right measure and integrals."""
    torch = pytest.importorskip("torch")
    p = np.full((2, dim), 0.5)
    p[0, 0], p[1, 0] = 0.2, 0.45
    r = np.array([0.3, 0.0])
    res = compute(dim, p, "per_cell", radii=r)
    # The cell boundary is the plane x = x0, with
    # (x0 - 0.2)^2 - 0.09 = (x0 - 0.45)^2, so x0 = 0.505.
    assert res.measures[1] == pytest.approx(1.0 - 0.505, rel=1e-12)
    assert np.all(res.cell(1).vertices[:, 0] >= 0.505 - 1e-12)
    pts = torch.tensor(p)
    v = voroplex.cell_vertices(res, pts)
    owners, simplices = voroplex.cell_fan_simplices(res)
    m = cell_measures(pts, v, owners, simplices, 2).numpy()
    assert np.allclose(m, res.measures, rtol=1e-12)
    # Second moment of cell 1 about its generator, compared with the exact
    # integral of (x - 0.45)^2 + |y - 0.5|^2 over [0.505, 1] x [0, 1]^(d-1).
    e1 = cvt_energy(pts, v, owners[owners == 1], simplices[owners == 1]).item()
    a, b = 0.505 - 0.45, 1.0 - 0.45
    exact = (b ** 3 - a ** 3) / 3 + (dim - 1) * (1 - 0.505) / 12
    assert e1 == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_power_energy_gradient_closed_form(dim, periodic):
    pytest.importorskip("torch")
    p = random_points(dim, 1500, seed=2024)
    r = random_radii(1500, scale=0.05)
    e, gp, gr, res = power_energy_and_gradients(dim, p, r, periodic)
    assert np.any(res.measures == 0)
    assert rel_err(gp, closed_form_cvt_gradient(p, res)) < 1e-9
    assert rel_err(gr, -2.0 * r * res.measures) < 1e-9


@pytest.mark.parametrize("dim", [2, 3])
def test_cvt_energy_gradient_finite_differences(dim):
    """The usual CVT energy, whose gradient has no simple closed form in a
    radical tessellation, compared with finite differences."""
    torch = pytest.importorskip("torch")
    n = 400
    p = random_points(dim, n, seed=8)
    r = random_radii(n, scale=0.04)

    def energy(q, s, grad=False):
        res = compute(dim, q, "per_cell", radii=s)
        pts = torch.tensor(q, requires_grad=grad)
        rad = torch.tensor(s, requires_grad=grad)
        v = voroplex.cell_vertices(res, pts, rad)
        e = cvt_energy(pts, v, *voroplex.cell_fan_simplices(res))
        if grad:
            e.backward()
            return pts.grad.numpy(), rad.grad.numpy()
        return e.item()

    gp, gr = energy(p, r, grad=True)
    h = 1e-6
    for i in range(0, n, 80):
        k = i % dim
        q1, q2 = p.copy(), p.copy()
        q1[i, k] += h
        q2[i, k] -= h
        fd = (energy(q1, r) - energy(q2, r)) / (2 * h)
        assert fd == pytest.approx(gp[i, k], rel=1e-5, abs=1e-9)
        s1, s2 = r.copy(), r.copy()
        s1[i] += h
        s2[i] -= h
        fd = (energy(p, s1) - energy(p, s2)) / (2 * h)
        assert fd == pytest.approx(gr[i], rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_topology_matches_per_cell(dim):
    """Vertices and gradients from the topology reconstruction agree with the
    per-cell vertices."""
    torch = pytest.importorskip("torch")
    p = random_points(dim, 800)
    r = random_radii(800, scale=0.05)
    per = compute(dim, p, "per_cell", threads=1, radii=r)
    top = compute(dim, p, "topology", threads=1, radii=r)
    positions = (voroplex.compute_padded_cell_vertices if dim == 3
                 else voroplex.compute_padded_cell_vertices)
    grads = []
    for res in (per, top):
        pts = torch.tensor(p, requires_grad=True)
        rad = torch.tensor(r, requires_grad=True)
        if res is per:
            v = voroplex.cell_vertices(res, pts, rad)
        else:
            padded, counts = positions(res, pts, radii=rad)
            mask = torch.arange(padded.shape[1]) < counts.unsqueeze(1)
            v = padded[mask]
            assert np.abs(v.detach().numpy() - per.vertices).max() < 1e-12
        (v ** 2).sum().backward()
        grads.append((pts.grad.numpy(), rad.grad.numpy()))
    assert rel_err(grads[1][0], grads[0][0]) < 1e-8
    assert rel_err(grads[1][1], grads[0][1]) < 1e-8


def test_stored_radii_are_used_as_constants():
    torch = pytest.importorskip("torch")
    p = random_points(2, 300)
    r = random_radii(300, scale=0.05)
    res = compute(2, p, "per_cell", radii=r)
    g = []
    for radii in (None, torch.tensor(r)):
        pts = torch.tensor(p, requires_grad=True)
        voroplex.cell_vertices(res, pts, radii).square().sum().backward()
        g.append(pts.grad.numpy())
    assert rel_err(g[0], g[1]) < 1e-12


def test_radii_that_differ_from_the_tessellation_raise():
    torch = pytest.importorskip("torch")
    p = random_points(2, 300)
    r = random_radii(300, scale=0.05)
    res = compute(2, p, "per_cell", radii=r)
    pts = torch.tensor(p, requires_grad=True)
    rad = torch.tensor(r + 0.01, requires_grad=True)
    with pytest.raises(RuntimeError, match="differ from those used"):
        voroplex.cell_vertices(res, pts, rad).sum().backward()


def test_input_checks():
    torch = pytest.importorskip("torch")
    p = random_points(2, 50)
    for bad, match in ((np.ones(49), "shape"), (np.full(50, np.nan), "finite"),
                       (-np.ones(50), "non-negative")):
        with pytest.raises(ValueError, match=match):
            compute(2, p, radii=bad)
    res = compute(2, p, "per_cell")
    with pytest.raises(ValueError, match="without radii"):
        voroplex.cell_vertices(res, torch.tensor(p), torch.ones(50, dtype=torch.float64))
    res = compute(2, p, "per_cell", radii=np.ones(50))
    with pytest.raises(ValueError, match="shape"):
        voroplex.cell_vertices(res, torch.tensor(p), torch.ones(49, dtype=torch.float64))


@pytest.mark.parametrize("dim", [2, 3])
def test_coincident_generators_raise_even_with_different_radii(dim):
    p = random_points(dim, 200)
    p[7] = p[3]
    r = random_radii(200)
    with pytest.raises(ValueError, match="identical positions.*3 and 7"):
        compute(dim, p, radii=r)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_npz_round_trip(representation, tmp_path):
    p = random_points(3, 300)
    res = compute(3, p, representation, radii=random_radii(300, scale=0.05))
    path = tmp_path / "result.npz"
    voroplex.export_npz(res, str(path))
    back = voroplex.load_npz(str(path))
    assert repr(back) == repr(res) and "radical=True" in repr(res)
    assert np.array_equal(back.radii, res.radii)
    assert np.array_equal(back.measures, res.measures)


@pytest.mark.parametrize("dim", [2, 3])
def test_surface_mesh_with_empty_cells(dim):
    """The shared-vertex meshes work with empty cells."""
    pytest.importorskip("torch")
    p = random_points(dim, 1000)
    r = random_radii(1000, scale=0.05)
    inner = np.linalg.norm(p - 0.5, axis=1) < 0.3
    order = np.argsort(~inner, kind="stable")
    res = compute(dim, p[order], radii=r[order], n_inner=int(inner.sum()))
    assert np.any(res.measures[:int(inner.sum())] == 0)
    if dim == 3:
        mesh = voroplex.surface_mesh(res)
        assert mesh["facet_vertex_ids"].shape[0] > 0
    else:
        mesh = voroplex.surface_mesh(res)
        assert len(mesh) > 0


def test_polygon_analysis_and_vtu_export_with_empty_cells(tmp_path):
    torch = pytest.importorskip("torch")
    from voroplex import analysis
    p = random_points(2, 400)
    r = random_radii(400, scale=0.08)
    top = compute(2, p, "topology", radii=r)
    assert np.any(top.measures == 0)
    vertices = voroplex.compute_unique_vertices(top, torch.tensor(p))
    polygons, ids = voroplex.cell_polytopes(top)
    width = max(len(c) for c in polygons)
    vids = torch.full((len(polygons), width), -1, dtype=torch.int64)
    for i, c in enumerate(polygons):
        vids[i, :len(c)] = torch.tensor(c, dtype=torch.int64)
    counts = torch.tensor([len(c) for c in polygons])
    info = analysis.compute_polygon_geometry(vertices, vids, counts,
                                                         vids >= 0)
    areas = np.zeros(len(p))
    areas[np.asarray(ids)] = info["cell_areas"].numpy()
    assert np.allclose(areas, top.areas, rtol=1e-8, atol=1e-15)
    pytest.importorskip("vtk")
    for dim in (2, 3):
        q = random_points(dim, 300)
        res = compute(dim, q, radii=random_radii(300, scale=0.08), n_inner=150)
        assert np.any(res.measures == 0)
        path = tmp_path / "mesh{}.vtu".format(dim)
        voroplex.export_vtu(res, torch.tensor(q), str(path))
        assert path.stat().st_size > 0


@requires_cuda
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("backward", ["cpu", "torch"])
def test_cuda_matches_cpu(dim, backward, monkeypatch):
    import torch
    from voroplex._mesh import per_cell
    monkeypatch.setattr(per_cell, "_BACKWARD", backward)
    p = random_points(dim, 1000)
    r = random_radii(1000, scale=0.05)
    res = compute(dim, p, "per_cell", radii=r)
    out = []
    for device in ("cpu", "cuda"):
        pts = torch.tensor(p, device=device, requires_grad=True)
        rad = torch.tensor(r, device=device, requires_grad=True)
        v = voroplex.cell_vertices(res, pts, rad)
        owners, simplices = voroplex.cell_fan_simplices(res, device=device)
        cvt_energy(pts, v, owners, simplices).backward()
        out.append((pts.grad.cpu().numpy(), rad.grad.cpu().numpy()))
    assert rel_err(out[1][0], out[0][0]) < 1e-9
    assert rel_err(out[1][1], out[0][1]) < 1e-9
