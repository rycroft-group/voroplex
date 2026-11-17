"""Tests of the per-cell differentiable vertices (cell_vertices and
cell_fan_simplices).

The gradient of the uniform-density CVT energy is compared with its closed
form, 2 m_i (p_i - c_i), and with finite differences, in boxes and periodic
domains, and near lattices where the vertex systems are nearly degenerate.
"""
import numpy as np
import pytest

import voroplex
from conftest import (closed_form_cvt_gradient, compute, cvt_energy, lattice,
                      random_points, rel_err, requires_cuda, requires_torch)

torch = pytest.importorskip("torch")


def energy_and_gradient(dim, p, periodic=False, device="cpu", dtype=torch.float64):
    res = compute(dim, p, "per_cell", periodic=periodic)
    pts = torch.tensor(p, dtype=dtype, device=device, requires_grad=True)
    v = voroplex.cell_vertices(res, pts)
    owners, simplices = voroplex.cell_fan_simplices(res, device=device)
    e = cvt_energy(pts, v, owners, simplices)
    e.backward()
    return e.item(), pts.grad.detach().cpu().numpy(), res


def finite_difference(dim, p, entries, h, periodic=False):
    out = []
    for i, k in entries:
        values = []
        for s in (1, -1):
            q = p.copy()
            q[i, k] += s * h
            res = compute(dim, q, "per_cell", periodic=periodic)
            owners, simplices = voroplex.cell_fan_simplices(res)
            values.append(cvt_energy(torch.tensor(q), torch.tensor(res.vertices),
                                     owners, simplices).item())
        out.append((values[0] - values[1]) / (2 * h))
    return np.array(out)


@pytest.mark.parametrize("dim", [2, 3])
def test_vertices_are_the_voro_vertices(dim):
    p = random_points(dim, 500)
    res = compute(dim, p, "per_cell")
    v = voroplex.cell_vertices(res, torch.tensor(p))
    assert np.array_equal(v.detach().numpy(), res.vertices)


@pytest.mark.parametrize("dim", [2, 3])
def test_energy_matches_geometry(dim):
    """The energy from the simplices equals sum m_i |p_i - c_i|^2 + const."""
    p = random_points(dim, 800)
    e, g, res = energy_and_gradient(dim, p)
    assert e > 0
    assert rel_err(g, closed_form_cvt_gradient(p, res)) < 1e-9


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [False, True, "partial"])
def test_gradient_closed_form_and_finite_differences(dim, periodic):
    if periodic == "partial":
        periodic = [True] + [False] * (dim - 1)
    n = 1500
    p = random_points(dim, n, seed=2024)
    # Some points close to the boundary, where periodic images matter.
    rng = np.random.default_rng(3)
    k = n // 20
    p[np.arange(k), rng.integers(0, dim, k)] = rng.choice([2e-3, 1 - 2e-3], k)
    e, g, res = energy_and_gradient(dim, p, periodic)
    assert rel_err(g, closed_form_cvt_gradient(p, res)) < 1e-9
    entries = [(int(i), int(j)) for i, j in zip(range(0, k, max(1, k // 5)),
                                               rng.integers(0, dim, 6))][:5]
    h = 1e-5 * n ** (-1.0 / dim)
    fd = finite_difference(dim, p, entries, h, periodic)
    # Finite differences are limited by rounding error; a mistake in the
    # gradient would give an error of order one.
    assert rel_err(g[tuple(np.array(entries).T)], fd) < 1e-4


@pytest.mark.parametrize("dim,m", [(2, 20), (3, 8)])
@pytest.mark.parametrize("eps", [0.0, 1e-9, 1e-3])
@pytest.mark.parametrize("periodic", [False, True])
def test_gradient_near_lattices(dim, m, eps, periodic):
    """Near a lattice, many vertices are nearly degenerate. Errors are relative
    to the natural size of dE/dp, since the gradient itself is nearly zero."""
    p = lattice(dim, m, eps)
    if periodic:
        p = np.mod(p, 1.0)
    e, g, res = energy_and_gradient(dim, p, periodic)
    err = np.abs(g - closed_form_cvt_gradient(p, res)).max()
    assert err / (1.0 / m) ** (dim + 1) < 1e-6


def test_translation_invariance_in_a_periodic_box():
    p = random_points(3, 1000)
    e1, g1, _ = energy_and_gradient(3, p, True)
    e2, g2, _ = energy_and_gradient(3, np.mod(p + [0.31, 0.72, 0.15], 1.0), True)
    assert e2 == pytest.approx(e1, rel=1e-12)
    assert rel_err(g2, g1) < 1e-9
    assert np.abs(g1.sum(axis=0)).max() < 1e-12 * np.abs(g1).sum(axis=0).max()


@pytest.mark.parametrize("dim,n", [(2, 3), (3, 4), (2, 8), (3, 12)])
def test_tiny_periodic_boxes(dim, n):
    """A tiny periodic system gives a correct gradient or a clear error."""
    p = random_points(dim, n, seed=11)
    try:
        e, g, res = energy_and_gradient(dim, p, True)
    except RuntimeError as exc:
        assert "box is too small" in str(exc)
        return
    assert rel_err(g, closed_form_cvt_gradient(p, res)) < 1e-9


@pytest.mark.parametrize("dim", [2, 3])
def test_fan_simplices_match_reference(dim):
    from voroplex._mesh.per_cell import _cell_fan_simplices_numpy
    res = compute(dim, random_points(dim, 600), "per_cell")
    o1, s1 = voroplex.cell_fan_simplices(res)
    o0, s0 = _cell_fan_simplices_numpy(res)
    assert torch.equal(o1, o0) and torch.equal(s1, s0)


def test_n_compute_gradient_reaches_uncomputed_generators():
    """Cells computed for the first generators depend on the others too."""
    p = random_points(2, 300)
    res = compute(2, p, "per_cell", n_compute=100)
    pts = torch.tensor(p, requires_grad=True)
    voroplex.cell_vertices(res, pts).square().sum().backward()
    assert pts.grad[100:].abs().sum() > 0


def test_float32_points_used_consistently():
    p = random_points(2, 500).astype(np.float32)
    e, g, res = energy_and_gradient(2, p, dtype=torch.float32)
    assert g.dtype == np.float32
    assert rel_err(g, closed_form_cvt_gradient(p.astype(np.float64), res)) < 1e-4


def test_points_that_differ_from_the_tessellation_raise():
    p = random_points(2, 300)
    res = compute(2, p, "per_cell")
    pts = torch.tensor(p + 0.01, requires_grad=True)
    with pytest.raises(RuntimeError, match="differ from those used"):
        voroplex.cell_vertices(res, pts).sum().backward()


def test_second_derivatives_raise():
    p = random_points(2, 200)
    res = compute(2, p, "per_cell")
    pts = torch.tensor(p, requires_grad=True)
    e = voroplex.cell_vertices(res, pts).square().sum() + (pts ** 3).sum()
    with pytest.raises(RuntimeError, match="second derivatives"):
        torch.autograd.grad(e, pts, create_graph=True)


def test_input_checks():
    res = compute(2, random_points(2, 50), "per_cell")
    with pytest.raises(TypeError):
        voroplex.cell_vertices(res, random_points(2, 50))
    with pytest.raises(ValueError, match="shape"):
        voroplex.cell_vertices(res, torch.zeros(49, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="per_cell"):
        voroplex.cell_vertices(compute(2, random_points(2, 50)), torch.zeros(50, 2))


@requires_cuda
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("backward", ["cpu", "torch"])
def test_cuda_matches_cpu(dim, backward, monkeypatch):
    from voroplex._mesh import per_cell
    monkeypatch.setattr(per_cell, "_BACKWARD", backward)
    p = random_points(dim, 1000)
    e_cpu, g_cpu, _ = energy_and_gradient(dim, p)
    e_gpu, g_gpu, _ = energy_and_gradient(dim, p, device="cuda")
    assert e_gpu == pytest.approx(e_cpu, rel=1e-12)
    assert rel_err(g_gpu, g_cpu) < 1e-9
