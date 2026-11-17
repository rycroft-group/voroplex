"""Shared helpers for the VoroPlex tests.

Run the tests from the repository root with ``python -m pytest tests``, after
building the extension in place (``python setup.py build_ext --inplace``).
Tests that need a GPU are skipped when CUDA is not available.
"""
import numpy as np
import pytest

import voroplex

try:
    import torch
except ImportError:  # pragma: no cover - PyTorch is optional for VoroPlex
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="requires PyTorch")
requires_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="requires CUDA"
)


def compute(dim, points, representation="geometry", limits=None, **kwargs):
    """compute_voronoi_3d or compute_voronoi_2d, in the unit box by default."""
    fn = voroplex.compute_voronoi_3d if dim == 3 else voroplex.compute_voronoi_2d
    if limits is None:
        limits = [[0.0, 1.0]] * dim
    return fn(points, limits, representation=representation, **kwargs)


def random_points(dim, n, seed=12345):
    return np.random.default_rng(seed).random((n, dim))


def lattice(dim, m, eps=0.0, seed=7):
    """An m^dim lattice of cell centers in the unit box, perturbed by eps."""
    g = (np.arange(m) + 0.5) / m
    p = np.stack(np.meshgrid(*([g] * dim), indexing="ij"), -1).reshape(-1, dim)
    return p + eps * np.random.default_rng(seed).standard_normal(p.shape)


def cvt_energy(points, vertices, owners, simplices):
    """Uniform-density CVT energy, sum_i int_{V_i} |x - p_i|^2 dx.

    Each simplex has the generator points[owners] as one corner, and the
    vertices[simplices] as the others. With q_k the other corners relative to
    the generator, the integral over a triangle is (A/6)(|q1|^2 + |q2|^2 +
    q1.q2), and over a tetrahedron (V/10)(sum |q_k|^2 + sum_{j<k} q_j.q_k),
    where A and V are signed, so that this also holds for a generator outside
    its own cell (in a radical tessellation).
    """
    q = vertices[simplices] - points[owners].unsqueeze(1)
    if q.shape[2] == 2:
        q1, q2 = q[:, 0], q[:, 1]
        area = 0.5 * (q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0])
        quad = (q1 * q1).sum(1) + (q2 * q2).sum(1) + (q1 * q2).sum(1)
        return (area * quad / 6.0).sum()
    q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
    vol = (q1 * torch.cross(q2, q3, dim=1)).sum(1) / 6.0
    quad = ((q1 * q1).sum(1) + (q2 * q2).sum(1) + (q3 * q3).sum(1)
            + (q1 * q2).sum(1) + (q1 * q3).sum(1) + (q2 * q3).sum(1))
    return (vol * quad / 10.0).sum()


def closed_form_cvt_gradient(points, result):
    """2 m_i (p_i - c_i), from the per-cell geometry without autodiff."""
    dim = result.dimension
    owners, simplices = voroplex.cell_fan_simplices(result)
    owners, simplices = owners.numpy(), simplices.numpy()
    g = points[owners]
    corners = result.vertices[simplices]
    q = corners - g[:, None, :]
    if dim == 2:
        vol = 0.5 * (q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0])
    else:
        vol = np.einsum("si,si->s", q[:, 0], np.cross(q[:, 1], q[:, 2])) / 6.0
    centroid = (g + corners.sum(axis=1)) / (dim + 1)
    n = len(points)
    m = np.bincount(owners, vol, minlength=n)
    mc = np.stack(
        [np.bincount(owners, vol * centroid[:, k], minlength=n) for k in range(dim)], 1
    )
    return 2.0 * (m[:, None] * points - mc)


def rel_err(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
