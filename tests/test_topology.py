"""Tests of the topology representation and the shared-vertex mesh utilities."""
import numpy as np
import pytest

import voroplex
from conftest import compute, random_points

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (1e-6, 1.0), (1e3, -5e3)])
def test_reconstruction_matches_voro_vertices(dim, scale, shift):
    """Vertices recomputed from the signatures equal Voro++'s coordinates,
    including for small cells far from the origin."""
    p = random_points(dim, 1000) * scale + shift
    limits = [[shift, shift + scale]] * dim
    geo = compute(dim, p, limits=limits, threads=1)
    top = compute(dim, p, "topology", limits=limits, threads=1)
    get = (voroplex.compute_padded_cell_vertices if dim == 3
           else voroplex.compute_padded_cell_vertices)
    padded, counts = get(top, torch.tensor(p))
    assert np.array_equal(counts.numpy(), np.diff(geo.cell_vertex_offsets))
    cells = np.repeat(np.arange(len(geo)), counts.numpy())
    local = np.arange(geo.vertices.shape[0]) - geo.cell_vertex_offsets[cells]
    assert np.abs(padded[cells, local].numpy() - geo.vertices).max() < 1e-12 * scale


@pytest.mark.parametrize("dim", [2, 3])
def test_reconstruction_gradient_matches_per_cell(dim):
    """Away from degenerate vertices, the two differentiable paths agree."""
    p = random_points(dim, 800)
    pts = torch.tensor(p, requires_grad=True)
    v = voroplex.cell_vertices(compute(dim, p, "per_cell"), pts)
    v.square().sum().backward()
    g_cell = pts.grad.clone()

    top = compute(dim, p, "topology")
    get = (voroplex.compute_padded_cell_vertices if dim == 3
           else voroplex.compute_padded_cell_vertices)
    pts.grad = None
    padded, counts = get(top, pts)
    mask = torch.arange(padded.shape[1]) < counts.unsqueeze(1)
    (padded[mask] ** 2).sum().backward()
    assert torch.allclose(pts.grad, g_cell, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_reconstruction_requires_topology(dim):
    rec = voroplex.compute_unique_vertices if dim == 3 else voroplex.compute_unique_vertices
    with pytest.raises(ValueError, match="topology"):
        rec(compute(dim, random_points(dim, 50)), torch.tensor(random_points(dim, 50)))


@pytest.mark.parametrize("dim", [2, 3])
def test_unique_facets(dim):
    """Each interior facet appears once; each facet on the boundary once."""
    p = random_points(dim, 600)
    res = compute(dim, p, threads=1)
    extract = (voroplex.unique_facets if dim == 3
               else lambda r, facet_type: voroplex.unique_facets(r, facet_type=facet_type))
    facets, info = extract(res, facet_type="all")
    n_boundary = int((res.facet_neighbors < 0).sum())
    n_interior = int((res.facet_neighbors >= 0).sum())
    assert len(facets) == n_boundary + n_interior // 2
    assert np.all(info[:, 0] >= 0)


@pytest.mark.parametrize("dim", [2, 3])
def test_interface_mesh_geometry_and_topology_agree(dim):
    """The shared-vertex interface mesh works with geometry results too, and
    agrees with the topology version."""
    p = random_points(dim, 600)
    geo = compute(dim, p, n_inner=300)
    top = compute(dim, p, "topology", n_inner=300)
    if dim == 2:
        mg = voroplex.surface_mesh(geo)
        mt = voroplex.surface_mesh(top, torch.tensor(p))
        key_ids, key_xyz = "facet_vertex_ids", "vertex_coords"
    else:
        mg = voroplex.surface_mesh(geo)
        mt = voroplex.surface_mesh(top, torch.tensor(p))
        key_ids, key_xyz = "facet_vertex_ids", "vertex_coords"
    # The two paths may number the vertices differently, so compare each
    # edge or face as the set of its vertex coordinates.
    def facets(mesh):
        xyz = np.round(mesh[key_xyz].numpy(), 9)
        out = []
        for row in mesh[key_ids].numpy():
            ids = row[row >= 0]
            out.append(tuple(sorted(map(tuple, xyz[ids]))))
        return sorted(out)

    assert len(mg[key_ids]) == len(mt[key_ids])
    assert facets(mg) == facets(mt)


def test_topology_interface_mesh_is_differentiable():
    p = random_points(2, 400)
    top = compute(2, p, "topology", n_inner=200)
    pts = torch.tensor(p, requires_grad=True)
    mesh = voroplex.surface_mesh(top, pts)
    mesh["vertex_coords"].sum().backward()
    assert pts.grad.abs().sum() > 0


def test_geometry_mesh_needs_neighbors_and_no_periodicity():
    p = random_points(2, 200)
    with pytest.raises(AttributeError, match="neighbors=False"):
        voroplex.surface_mesh(compute(2, p, neighbors=False))
    for representation in ("geometry", "per_cell"):
        periodic = compute(2, p, representation, n_inner=100, periodic=True)
        with pytest.raises(NotImplementedError, match="periodic"):
            voroplex.surface_mesh(periodic)
        with pytest.raises(NotImplementedError, match="periodic"):
            voroplex.unique_facets(periodic, facet_type="all")


def test_polyhedral_cells_are_closed():
    """Every edge of every polyhedron is shared by exactly two of its faces."""
    res = compute(3, random_points(3, 300), threads=1)
    cells, ids = voroplex.cell_polytopes(res)
    assert len(cells) == len(ids) == 300
    for faces in cells[:50]:
        edges = {}
        for face in faces:
            for a, b in zip(face, face[1:] + face[:1]):
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0) + 1
        assert set(edges.values()) == {2}
