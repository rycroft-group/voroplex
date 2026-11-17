"""Tests of the per-cell properties (the properties argument), and of results
computed with representation=None, which store only those properties.

Each property is compared with an independent calculation from the cell
vertices and faces of a geometry or per-cell result.
"""
import numpy as np
import pytest

import voroplex
from conftest import compute, random_points

ALL = {
    2: ("area", "perimeter", "centroid", "n_vertices", "max_radius",
        "second_moment", "second_moment_tensor"),
    3: ("volume", "surface_area", "centroid", "n_faces", "n_vertices",
        "n_edges", "total_edge_length", "max_radius", "second_moment",
        "second_moment_tensor"),
}


def reference_properties(p, res):
    """The properties of each cell, from its vertices and faces."""
    dim = res.dimension
    out = {name: [] for name in ALL[dim]}
    for i in range(len(res)):
        cell = res.cell(i)
        v = cell.vertices - p[i]          # vertices relative to the generator
        faces = [f.vertex_ids for f in cell.facets]
        size, mom, tensor, boundary, edges = 0.0, np.zeros(dim), np.zeros((dim, dim)), 0.0, 0.0
        for f in faces:
            if dim == 2:
                a, b = v[f[0]], v[f[1]]
                d = a[0] * b[1] - a[1] * b[0]
                s = a + b
                size += d / 2
                mom += d / 6 * s
                tensor += d / 24 * (np.outer(a, a) + np.outer(b, b) + np.outer(s, s))
                boundary += np.linalg.norm(b - a)
            else:
                for k in range(1, len(f) - 1):
                    a, b, c = v[f[0]], v[f[k + 1]], v[f[k]]
                    d = np.dot(a, np.cross(b, c))
                    s = a + b + c
                    size += d / 6
                    mom += d / 24 * s
                    tensor += d / 120 * (np.outer(a, a) + np.outer(b, b)
                                         + np.outer(c, c) + np.outer(s, s))
                q = v[np.r_[f, f[:1]]]
                boundary += 0.5 * np.linalg.norm(
                    sum(np.cross(q[k] - q[0], q[k + 1] - q[0]) for k in range(1, len(f) - 1)))
                edges += np.linalg.norm(np.diff(q, axis=0), axis=1).sum() / 2
        out["area" if dim == 2 else "volume"].append(size)
        out["perimeter" if dim == 2 else "surface_area"].append(boundary)
        out["centroid"].append(p[i] + mom / size)
        out["n_vertices"].append(len(v))
        out["max_radius"].append(np.linalg.norm(v, axis=1).max())
        out["second_moment"].append(np.trace(tensor))
        out["second_moment_tensor"].append(tensor)
        if dim == 3:
            out["n_faces"].append(len(faces))
            out["n_edges"].append(sum(len(f) for f in faces) // 2)
            out["total_edge_length"].append(edges)
    return {k: np.array(x) for k, x in out.items()}


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_properties_match_the_cell_geometry(dim, periodic):
    p = random_points(dim, 300)
    res = compute(dim, p, "per_cell", periodic=periodic, properties=ALL[dim])
    ref = reference_properties(p, res)
    for name in ALL[dim]:
        got = res.properties[name]
        scale = np.abs(ref[name]).max()
        assert np.abs(got - ref[name]).max() <= 1e-12 * scale, name
    assert res.measures.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_second_moments_of_box_cells(dim):
    """A lattice of generators at the centers of boxes of side h gives box
    cells, whose second moment tensor is h^(d+2)/12 times the identity."""
    m = 5
    g = (np.arange(m) + 0.5) / m
    p = np.stack(np.meshgrid(*([g] * dim), indexing="ij"), -1).reshape(-1, dim)
    res = compute(dim, p, None, properties=("second_moment_tensor", "centroid"))
    h = 1.0 / m
    exact = h ** (dim + 2) / 12 * np.eye(dim)
    assert np.allclose(res.second_moment_tensors, exact, rtol=0, atol=1e-14 * h ** dim)
    assert np.allclose(res.centroids, p, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", ["geometry", "per_cell", "topology"])
@pytest.mark.parametrize("threads", [1, 6])
def test_representation_none_gives_the_same_properties(dim, representation, threads):
    p = random_points(dim, 2000)
    full = compute(dim, p, representation, threads=threads, properties=ALL[dim])
    only = compute(dim, p, None, threads=threads, properties=ALL[dim])
    assert len(only) == len(full) == 2000
    for name in ALL[dim]:
        assert np.allclose(only.properties[name], full.properties[name],
                           rtol=1e-13, atol=1e-15), name


@pytest.mark.parametrize("dim", [2, 3])
def test_representation_none_stores_only_the_properties(dim):
    p = random_points(dim, 500)
    res = compute(dim, p, None, properties="centroid", n_compute=400)
    assert res.centroids.shape == (400, dim) and len(res) == 400
    assert not res.has_cells and not res.has_vertices and not res.has_neighbors
    assert "representation=None" in repr(res)
    for get in (lambda r: r.vertices, lambda r: r.cell_vertex_offsets,
                lambda r: r.facet_neighbors, lambda r: r.surface_cells,
                lambda r: r.cell(0).vertices, lambda r: r.measures):
        with pytest.raises(AttributeError):
            get(res)
    with pytest.raises(ValueError, match="representation=None"):
        (voroplex.unique_facets if dim == 3
         else voroplex.unique_facets)(res)
    with pytest.raises(ValueError, match="n_inner"):
        compute(dim, p, None, n_inner=100)


def test_default_and_invalid_properties():
    p2, p3 = random_points(2, 100), random_points(3, 100)
    assert tuple(compute(2, p2).properties) == ("area",)
    assert tuple(compute(3, p3).properties) == ("volume",)
    res = compute(3, p3, properties=())
    assert res.properties == {}
    with pytest.raises(AttributeError, match="volumes were not computed"):
        res.volumes
    with pytest.raises(AttributeError, match="2D results have areas"):
        compute(2, p2).volumes
    with pytest.raises(ValueError, match="'surface_area' is a 3D property"):
        compute(2, p2, properties=("surface_area",))
    with pytest.raises(ValueError, match="unknown 3D property"):
        compute(3, p3, properties=("volume", "colour"))
    # A repeated name is computed once.
    assert tuple(compute(3, p3, properties=("centroid", "centroid")).properties) == ("centroid",)


@pytest.mark.parametrize("dim", [2, 3])
def test_empty_radical_cells(dim):
    p = np.full((3, dim), 0.5)
    p[0, 0], p[1, 0] = 0.3, 0.7
    r = np.array([0.25, 0.25, 0.01])
    for rep in ("geometry", None):
        res = compute(dim, p, rep, radii=r, properties=ALL[dim])
        assert res.measures[2] == 0 and res.n_vertices[2] == 0
        assert np.isnan(res.centroids[2]).all() and np.isnan(res.max_radii[2])
        assert res.second_moments[2] == 0
        assert np.isfinite(res.centroids[:2]).all()


def test_generator_outside_its_cell():
    """The moments use signed volumes, so they are correct for a generator
    outside its own cell: here the cell of generator 1 is [0.505, 1] x [0, 1]."""
    p = np.array([[0.2, 0.5], [0.45, 0.5]])
    res = compute(2, p, None, radii=np.array([0.3, 0.0]),
                  properties=("area", "centroid", "second_moment_tensor"))
    a, b = 0.505 - 0.45, 1.0 - 0.45
    assert res.areas[1] == pytest.approx(0.495, rel=1e-12)
    assert res.centroids[1] == pytest.approx([0.7525, 0.5], rel=1e-12)
    xx = (b ** 3 - a ** 3) / 3
    yy = 0.495 / 12
    assert np.allclose(res.second_moment_tensors[1], [[xx, 0], [0, yy]],
                       rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("representation", [None, "geometry", "per_cell"])
def test_npz_round_trip(representation, tmp_path):
    p = random_points(3, 300)
    res = compute(3, p, representation, properties=ALL[3])
    path = tmp_path / "result.npz"
    voroplex.export_npz(res, str(path))
    back = voroplex.load_npz(str(path))
    assert repr(back) == repr(res)
    for name in ALL[3]:
        assert np.array_equal(back.properties[name], res.properties[name])


def test_lloyd_iterations_reduce_the_energy():
    """Lloyd's algorithm, using only the centroids and second moments."""
    p = random_points(2, 400)
    energies = []
    for _ in range(10):
        res = compute(2, p, None, properties=("centroid", "second_moment"))
        energies.append(res.second_moments.sum())
        p = res.centroids
    assert all(b < a for a, b in zip(energies, energies[1:]))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", ["geometry", "per_cell", "topology", None])
def test_compute_properties_for_an_existing_result(dim, representation):
    p = random_points(dim, 1000)
    names = ("centroid", "second_moment", "max_radius")
    for kwargs in (dict(), dict(properties=())):
        if representation == "topology":
            kwargs["periodic"] = False
        res = compute(dim, p, representation, n_compute=800, **kwargs)
        props = voroplex.compute_properties(res, p, names)
        direct = compute(dim, p, None, n_compute=800, properties=names)
        assert tuple(props) == names
        for name in names:
            assert props[name].shape[0] == 800
            assert np.allclose(props[name], direct.properties[name],
                               rtol=1e-13, atol=1e-15), name


def test_compute_properties_uses_the_settings_of_the_result():
    p = random_points(3, 800)
    r = 0.05 * np.random.default_rng(3).random(800)
    res = compute(3, p, periodic=[True, False, True], radii=r)
    props = voroplex.compute_properties(res, p, ["centroid"])
    direct = compute(3, p, None, periodic=[True, False, True], radii=r,
                     properties=["centroid"])
    assert np.array_equal(np.isnan(props["centroid"]), np.isnan(direct.centroids))
    assert np.allclose(props["centroid"], direct.centroids, rtol=1e-13,
                       atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("properties", [None, ()])
def test_compute_properties_rejects_other_points(properties):
    p = random_points(2, 500)
    res = compute(2, p, properties=properties)
    q = random_points(2, 500, seed=99)
    with pytest.raises(ValueError, match="do not match"):
        voroplex.compute_properties(res, q, ["centroid"])
    with pytest.raises(ValueError, match="shape"):
        voroplex.compute_properties(res, p[:, :1], ["centroid"])
    with pytest.raises(ValueError, match="same 500 generators"):
        voroplex.compute_properties(res, p[:400], ["centroid"])
    with pytest.raises(ValueError, match="3D property"):
        voroplex.compute_properties(res, p, ["volume"])


def test_npz_with_points(tmp_path):
    p = random_points(3, 400)
    res = compute(3, p, properties=())
    path = tmp_path / "result.npz"
    voroplex.export_npz(res, str(path), points=p)
    back, points = voroplex.load_npz(str(path), return_points=True)
    assert np.array_equal(points, p)
    assert isinstance(voroplex.load_npz(str(path)), voroplex.VoronoiResult)
    props = voroplex.compute_properties(back, points, ["volume", "centroid"])
    assert props["volume"].sum() == pytest.approx(1.0, abs=1e-12)
    # Without saved points, return_points gives None.
    voroplex.export_npz(res, str(path))
    assert voroplex.load_npz(str(path), return_points=True)[1] is None
    with pytest.raises(ValueError, match="shape"):
        voroplex.export_npz(res, str(path), points=p[:10])
