"""Tests of saving and loading results, and of the VTU export."""
import numpy as np
import pytest

import voroplex
from conftest import compute, random_points

ARRAYS = ("measures", "cell_vertex_offsets", "cell_facet_offsets",
          "facet_vertex_offsets", "facet_vertices")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kwargs", [dict(), dict(representation="per_cell"),
                                    dict(representation="topology"),
                                    dict(neighbors=False),
                                    dict(representation="per_cell", periodic=True)])
def test_npz_round_trip(dim, kwargs, tmp_path):
    res = compute(dim, random_points(dim, 300), n_compute=250, **kwargs)
    path = tmp_path / "result.npz"
    voroplex.export_npz(res, str(path))
    back = voroplex.load_npz(str(path))
    assert repr(back) == repr(res)
    for name in ARRAYS:
        assert np.array_equal(getattr(back, name), getattr(res, name)), name
    if res.has_vertices:
        assert np.array_equal(back.vertices, res.vertices)
    if res.has_signatures:
        assert np.array_equal(back.vertex_signature_ids, res.vertex_signature_ids)
    if res.has_neighbors:
        assert np.array_equal(back.facet_neighbors, res.facet_neighbors)
        assert np.array_equal(back.cell_surface_flags, res.cell_surface_flags)


def test_npz_reads_version_1_archives(tmp_path):
    """Archives written before neighbors=False existed can still be read."""
    res = compute(2, random_points(2, 100))
    path = tmp_path / "result.npz"
    voroplex.export_npz(res, str(path))
    with np.load(str(path)) as data:
        arrays = dict(data)
    arrays["schema_version"] = np.asarray(1, dtype=np.int32)
    np.savez_compressed(str(path), **arrays)
    assert np.array_equal(voroplex.load_npz(str(path)).measures, res.measures)


def test_npz_reads_archives_with_earlier_names(tmp_path):
    """Archives that use the earlier names cell_boundary_flags and num_* load."""
    res = compute(3, random_points(3, 100), n_inner=60,
                  properties=("volume", "n_faces"))
    path = tmp_path / "result.npz"
    voroplex.export_npz(res, str(path))
    with np.load(str(path)) as data:
        arrays = dict(data)
    arrays["cell_boundary_flags"] = arrays.pop("cell_surface_flags")
    arrays["property_num_faces"] = arrays.pop("property_n_faces")
    np.savez_compressed(str(path), **arrays)
    back = voroplex.load_npz(str(path))
    assert np.array_equal(back.cell_surface_flags, res.cell_surface_flags)
    assert np.array_equal(back.n_faces, res.n_faces)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", ["geometry", "per_cell", "topology"])
def test_vtu_export(dim, representation, tmp_path):
    pytest.importorskip("vtk")
    pytest.importorskip("torch")
    import torch
    p = random_points(dim, 200)
    res = compute(dim, p, representation, n_inner=100)
    path = tmp_path / "mesh.vtu"
    voroplex.export_vtu(res, torch.tensor(p), str(path))
    assert path.stat().st_size > 0
