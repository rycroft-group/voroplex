"""Tests of the surface and 2D cell analysis utilities."""
import numpy as np
import pytest

import voroplex
from conftest import compute, random_points

torch = pytest.importorskip("torch")
from voroplex import analysis  # noqa: E402


def surface_info(res, points=None):
    mesh = voroplex.surface_mesh(res, points)
    info = analysis.compute_surface_geometry(
        mesh["vertex_coords"], mesh["facet_vertex_ids"],
        mesh["n_vertices_per_facet"], mesh["facet_mask"])
    return mesh, info


@pytest.mark.parametrize("representation", ["geometry", "topology"])
def test_surface_normals_point_out_of_the_inner_region(representation):
    p = random_points(3, 300, seed=1)
    p[0] = [0.5, 0.5, 0.5]
    res = compute(3, p, representation, n_inner=1)
    mesh, info = surface_info(res, torch.tensor(p))
    outward = ((info["face_centers"] - torch.tensor(p[0])) * info["face_normals"]).sum(1)
    assert bool((outward > 0).all())
    assert torch.allclose(info["face_normals"].norm(dim=1),
                          torch.ones(len(outward), dtype=torch.float64))


def test_surface_areas_match_cell_volume():
    """For a single inner cell, the divergence theorem gives its volume."""
    p = random_points(3, 300, seed=1)
    p[0] = [0.5, 0.5, 0.5]
    res = compute(3, p, n_inner=1)
    mesh, info = surface_info(res)
    # V = (1/3) sum over faces of A (x_f . n) for any point x_f on the face.
    volume = (info["face_areas"] * ((info["face_centers"] - torch.tensor(p[0]))
                                    * info["face_normals"]).sum(1)).sum() / 3
    assert float(volume) == pytest.approx(res.volumes[0], rel=1e-12)


def test_surface_analysis_of_an_empty_mesh():
    res = compute(3, random_points(3, 100), n_inner=0)
    mesh, info = surface_info(res)
    assert info["face_areas"].shape == (0,)
    assert info["face_normals"].shape == (0, 3)


def test_surface_analysis_is_scale_free():
    p = random_points(3, 300, seed=1)
    p[0] = [0.5, 0.5, 0.5]
    mb, big = surface_info(compute(3, p, n_inner=1))
    ms, small = surface_info(compute(3, p * 1e-6, limits=[[0, 1e-6]] * 3, n_inner=1))
    # The faces may be listed in a different order, so match them by the
    # neighboring cell on their other side.
    ib = torch.argsort(mb["facet_neighbors"])
    i_s = torch.argsort(ms["facet_neighbors"])
    assert torch.equal(mb["facet_neighbors"][ib], ms["facet_neighbors"][i_s])
    assert torch.allclose(big["face_normals"][ib], small["face_normals"][i_s], atol=1e-9)
    assert torch.allclose(big["face_circularities"][ib],
                          small["face_circularities"][i_s], atol=1e-9)


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (1e-6, 1.0)])
def test_polygon_cell_info_2d(scale, shift):
    """Areas and centroids of the 2D cells, including small cells far from the
    origin, where the shoelace formula would otherwise lose precision."""
    p = random_points(2, 400) * scale + shift
    limits = [[shift, shift + scale]] * 2
    top = compute(2, p, "topology", limits=limits, threads=1)
    vertices = voroplex.compute_unique_vertices(top, torch.tensor(p))
    polygons, ids = voroplex.cell_polytopes(top)
    width = max(len(c) for c in polygons)
    vids = torch.full((len(polygons), width), -1, dtype=torch.int64)
    for i, c in enumerate(polygons):
        vids[i, :len(c)] = torch.tensor(c)
    counts = torch.tensor([len(c) for c in polygons])
    mask = vids >= 0
    info = analysis.compute_polygon_geometry(vertices, vids, counts, mask)
    assert np.allclose(info["cell_areas"].numpy(), top.areas, rtol=1e-8)
    # Reference centroids from the per-cell fan triangles, whose corners are
    # measured relative to each generator.
    per = compute(2, p, "per_cell", limits=limits, threads=1)
    owners, simplices = voroplex.cell_fan_simplices(per)
    owners, simplices = owners.numpy(), simplices.numpy()
    q = per.vertices[simplices] - p[owners][:, None, :]
    area = 0.5 * (q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0])
    offset = q.sum(axis=1) / 3.0
    m = np.bincount(owners, area, minlength=len(p))
    centroid = p + np.stack([np.bincount(owners, area * offset[:, k], minlength=len(p))
                             for k in range(2)], 1) / m[:, None]
    error = np.abs(info["cell_centroids"].numpy() - centroid).max()
    assert error < 1e-9 * scale
