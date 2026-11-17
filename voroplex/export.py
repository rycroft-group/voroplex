"""Export of tessellations for ParaView, and saving and loading of results.

export_vtu writes the faces (3D) or edges (2D) as a VTU file, and export_npz
and load_npz save a VoronoiResult to, and load it from, a NumPy archive.
"""
import numpy as np


# Version 2 added results without neighbor information (neighbors=False),
# which have no facet_neighbors array; radical tessellations, which have a
# radii array; the per-cell properties, stored as property_<name>, which
# replace the measures array; and results without cells
# (representation=None). Version 1 archives can still be read.
RESULT_SCHEMA_VERSION = 2
_READABLE_SCHEMA_VERSIONS = (1, 2)


def _export_vtu_2d(result, points, filename, facet_type="all", device="cpu"):
    try:
        from voroplex._mesh import compute_unique_vertices, unique_facets
        import torch
    except ImportError:
        raise ImportError(
            "2D VTU export requires PyTorch for mesh reconstruction.\n"
            "Install with: pip install torch"
        )

    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        raise ImportError(
            "VTU export requires VTK library.\n"
            "Install with: pip install vtk"
        )

    if result.representation == "topology":
        vertices_xy = compute_unique_vertices(
            result, points, device=device
        )
    else:
        from voroplex._mesh._packed import (
            global_vertex_map,
            unique_geometry_coordinates,
        )
        _, inverse = global_vertex_map(result, device)
        vertices_xy = unique_geometry_coordinates(
            result, inverse, device
        )
    vertices_xy_np = vertices_xy.detach().cpu().numpy()
    vertices_np = np.column_stack([
        vertices_xy_np,
        np.zeros(vertices_xy_np.shape[0], dtype=vertices_xy_np.dtype)
    ])

    if facet_type == "all":
        edges_vID, edge_info = unique_facets(
            result, facet_type="all", device=device)
        is_surface = (
            (edge_info[:, 0] < result.n_inner)
            & (
                (edge_info[:, 1] >= result.n_inner)
                | (edge_info[:, 1] < 0)
            )
        ).astype(np.int32)
    else:
        edge_kind = "surface" if facet_type == "surface" else "inner"
        edges_vID, edge_info = unique_facets(
            result, facet_type=edge_kind, device=device)
        fill = 1 if facet_type == "surface" else 0
        is_surface = np.full(len(edges_vID), fill, dtype=np.int32)

    edge_ids = np.asarray(edges_vID, dtype=np.int64).reshape((-1, 2))
    if len(edge_ids):
        edge_ids_t = torch.tensor(edge_ids, dtype=torch.int64, device=vertices_xy.device)
        edge_lengths = torch.linalg.norm(
            vertices_xy[edge_ids_t[:, 1]] - vertices_xy[edge_ids_t[:, 0]], dim=1
        ).detach().cpu().numpy()
    else:
        edge_lengths = np.empty((0,), dtype=np.float64)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices_np))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    for edge in edge_ids:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(edge[0]))
        line.GetPointIds().SetId(1, int(edge[1]))
        grid.InsertNextCell(line.GetCellType(), line.GetPointIds())

    if len(edge_info):
        owner_cells = edge_info[:, 0].astype(np.int64)
        neighbor_cells = edge_info[:, 1].astype(np.int64)
    else:
        owner_cells = np.empty((0,), dtype=np.int64)
        neighbor_cells = np.empty((0,), dtype=np.int64)

    owner_vtk = numpy_support.numpy_to_vtk(owner_cells)
    owner_vtk.SetName("OwnerCell")
    grid.GetCellData().AddArray(owner_vtk)

    neighbor_vtk = numpy_support.numpy_to_vtk(neighbor_cells)
    neighbor_vtk.SetName("NeighborCell")
    grid.GetCellData().AddArray(neighbor_vtk)

    surf_vtk = numpy_support.numpy_to_vtk(is_surface)
    surf_vtk.SetName("IsSurface")
    grid.GetCellData().AddArray(surf_vtk)

    lengths_vtk = numpy_support.numpy_to_vtk(edge_lengths)
    lengths_vtk.SetName("EdgeLength")
    grid.GetCellData().AddArray(lengths_vtk)

    # The owner cell's area, if it was computed.
    if "area" in result.properties:
        owner_area = np.asarray(result.areas, dtype=np.float64)[owner_cells]
        oarea_vtk = numpy_support.numpy_to_vtk(owner_area)
        oarea_vtk.SetName("OwnerArea")
        grid.GetCellData().AddArray(oarea_vtk)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

    print(f"Exported 2D Voronoi tessellation to {filename}")
    print(f"  Vertices:      {vertices_np.shape[0]}")
    print(f"  Total edges:   {len(edges_vID)}")
    if facet_type == "all":
        print(f"    Non-surface:   {int(np.sum(is_surface == 0))}  (IsSurface=0)")
        print(f"    Surface edges: {int(np.sum(is_surface == 1))}  (IsSurface=1)")


def export_vtu(result, points, filename, facet_type="all", device="cpu"):
    """
    Export Voronoi tessellation faces to VTK XML format (.vtu) for ParaView.

    Each Voronoi face is exported as a polygon, making the full cell-boundary
    structure directly visible in ParaView. Use the ``IsSurface`` cell attribute
    to filter between inner faces (between two generator cells) and surface
    faces (between a generator cell and an outer cell or the boundary of the domain).

    Parameters
    ----------
    result : VoronoiResult
        Packed 2D or 3D geometry or topology result.
    points : array_like, shape (N, 2) for 2D or (N, 3) for 3D
        Original generator points.
    filename : str
        Output filename (should end with .vtu).
    facet_type : {'all', 'surface', 'inner'}, default='all'
        Which faces/edges to include:

        - ``'all'``     — every unique face/edge of the computed cells.
          ``IsSurface=1`` marks the inner/outer interface or the boundary of the domain;
          all other faces/edges have ``IsSurface=0``.
        - ``'surface'`` — only faces/edges on the boundary of the domain or an outer cell.
        - ``'inner'``   — only faces/edges shared by two inner cells.
    device : str, default='cpu'
        Device for mesh reconstruction (``'cpu'`` or ``'cuda'``).

    Per-element cell attributes in the output (3D — polygon faces)
    ---------------------------------------------------------------
    OwnerCell    : int   — generator cell ID that owns this face
    NeighborCell : int   — adjacent cell ID; negative = boundary code
                          (-1=xmin, -2=xmax, -3=ymin, -4=ymax, -5=zmin, -6=zmax)
    IsSurface    : int   — 1 if surface face, 0 otherwise (only when facet_type='all')
    FaceArea     : float — area of the polygon
    OwnerVolume  : float — Voronoi volume of the owner cell, if computed

    Per-element cell attributes in the output (2D — line edges)
    ------------------------------------------------------------
    OwnerCell    : int   — generator cell ID that owns this edge
    NeighborCell : int   — adjacent cell ID; negative = boundary code
                          (-1=xmin, -2=xmax, -3=ymin, -4=ymax)
    IsSurface    : int   — 1 if surface edge, 0 otherwise (only when facet_type='all')
    EdgeLength   : float — length of the edge segment
    OwnerArea    : float — Voronoi area of the owner cell

    Examples
    --------
    >>> import voroplex
    >>> result = voroplex.compute_voronoi_3d(points)
    >>> voroplex.export_vtu(result, points, "voronoi.vtu")
    >>> # Open voronoi.vtu in ParaView — all cell boundaries are visible.
    >>> # Use Threshold on IsSurface==1 to isolate the boundary surface.
    """
    from voroplex._result import VoronoiResult

    if not isinstance(result, VoronoiResult):
        raise TypeError("result must be a VoronoiResult")
    if facet_type not in ("all", "surface", "inner"):
        raise ValueError(
            "facet_type must be 'all', 'surface', or 'inner', "
            f"got {facet_type!r}"
        )
    if result.dimension == 2:
        return _export_vtu_2d(
            result, points, filename, facet_type=facet_type, device=device
        )

    try:
        from voroplex._mesh import compute_unique_vertices, unique_facets
        import torch
    except ImportError:
        raise ImportError(
            "VTU export requires PyTorch for mesh reconstruction.\n"
            "Install with: pip install torch"
        )

    def _compute_face_areas(faces_vID, vertices):
        """Fan-triangulated polygon area, vectorized over every triangle
        of every face at once (single batched cross product + scatter-sum)
        instead of a Python loop launching one tensor op per triangle --
        the latter measured at 156s for 1.5M faces (200K points, 3D),
        entirely from per-call torch dispatch/kernel-launch overhead, not
        the actual arithmetic.
        """
        n_faces = len(faces_vID)
        if n_faces == 0:
            return torch.empty(
                0, dtype=vertices.dtype, device=vertices.device
            )
        counts = np.fromiter(
            (len(face) for face in faces_vID), dtype=np.int64, count=n_faces
        )
        flat_vertex_ids = np.fromiter(
            (v for face in faces_vID for v in face), dtype=np.int64
        )
        offsets = np.empty(n_faces + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        n_triangles = np.maximum(counts - 2, 0)
        total = int(n_triangles.sum())
        device = vertices.device
        if total == 0:
            return torch.zeros(n_faces, dtype=vertices.dtype, device=device)
        face_ids = np.repeat(np.arange(n_faces, dtype=np.int64), n_triangles)
        triangle_starts = np.cumsum(n_triangles) - n_triangles
        local_index = np.arange(total) - np.repeat(triangle_starts, n_triangles)
        base = offsets[face_ids]

        v0 = vertices[torch.as_tensor(flat_vertex_ids[base], device=device)]
        v1 = vertices[
            torch.as_tensor(flat_vertex_ids[base + 1 + local_index], device=device)
        ]
        v2 = vertices[
            torch.as_tensor(flat_vertex_ids[base + 2 + local_index], device=device)
        ]
        triangle_areas = (
            torch.linalg.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1) * 0.5
        )

        areas = torch.zeros(n_faces, dtype=vertices.dtype, device=device)
        areas.index_add_(
            0, torch.as_tensor(face_ids, device=device), triangle_areas
        )
        return areas

    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        raise ImportError(
            "VTU export requires VTK library.\n"
            "Install with: pip install vtk"
        )

    if result.representation == "topology":
        vertices = compute_unique_vertices(result, points, device=device)
    else:
        from voroplex._mesh._packed import (
            global_vertex_map,
            unique_geometry_coordinates,
        )
        _, inverse = global_vertex_map(result, device)
        vertices = unique_geometry_coordinates(result, inverse, device)
    vertices_np = vertices.detach().cpu().numpy()

    # Collect faces and build IsSurface flag

    if facet_type == "all":
        faces_vID, face_info = unique_facets(
            result, facet_type="all", device=device)
        is_surface = (
            (face_info[:, 0] < result.n_inner)
            & (
                (face_info[:, 1] >= result.n_inner)
                | (face_info[:, 1] < 0)
            )
        ).astype(np.int32)
    else:
        faces_vID, face_info = unique_facets(
            result, facet_type=facet_type, device=device)
        is_surface = None

    # Compute face areas
    face_areas = _compute_face_areas(faces_vID, vertices).cpu().numpy()

    # Set up VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices_np))

    # Create unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    # Insert each Voronoi face as a vtkPolygon
    for face in faces_vID:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(face))
        for i, vid in enumerate(face):
            polygon.GetPointIds().SetId(i, int(vid))
        grid.InsertNextCell(polygon.GetCellType(), polygon.GetPointIds())

    # --- Cell data arrays ---
    owner_cells = face_info[:, 0].astype(np.int64)
    neighbor_cells = face_info[:, 1].astype(np.int64)

    owner_vtk = numpy_support.numpy_to_vtk(owner_cells)
    owner_vtk.SetName("OwnerCell")
    grid.GetCellData().AddArray(owner_vtk)

    neighbor_vtk = numpy_support.numpy_to_vtk(neighbor_cells)
    neighbor_vtk.SetName("NeighborCell")
    grid.GetCellData().AddArray(neighbor_vtk)

    if is_surface is not None:
        surf_vtk = numpy_support.numpy_to_vtk(is_surface)
        surf_vtk.SetName("IsSurface")
        grid.GetCellData().AddArray(surf_vtk)

    areas_vtk = numpy_support.numpy_to_vtk(face_areas)
    areas_vtk.SetName("FaceArea")
    grid.GetCellData().AddArray(areas_vtk)

    # The owner cell's volume, if it was computed.
    if "volume" in result.properties:
        owner_vol = np.asarray(result.volumes, dtype=np.float64)[owner_cells]
        ovol_vtk = numpy_support.numpy_to_vtk(owner_vol)
        ovol_vtk.SetName("OwnerVolume")
        grid.GetCellData().AddArray(ovol_vtk)

    # Write to file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

    n_non_surface_out = (
        int(np.sum(is_surface == 0)) if is_surface is not None else 0
    )
    n_surf_out = (
        int(np.sum(is_surface == 1))
        if is_surface is not None
        else (len(faces_vID) if facet_type == "surface" else 0)
    )

    print(f"Exported Voronoi tessellation to {filename}")
    print(f"  Vertices:      {vertices_np.shape[0]}")
    print(f"  Total faces:   {len(faces_vID)}")
    if facet_type == "all":
        print(f"    Non-surface:   {n_non_surface_out}  (IsSurface=0)")
        print(f"    Surface faces: {n_surf_out}  (IsSurface=1)")
    print("  Open in ParaView — use Threshold on IsSurface to isolate boundary")


def export_npz(result, filename, points=None):
    """
    Save a packed VoroPlex result to a versioned NumPy archive.

    The archive stores numeric and Unicode arrays and loads with
    ``allow_pickle=False``.

    Parameters
    ----------
    result : VoronoiResult
        Packed result from ``compute_voronoi_3d()`` or ``compute_voronoi_2d()``.
    filename : str
        Output filename. A .npz extension is appended if not present.
    points : array_like, shape (N, d), optional
        The generator positions. A result does not store them, so save them
        here if they will be needed later, for example to compute further
        properties with :func:`compute_properties`. Load them with
        ``load_npz(filename, return_points=True)``.

    Examples
    --------
    >>> result = voroplex.compute_voronoi_3d(points, limits)
    >>> voroplex.export_npz(result, "result.npz", points=points)
    >>> result2, points2 = voroplex.load_npz("result.npz", return_points=True)
    >>> centroids = voroplex.compute_properties(result2, points2, ["centroid"])
    """
    from voroplex._result import VoronoiResult

    if not isinstance(result, VoronoiResult):
        raise TypeError("result must be a VoronoiResult")
    arrays = {
        "schema_version": np.asarray(
            RESULT_SCHEMA_VERSION, dtype=np.int32
        ),
        "dimension": np.asarray(result.dimension, dtype=np.int32),
        "representation": np.asarray(
            "none" if result.representation is None
            else result.representation
        ),
        "limits": result.limits,
        "periodic": result.periodic,
        "n_points": np.asarray(result.n_points, dtype=np.int64),
        "n_inner": np.asarray(result.n_inner, dtype=np.int64),
        "n_cells": np.asarray(result.n_cells, dtype=np.int64),
        "threads": np.asarray(result.threads, dtype=np.int32),
    }
    for name, values in result.properties.items():
        arrays["property_" + name] = values
    if points is not None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape != (result.n_points, result.dimension):
            raise ValueError(
                "points must have shape ({}, {}), got {}".format(
                    result.n_points, result.dimension, pts.shape)
            )
        arrays["points"] = pts
    if result.is_radical:
        arrays["radii"] = result.radii
    if result.has_cells:
        arrays.update({
            "cell_vertex_offsets": result.cell_vertex_offsets,
            "cell_facet_offsets": result.cell_facet_offsets,
            "facet_vertex_offsets": result.facet_vertex_offsets,
            "facet_vertices": result.facet_vertices,
        })
        if result.has_neighbors:
            arrays["facet_neighbors"] = result.facet_neighbors
            arrays["cell_surface_flags"] = result.cell_surface_flags
        else:
            arrays["cell_surface_flags"] = np.zeros(
                len(result), dtype=np.uint8
            )
        if result.has_vertices:
            arrays["vertices"] = result.vertices
        if result.has_signatures:
            arrays["vertex_signature_offsets"] = (
                result.vertex_signature_offsets
            )
            arrays["vertex_signature_ids"] = result.vertex_signature_ids
    np.savez_compressed(filename, **arrays)


def load_npz(filename, return_points=False):
    """
    Load a packed VoronoiResult archive.

    Parameters
    ----------
    filename : str
        Path to the .npz file.
    return_points : bool
        If True, also return the generator positions, if they were saved with
        ``export_npz(..., points=points)``, or None otherwise.

    Returns
    -------
    VoronoiResult, or (VoronoiResult, ndarray or None)
        The restored read-only result, and the points if ``return_points``.

    Examples
    --------
    >>> result = voroplex.load_npz("result.npz")
    >>> print(result.volumes)
    """
    from voroplex._result import VoronoiResult

    with np.load(filename, allow_pickle=False) as data:
        if data["schema_version"].item() not in _READABLE_SCHEMA_VERSIONS:
            raise ValueError("unsupported VoroPlex result schema version")
        representation = data["representation"].item()
        if representation == "none":
            representation = None
        dimension = data["dimension"].item()
        properties = {
            key[len("property_"):]: data[key]
            for key in data.files if key.startswith("property_")
        }
        # Earlier versions named the counts num_faces, num_vertices, and
        # num_edges.
        for old_name in ("num_faces", "num_vertices", "num_edges"):
            if old_name in properties:
                properties["n_" + old_name[4:]] = properties.pop(old_name)
        # Older archives store the areas or volumes as "measures".
        if "measures" in data.files:
            properties["volume" if dimension == 3 else "area"] = data["measures"]
        cells = {}
        if representation is not None:
            has_neighbors = "facet_neighbors" in data.files
            cells = dict(
                vertices=(
                    data["vertices"] if representation != "topology" else None
                ),
                cell_vertex_offsets=data["cell_vertex_offsets"],
                cell_facet_offsets=data["cell_facet_offsets"],
                facet_neighbors=(
                    data["facet_neighbors"] if has_neighbors else None
                ),
                has_neighbors=has_neighbors,
                facet_vertex_offsets=data["facet_vertex_offsets"],
                facet_vertices=data["facet_vertices"],
                # Earlier versions named these flags cell_boundary_flags.
                cell_surface_flags=data[
                    "cell_surface_flags" if "cell_surface_flags" in data.files
                    else "cell_boundary_flags"
                ],
                vertex_signature_offsets=(
                    data["vertex_signature_offsets"]
                    if representation != "geometry"
                    else None
                ),
                vertex_signature_ids=(
                    data["vertex_signature_ids"]
                    if representation != "geometry"
                    else None
                ),
            )
        result = VoronoiResult(
            dimension=dimension,
            representation=representation,
            limits=data["limits"],
            periodic=data["periodic"],
            n_points=data["n_points"].item(),
            n_inner=data["n_inner"].item(),
            threads=data["threads"].item(),
            n_cells=(
                data["n_cells"].item() if "n_cells" in data.files else None
            ),
            properties=properties,
            radii=data["radii"] if "radii" in data.files else None,
            **cells,
        )
        if not return_points:
            return result
        points = data["points"] if "points" in data.files else None
        return result, points


__all__ = ["export_vtu", "export_npz", "load_npz"]
