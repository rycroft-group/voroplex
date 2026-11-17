"""
Render an optimized 3D foam to PNG with PyVista -- whole cube and corner cutaway.

Replaces the manual ParaView workflow. Two things it fixes:

  * **The cut is exact.** ParaView's Clip triangulates every polygon it
    touches, and the whole-cell Programmable Filter leaves cells protruding
    into the removed corner, with their facets seen edge-on as a mess of thin
    slivers. Here the corner notch is removed analytically: each Voronoi facet
    is clipped against the notch, and the exposed surfaces are the exact
    cross-sections of the cells with the three notch planes. The result is a
    flat cut showing the cell cross-sections as clean polygons, with no
    protruding cells and no triangulation.

  * **Every image is the same size, from the same camera.** The window size
    and camera are fixed in code, so the renders line up without cropping or
    padding.

Geometry notes:

  The notch N = [a,1] x [b,1] x [c,1] is removed. N is convex but its
  complement is not, so a facet F is kept as the union of three convex pieces,

      F n {x<=a},  F n {x>=a, y<=b},  F n {x>=a, y>=b, z<=c},

  which tile F \\ N exactly. The seams between pieces are invisible because
  surfaces are drawn without edges; every line in the figure comes from an
  explicit line set (the Voronoi edges clipped to outside N, plus the outlines
  of the cut cross-sections), so no seam is ever drawn.

Usage:
    python examples/render_foam_pyvista.py <history.npz> [out_dir]
                                           [--cut=a,b,c] [--size=1400] [--zoom=Z]
                                           [--mode=cells|cut] [--edges=flat|tube]
                                           [--rule=peel|volume|vertex|centroid] [--vote=F]
                                           [--threads=4] [--stem=NAME]

Writes <stem>.png (whole cube) and <stem>_inside.png (corner cutaway).
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

import voroplex

pv.OFF_SCREEN = True

EPS = 1e-12

# A cut plane placed arbitrarily will graze some cells nearly tangentially,
# and those cells' cross-sections come out as slivers a few thousandths of a
# cell wide. They draw as two nearly-coincident lines, which is what makes an
# otherwise clean cut look ragged. `choose_cut` shifts each plane by a few
# percent to avoid them. A cross-section counts as a sliver when its shorter
# in-plane extent falls below this fraction of the mean cell size.
SLIVER_FRACTION = 0.15
CUT_SEARCH = (0.44, 0.56, 0.005)

SKIN_COLOR = "#aaffff"      # outer boundary of the cube: light cyan
CUT_COLOR = "#aaff7f"       # exposed interior cell walls: light green
EDGE_COLOR = "#000000"
# Line width at the 1500 px render size. The figure is included at
# \\textwidth alongside five others, so thin edges disappear on the page. 8.0
# matches the weight the tubes below used to give: measured on the same run,
# flat lines at this width darken 11.8% of the frame against the tubes' 11.4%,
# at the same 8 px median stroke.
EDGE_WIDTH = 8.0
# Radius used when edges are drawn as tubes, in box units. A cell is about
# 0.11 across, so 0.005 is roughly one twentieth of a cell width, about 9 px
# at the framing used here.
EDGE_TUBE_RADIUS_WHEN_TUBED = 0.005
# Edges are flat lines of EDGE_WIDTH pixels. Tubes are real geometry and take
# the lighting, so each edge gets a specular highlight and a visible
# thickness; at the size the figure is printed that reads as a rod-and-joint
# model rather than a wireframe over the cell surfaces. --edges=tube sets this
# to the radius above. It is a module-level default so that callers importing
# render_run(), such as make_foam_3d_grid.py, get flat edges too.
EDGE_TUBE_RADIUS = None
EDGE_TUBE_SIDES = 20
# What makes a tube read as round is the specular highlight, not the diffuse
# term: a near-black tube has no diffuse range to shade with. The highlight is
# white and additive, so the base can stay black and the tube still shows a
# bright line down its length -- but only once the specular COLOUR is forced
# white, which is done where the mesh is added. Measured across one tube, the
# brightness goes from flat (11-15) to 18-90 with that one change.
EDGE_TUBE_COLOR = "#141414"
EDGE_TUBE_AMBIENT = 0.02
EDGE_TUBE_DIFFUSE = 0.90
EDGE_TUBE_SPECULAR = 1.0
# Higher powers pinch the highlight into a thin streak; 8 keeps it broad
# enough to model the whole curve of the tube.
EDGE_TUBE_SPECULAR_POWER = 8
AMBIENT = 0.33

# Looking down the (1,1,1) diagonal, so the notch -- cut out of that same
# corner -- faces the viewer. Fixed so every render frames the box identically.
CAMERA = [(3.3, 3.0, 2.6), (0.5, 0.5, 0.5), (0, 0, 1)]
# Fraction of the frame kept clear around the model.
FIT_MARGIN = 0.02
# Depth-buffer units the edge lines are pulled toward the camera. A flat line
# is drawn at the depth of its own edge, but the two faces meeting there slope
# away from it, so the further a pixel sits from the centre of the stroke the
# more likely the face in front wins the depth test. The bite taken out of the
# line therefore grows with the line's width, and the offset has to grow with
# it: at 12 units per pixel of EDGE_WIDTH every edge draws at its full nominal
# width, cube silhouette and interior cell edge alike. Too small and the
# interior edges come out visibly thinner than the silhouette, which has no
# coplanar surface behind it; far too large and edges on the far side of the
# model start showing through it.
COINCIDENT_OFFSET = 12 * EDGE_WIDTH
# How much of a cell must fall inside the notch for it to be removed, and by
# what measure. See wholecell_actors for why volume at one half is the best
# whole-cell approximation of the notch.
VOTE_DEFAULT = 0.5
RULE_DEFAULT = "peel"
# Notch for the whole-cell view. Removing whole cells empties rather more than
# the notch itself, so it is set deeper than the analytic cut's would be.
CELL_CUT_DEFAULT = (0.5, 0.5, 0.5)
# Grid searched for the whole-cell notch, and how heavily an isolated cell is
# penalised relative to a cell width of staircase roughness.
CELL_CUT_SEARCH = (0.45, 0.58, 0.015)
ISOLATION_WEIGHT = 0.01


# ── Geometry helpers ───────────────────────────────────────────────────────


def clip_polygon(poly, axis, value, keep_below):
    """Sutherland-Hodgman clip of a polygon by one axis-aligned half-space.

    `poly` is (n, 3). Keeps the side with coordinate <= value when
    `keep_below`, otherwise >= value. Returns (m, 3), possibly empty.
    """
    if len(poly) == 0:
        return poly
    sign = 1.0 if keep_below else -1.0
    d = sign * (poly[:, axis] - value)          # <= 0 means inside
    out = []
    n = len(poly)
    for k in range(n):
        j = (k + 1) % n
        pk, pj, dk, dj = poly[k], poly[j], d[k], d[j]
        if dk <= 0:
            out.append(pk)
        if (dk < 0 < dj) or (dj < 0 < dk):
            t = dk / (dk - dj)
            out.append(pk + t * (pj - pk))
    return np.asarray(out) if out else np.empty((0, 3))


def clip_polygon_many(poly, constraints):
    """Clip by a list of (axis, value, keep_below) half-spaces in turn."""
    for axis, value, keep_below in constraints:
        poly = clip_polygon(poly, axis, value, keep_below)
        if len(poly) < 3:
            return np.empty((0, 3))
    return poly


def subtract_notch(poly, cut):
    """Return the pieces of `poly` lying outside the corner notch.

    The notch is [cut0,1] x [cut1,1] x [cut2,1]; its complement is not convex,
    so the result is up to three convex pieces that tile poly \\ notch.
    """
    a, b, c = cut
    pieces = [
        clip_polygon_many(poly, [(0, a, True)]),
        clip_polygon_many(poly, [(0, a, False), (1, b, True)]),
        clip_polygon_many(poly, [(0, a, False), (1, b, False), (2, c, True)]),
    ]
    return [p for p in pieces if len(p) >= 3]


def order_planar_points(pts, axis):
    """Order coplanar points into a convex polygon, by angle about the mean.

    Voronoi cells are convex, so any cross-section of one is a convex polygon
    and angular ordering reconstructs it exactly.
    """
    uv = [i for i in range(3) if i != axis]
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, uv[1]] - centre[uv[1]],
                     pts[:, uv[0]] - centre[uv[0]])
    return pts[np.argsort(ang)]


def cell_plane_section(face_polys, axis, value, tol=1e-9):
    """Cross-section of one convex cell with the plane coord[axis] = value."""
    pts = []
    for poly in face_polys:
        n = len(poly)
        for k in range(n):
            p, q = poly[k], poly[(k + 1) % n]
            dp, dq = p[axis] - value, q[axis] - value
            if abs(dp) <= tol:
                pts.append(p)
            if abs(dq) <= tol:
                pts.append(q)
            elif (dp < -tol and dq > tol) or (dp > tol and dq < -tol):
                pts.append(p + dp / (dp - dq) * (q - p))
    if len(pts) < 3:
        return np.empty((0, 3))

    pts = np.asarray(pts)
    # Points arrive once per incident face, so deduplicate before ordering.
    keys = np.round(pts / max(tol, 1e-12)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    pts = pts[np.sort(keep)]
    if len(pts) < 3:
        return np.empty((0, 3))
    return order_planar_points(pts, axis)


def clip_segment_outside_notch(p, q, cut):
    """Parts of segment p->q lying outside the corner notch (0, 1 or 2 pieces).

    A line meets a convex box in a single interval, so the outside part is the
    segment minus one interval.
    """
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        lo = cut[axis]
        dp, dq = p[axis], q[axis]
        delta = dq - dp
        if abs(delta) < EPS:
            if dp < lo:
                return [(p, q)]                  # never enters the notch
            continue
        t = (lo - dp) / delta
        if delta > 0:
            t0 = max(t0, t)                      # inside only for t >= t
        else:
            t1 = min(t1, t)
    if t0 >= t1:
        return [(p, q)]                          # no overlap with the notch
    out = []
    if t0 > 1e-9:
        out.append((p, p + t0 * (q - p)))
    if t1 < 1 - 1e-9:
        out.append((p + t1 * (q - p), q))
    return out


def section_thicknesses(cell_faces, face_polys, axis, value, cell_size):
    """Shorter in-plane extent of every cell cross-section on one plane.

    Returned in units of the mean cell size, so the numbers are comparable
    between runs and between axes.
    """
    out = []
    for faces in cell_faces:
        polys = [face_polys[i] for i in faces]
        lo = np.min([p.min(axis=0) for p in polys], axis=0)
        hi = np.max([p.max(axis=0) for p in polys], axis=0)
        if not (lo[axis] < value < hi[axis]):
            continue
        section = cell_plane_section(polys, axis, value)
        if len(section) < 3:
            continue
        uv = [k for k in range(3) if k != axis]
        extent = np.sort((section.max(axis=0) - section.min(axis=0))[uv])
        out.append(extent[0] / cell_size)
    return np.asarray(out) if out else np.zeros(0)


def choose_cut(cell_faces, face_polys, n_cells,
               window=CUT_SEARCH, verbose=True):
    """Pick each notch plane so that it grazes as few cells as possible.

    Each axis is chosen independently: the three planes cut disjoint sets of
    cells almost everywhere, so there is nothing to gain from a joint search.
    Ties are broken by the thickest worst-case cross-section.
    """
    cell_size = n_cells ** (-1.0 / 3.0)
    lo, hi, step = window
    chosen = []
    for axis in range(3):
        best = None
        for value in np.arange(lo, hi + 0.5 * step, step):
            thin = section_thicknesses(cell_faces, face_polys, axis,
                                       float(value), cell_size)
            if len(thin) == 0:
                continue
            score = (int((thin < SLIVER_FRACTION).sum()), -float(thin.min()))
            if best is None or score < best[0]:
                best = (score, float(value))
        chosen.append(0.5 if best is None else best[1])
        if verbose and best is not None:
            print(f"  cut axis {axis}: {chosen[-1]:.3f}  "
                  f"(slivers {best[0][0]}, thinnest {-best[0][1]:.3f} cells)")
    return tuple(chosen)


# ── Mesh assembly ──────────────────────────────────────────────────────────


def polygons_to_polydata(polys):
    """Build a PolyData from a list of (n_i, 3) polygons."""
    if not polys:
        return pv.PolyData()
    points = np.vstack(polys)
    faces, start = [], 0
    for poly in polys:
        n = len(poly)
        faces.append(n)
        faces.extend(range(start, start + n))
        start += n
    return pv.PolyData(points, np.asarray(faces, dtype=np.int64))


def segments_to_polydata(segments):
    if not segments:
        return pv.PolyData()
    points = np.vstack([np.vstack(s) for s in segments])
    lines = []
    for i in range(len(segments)):
        lines.extend([2, 2 * i, 2 * i + 1])
    return pv.PolyData(points, lines=np.asarray(lines, dtype=np.int64))


def polygon_outline_segments(poly):
    return [(poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))]


def build_foam(points_np, limits, threads):
    """Tessellate and return (vertices, faces, face_info, per-cell face lists)."""
    result = voroplex.compute_voronoi_3d(points_np, limits, representation="topology",
                                      threads=threads)
    pts_t = torch.tensor(points_np, dtype=torch.float64)
    verts = voroplex.compute_unique_vertices(result, pts_t).detach().cpu().numpy()

    inner_vids, inner_info = voroplex.unique_facets(
        result, facet_type="inner")
    surf_vids, surf_info = voroplex.unique_facets(
        result, facet_type="surface")

    face_vids = list(inner_vids) + list(surf_vids)
    info = np.concatenate([np.asarray(inner_info).reshape(-1, 2),
                           np.asarray(surf_info).reshape(-1, 2)], axis=0)
    is_surface = np.concatenate([np.zeros(len(inner_vids), dtype=bool),
                                 np.ones(len(surf_vids), dtype=bool)])

    n_cells = len(points_np)
    cell_faces = [[] for _ in range(n_cells)]
    for i, (owner, neigh) in enumerate(info):
        cell_faces[owner].append(i)
        if neigh >= 0:
            cell_faces[neigh].append(i)

    return verts, face_vids, is_surface, info, cell_faces


def unique_edges(face_vids):
    edges = set()
    for vids in face_vids:
        n = len(vids)
        for k in range(n):
            a, b = int(vids[k]), int(vids[(k + 1) % n])
            edges.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(edges), dtype=np.int64)


# ── Scenes ─────────────────────────────────────────────────────────────────


def choose_cut_cells(points, verts, face_vids, is_surface, info, cell_faces,
                     vote=VOTE_DEFAULT, samples=1_200_000, seed=0,
                     window=CELL_CUT_SEARCH, verbose=True):
    """Place the notch so the whole-cell cavity looks most like a clean box.

    Keeping cells whole means the cavity wall is a staircase rather than a
    plane, and how ragged that staircase looks depends on where the notch
    planes fall relative to the cell walls. Two things are scored:

      * roughness -- the volume that ends up on the wrong side of the notch,
        divided by the area of the three cut faces and by the mean cell size.
        That makes it a thickness, in cell widths, and it is scale free: the
        raw mismatch alone is minimised by simply shrinking the notch.
      * isolation -- cells that are almost entirely surrounded by the other
        side. These are the thin spikes and stranded islands that read as
        "messy" however small their volume.

    Volume fractions are estimated by sampling, which is cheap enough to scan
    the whole grid; the final selection is then recomputed exactly.
    """
    from scipy.spatial import cKDTree

    n = len(points)
    cell_size = n ** (-1.0 / 3.0)
    rng = np.random.default_rng(seed)
    S = rng.random((samples, 3))
    _, owner = cKDTree(points).query(S, workers=-1)
    total = np.bincount(owner, minlength=n).astype(float)

    neighbours = [[] for _ in range(n)]
    for i, (o, nb) in enumerate(info):
        if not is_surface[i]:
            neighbours[o].append(nb)
            neighbours[nb].append(o)
    degree = np.maximum(np.array([len(v) for v in neighbours], dtype=float), 1.0)

    def score(a, b, c):
        inside = (S[:, 0] > a) & (S[:, 1] > b) & (S[:, 2] > c)
        v_in = np.bincount(owner, weights=inside.astype(float), minlength=n)
        removed = (v_in / np.maximum(total, 1)) > vote
        mismatch = (removed[owner] != inside).mean()
        area = (1 - b) * (1 - c) + (1 - a) * (1 - c) + (1 - a) * (1 - b)
        rough = mismatch / area / cell_size
        opposed = np.array([sum(1 for j in neighbours[i] if removed[j] != removed[i])
                            for i in range(n)])
        isolated = int(((opposed / degree) > 0.7).sum())
        return rough + ISOLATION_WEIGHT * isolated, rough, isolated

    lo, hi, step = window
    grid = np.arange(lo, hi + 0.5 * step, step)
    best = None
    for a in grid:
        for b in grid:
            for c in grid:
                s = score(float(a), float(b), float(c))
                if best is None or s[0] < best[0][0]:
                    best = (s, (float(a), float(b), float(c)))
    (_, rough, isolated), cut = best
    if verbose:
        print(f"  notch at ({cut[0]:.3f}, {cut[1]:.3f}, {cut[2]:.3f}): "
              f"staircase {rough:.3f} cells thick, {isolated} isolated cells")
    return cut


def cell_neighbours(info, is_surface, n_cells):
    nbr = [[] for _ in range(n_cells)]
    for i, (owner, neigh) in enumerate(info):
        if not is_surface[i]:
            nbr[owner].append(neigh)
            nbr[neigh].append(owner)
    return nbr


def cell_layers(info, is_surface, neighbours, wall_code, n_cells):
    """Layer index of every cell, counted inward from one face of the box.

    Cells touching that face are layer 1, their neighbours layer 2, and so on.
    Because a step crosses one cell, the layer index advances with depth at the
    rate the cells themselves are sized -- so in a columnar foam, layer 2 in z
    is the second course of columns, not an arbitrary height.
    """
    layer = np.full(n_cells, np.iinfo(np.int64).max, dtype=np.int64)
    queue = deque()
    for i, (owner, neigh) in enumerate(info):
        if is_surface[i] and neigh == wall_code and layer[owner] > 1:
            layer[owner] = 1
            queue.append(owner)
    while queue:
        u = queue.popleft()
        for v in neighbours[u]:
            if layer[v] > layer[u] + 1:
                layer[v] = layer[u] + 1
                queue.append(v)
    return layer


def peel_depth(layer, volumes, target=0.5):
    """How many layers to peel so the slab holds `target` of the box by volume."""
    reachable = layer[layer < np.iinfo(np.int64).max]
    if reachable.size == 0:
        return 0
    total = volumes.sum()
    return min(range(1, int(reachable.max()) + 1),
               key=lambda k: abs(volumes[layer <= k].sum() / total - target))


def convex_volume(faces):
    """Volume of a convex polyhedron given its face polygons.

    Orientation is not assumed: the origin is moved inside the body and each
    face normal is flipped to point away from it, so every term of the
    divergence-theorem sum is positive. Sutherland-Hodgman preserves the
    orientation of the faces it clips but the cap polygons are rebuilt by
    angular sorting, so the two sources do not agree on winding.
    """
    if not faces:
        return 0.0
    origin = np.vstack(faces).mean(axis=0)
    total = 0.0
    for poly in faces:
        p = poly - origin
        nrm = np.zeros(3)
        for k in range(len(p)):
            nrm += np.cross(p[k], p[(k + 1) % len(p)])
        total += abs(np.dot(p[0], 0.5 * nrm))
    return total / 3.0


def cell_notch_volume_fraction(face_polys, cut):
    """Fraction of one convex cell's volume that lies inside the notch."""
    whole = convex_volume(face_polys)
    if whole <= 0:
        return 0.0
    keep_in = [(0, cut[0], False), (1, cut[1], False), (2, cut[2], False)]
    inside = []
    for poly in face_polys:
        piece = clip_polygon_many(poly, keep_in)
        if len(piece) >= 3:
            inside.append(piece)
    # The notch's own three planes close the clipped body.
    for axis, value in enumerate(cut):
        cap = cell_plane_section(face_polys, axis, value)
        if len(cap) < 3:
            continue
        cap = clip_polygon_many(
            cap, [(k, cut[k], False) for k in range(3) if k != axis])
        if len(cap) >= 3:
            inside.append(cap)
    return convex_volume(inside) / whole


def cell_vertex_ids(face_vids, cell_faces):
    """Unique vertex ids of each cell, from the faces that bound it."""
    return [np.unique(np.concatenate([np.asarray(face_vids[i]) for i in faces]))
            if faces else np.zeros(0, dtype=np.int64)
            for faces in cell_faces]


def wholecell_actors(verts, face_vids, is_surface, info, cell_faces, cut,
                     vote=VOTE_DEFAULT, rule=RULE_DEFAULT):
    """Corner cutaway that removes WHOLE cells. Nothing is ever sliced.

    Which cells go is decided by a vote over the cell, with `vote` the
    fraction that has to fall inside the notch:

      "peel"     no notch at all: whole courses of cells are peeled inward
                 from the three visible faces of the box, and a cell goes if
                 all three reached it. The depth of each peel is chosen so the
                 slab holds half the box by volume, which makes the cavity the
                 same size as a half-box corner notch. Layers follow the cells,
                 so this is the only rule whose cavity wall never cuts across
                 an elongated cell at an arbitrary point along its length.
      "volume"   fraction of the cell's VOLUME inside the notch, computed
                 exactly by clipping the cell against the notch. At vote=0.5
                 this is the selection that minimises the mismatch between the
                 kept solid and the intended notch -- provably, because each
                 cell contributes min(V_in, V_out) and is assigned the smaller
                 of the two independently of every other cell.
      "vertex"   fraction of the cell's VERTICES inside the notch.
      "centroid" the original rule: is the cell's centroid inside the notch.

    Whichever rule is used, roughly 2% of the box ends up on the wrong side --
    that is the price of keeping cells whole, and it is why the cavity wall is
    a staircase of complete cells rather than a flat face.

    Every surface drawn here is a complete Voronoi facet: nothing is clipped
    and nothing is cross-sectioned.
    """
    a, b, c = cut
    face_polys = [verts[np.asarray(vids)] for vids in face_vids]
    keep = np.ones(len(cell_faces), dtype=bool)

    if rule == "peel":
        # Peel whole courses of cells inward from the three visible faces and
        # keep what none of the three reached. Because a layer is one cell
        # thick by construction, the cavity wall lands on cell boundaries
        # rather than cutting across them -- which matters most where the
        # cells are elongated, as a plane there slices every cell at a
        # different point along its length.
        n_cells = len(cell_faces)
        neighbours = cell_neighbours(info, is_surface, n_cells)
        volumes = np.array([convex_volume([face_polys[i] for i in faces])
                            if faces else 0.0 for faces in cell_faces])
        removed = np.ones(n_cells, dtype=bool)
        depths = []
        for wall_code in (-2, -4, -6):          # x=1, y=1, z=1
            layer = cell_layers(info, is_surface, neighbours, wall_code,
                                n_cells)
            k = peel_depth(layer, volumes)
            depths.append(k)
            removed &= layer <= k
        keep = ~removed
        print(f"  peeled {tuple(depths)} layers from (x, y, z)")
    elif rule == "volume":
        for ci, faces in enumerate(cell_faces):
            if not faces:
                continue
            frac = cell_notch_volume_fraction([face_polys[i] for i in faces],
                                              cut)
            keep[ci] = frac < vote
    elif rule == "vertex":
        inside = (verts[:, 0] > a) & (verts[:, 1] > b) & (verts[:, 2] > c)
        for ci, vids in enumerate(cell_vertex_ids(face_vids, cell_faces)):
            if len(vids):
                keep[ci] = inside[vids].mean() < vote
    elif rule == "centroid":
        for ci, vids in enumerate(cell_vertex_ids(face_vids, cell_faces)):
            if len(vids):
                cen = verts[vids].mean(axis=0)
                keep[ci] = not (cen[0] > a and cen[1] > b and cen[2] > c)
    else:
        raise ValueError(f"unknown selection rule {rule!r}")

    skin, exposed = [], []
    for i, (owner, neigh) in enumerate(info):
        poly = verts[np.asarray(face_vids[i])]
        if is_surface[i]:
            if keep[owner]:
                skin.append(poly)
        elif keep[owner] != keep[neigh]:
            # A wall between a kept cell and a removed one: now on the surface.
            exposed.append(poly)

    segs = []
    for va, vb in unique_edges_from_polys(skin + exposed):
        segs.append((va, vb))

    print(f"  {rule} vote {vote:g}: kept {int(keep.sum())}/{len(keep)} cells, "
          f"{len(exposed)} walls exposed")
    return (polygons_to_polydata(skin), segments_to_polydata(segs),
            polygons_to_polydata(exposed))


def unique_edges_from_polys(polys, tol=1e-9):
    """Deduplicated boundary segments of a list of polygons, by geometry.

    The polygons come from a mix of sources, so they cannot be deduplicated by
    vertex id; two adjacent drawn faces share an edge and would otherwise draw
    it twice.
    """
    seen, out = set(), []
    for poly in polys:
        n = len(poly)
        for k in range(n):
            p, q = poly[k], poly[(k + 1) % n]
            kp = tuple(np.round(p / tol).astype(np.int64))
            kq = tuple(np.round(q / tol).astype(np.int64))
            key = (kp, kq) if kp <= kq else (kq, kp)
            if key in seen:
                continue
            seen.add(key)
            out.append((p, q))
    return out


def whole_cube_actors(verts, face_vids, is_surface):
    skin = [verts[np.asarray(vids)]
            for vids, s in zip(face_vids, is_surface) if s]
    edges = unique_edges([vids for vids, s in zip(face_vids, is_surface) if s])
    segs = [(verts[a], verts[b]) for a, b in edges]
    return polygons_to_polydata(skin), segments_to_polydata(segs), pv.PolyData()


def cutaway_actors(verts, face_vids, is_surface, cell_faces, cut):
    """Skin, edge and cut-surface meshes for the corner-notch view."""
    face_polys = [verts[np.asarray(vids)] for vids in face_vids]

    # Outer skin, with the notch bitten out of it.
    skin = []
    for poly, surf in zip(face_polys, is_surface):
        if surf:
            skin.extend(subtract_notch(poly, cut))

    # Cut surfaces: each cell's cross-section with each of the three notch
    # planes, restricted to the part of that plane that actually bounds the
    # notch. Together these tile the three walls of the notch exactly.
    a, b, c = cut
    limit = {0: [(1, b, False), (2, c, False)],
             1: [(0, a, False), (2, c, False)],
             2: [(0, a, False), (1, b, False)]}
    cuts = []
    for faces in cell_faces:
        polys = [face_polys[i] for i in faces]
        lo = np.min([p.min(axis=0) for p in polys], axis=0)
        hi = np.max([p.max(axis=0) for p in polys], axis=0)
        for axis, value in enumerate(cut):
            if not (lo[axis] < value < hi[axis]):
                continue
            section = cell_plane_section(polys, axis, value)
            if len(section) < 3:
                continue
            section = clip_polygon_many(section, limit[axis])
            if len(section) >= 3:
                cuts.append(section)

    # Every line in the figure is drawn explicitly: the Voronoi edges that
    # survive outside the notch, plus the outlines of the cut cross-sections.
    # Nothing is drawn from the surface meshes, so the seams between the
    # convex pieces above never appear.
    segs = []
    for va, vb in unique_edges(face_vids):
        segs.extend(clip_segment_outside_notch(verts[va], verts[vb], cut))
    for poly in cuts:
        segs.extend(polygon_outline_segments(poly))

    return (polygons_to_polydata(skin), segments_to_polydata(segs),
            polygons_to_polydata(cuts))


def _plot(skin, lines, cuts, size, zoom, out_path=None):
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    plotter.set_background("white")
    if skin.n_cells:
        plotter.add_mesh(skin, color=SKIN_COLOR, show_edges=False,
                         ambient=AMBIENT, diffuse=0.75, specular=0.05)
    if cuts.n_cells:
        plotter.add_mesh(cuts, color=CUT_COLOR, show_edges=False,
                         ambient=AMBIENT, diffuse=0.70, specular=0.05)
    if lines.n_cells:
        if EDGE_TUBE_RADIUS:
            # Real geometry rather than a screen-space line: the edge keeps a
            # fixed thickness in the scene, so it foreshortens with depth and
            # takes the lighting. It also sits proud of the surfaces it
            # bounds, so the coincident topology offset needed by flat lines
            # does not arise at all.
            actor = plotter.add_mesh(
                lines.tube(radius=EDGE_TUBE_RADIUS, n_sides=EDGE_TUBE_SIDES),
                color=EDGE_TUBE_COLOR, ambient=EDGE_TUBE_AMBIENT,
                diffuse=EDGE_TUBE_DIFFUSE, specular=EDGE_TUBE_SPECULAR,
                specular_power=EDGE_TUBE_SPECULAR_POWER)
            # VTK defaults the SPECULAR COLOUR to the mesh colour, so a dark
            # tube gets a dark highlight and renders as a flat bar no matter
            # how the specular coefficient is set. Forcing it white is what
            # actually makes the tube look round; without this line the
            # brightness across a tube is flat to within a couple of levels.
            actor.GetProperty().SetSpecularColor(1.0, 1.0, 1.0)
        else:
            actor = plotter.add_mesh(lines, color=EDGE_COLOR,
                                     line_width=EDGE_WIDTH, lighting=False,
                                     render_lines_as_tubes=False)
            # A flat cell edge lies exactly in the plane of the faces it
            # bounds, so the depth test eats into it and it renders thinner
            # and greyer than the cube's corner edges, which have no coplanar
            # surface behind them. Offsetting the lines toward the camera
            # makes every edge, cell or cube, draw at the same weight.
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(
                0, -COINCIDENT_OFFSET)
    plotter.camera_position = CAMERA
    plotter.camera.zoom(zoom)
    img = plotter.screenshot(None if out_path is None else str(out_path),
                             return_img=True)
    plotter.close()
    return img


def ink_bounds(img):
    """(x0, x1, y0, y1) of the non-white pixels of a rendered frame."""
    mask = np.asarray(img)[..., :3].min(axis=2) < 250
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def fit_zoom(skin, lines, cuts, size, margin=FIT_MARGIN):
    """Largest zoom that keeps the whole model inside the frame.

    A hand-picked zoom clips: the cube's silhouette is not symmetric about the
    frame centre -- its bottom vertex sits further from the centre than its top
    one -- so the bottom corner runs off the bottom edge before anything else
    touches a border. Probing at zoom 1, where the model is comfortably inside,
    measures how far the silhouette actually reaches and scales from that.
    """
    x0, x1, y0, y1 = ink_bounds(_plot(skin, lines, cuts, size, 1.0))
    centre = 0.5 * (size - 1)
    reach = max(centre - x0, x1 - centre, centre - y0, y1 - centre)
    return (centre - margin * size) / reach


def render(skin, lines, cuts, out_path, size, zoom):
    img = _plot(skin, lines, cuts, size, zoom, out_path)
    x0, x1, y0, y1 = ink_bounds(img)
    edge = min(x0, y0, size - 1 - x1, size - 1 - y1)
    if edge <= 0:
        raise RuntimeError(f"{out_path}: model is clipped by the frame "
                           f"(bounds x {x0}-{x1}, y {y0}-{y1} in {size}px)")
    print(f"  wrote {out_path}  (zoom {zoom:.3f}, smallest margin {edge} px)")


def render_run(history, out_dir, stem, size=1500, threads=4,
               cut=None, zoom=None, mode="cells", vote=VOTE_DEFAULT,
               rule=RULE_DEFAULT):
    """Render one run's whole-cube and corner-cutaway views.

    `cut=None` picks the notch position automatically, avoiding planes that
    graze cells; pass an explicit triple to override.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = np.load(history)["final_points"]
    print(f"Rendering {stem}: {len(points)} cells")

    limits = [[0, 1], [0, 1], [0, 1]]
    verts, face_vids, is_surface, info, cell_faces = build_foam(points, limits,
                                                                threads)
    if mode == "cells":
        if rule == "peel":
            cut = cut or CELL_CUT_DEFAULT      # unused, but keeps the signature
        elif cut is None:
            cut = choose_cut_cells(points, verts, face_vids, is_surface, info,
                                   cell_faces, vote)
        else:
            print(f"  notch at {cut}")
        inside = wholecell_actors(verts, face_vids, is_surface, info,
                                  cell_faces, cut, vote, rule)
    else:
        if cut is None:
            face_polys = [verts[np.asarray(vids)] for vids in face_vids]
            cut = choose_cut(cell_faces, face_polys, len(points))
        else:
            print(f"  notch at {cut}")
        inside = cutaway_actors(verts, face_vids, is_surface, cell_faces, cut)

    whole = whole_cube_actors(verts, face_vids, is_surface)
    # One zoom for both views, so the two rows of the composed figure show the
    # box at exactly the same size. The cutaway's silhouette is the same as the
    # whole cube's -- the notch is interior to it -- so fitting either works.
    if zoom is None:
        zoom = fit_zoom(*whole, size)
    render(*whole, out_dir / f"{stem}.png", size, zoom)
    render(*inside, out_dir / f"{stem}_inside.png", size, zoom)


def _flag(prefix, default):
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return a[len(prefix):]
    return default


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)

    history = Path(args[0])
    out_dir = Path(args[1]) if len(args) > 1 else history.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _flag("--stem=", None) or history.parent.name
    size = int(_flag("--size=", "1400"))
    threads = int(_flag("--threads=", "4"))
    raw_cut = _flag("--cut=", None)
    cut = tuple(float(v) for v in raw_cut.split(",")) if raw_cut else None
    raw_zoom = _flag("--zoom=", None)
    zoom = float(raw_zoom) if raw_zoom else None
    mode = _flag("--mode=", "cells")
    vote = float(_flag("--vote=", str(VOTE_DEFAULT)))
    rule = _flag("--rule=", RULE_DEFAULT)

    # Cell edges are drawn as flat lines by default. Tubes are real geometry
    # and take the lighting, which gives each edge a specular highlight and a
    # visible thickness; at the size the figure is printed that reads as a
    # rod-and-joint model rather than a wireframe over the cell surfaces.
    edges = _flag("--edges=", "flat")
    if edges not in ("flat", "tube"):
        raise SystemExit(f"--edges must be flat or tube, not {edges!r}")
    if edges == "tube":
        global EDGE_TUBE_RADIUS
        EDGE_TUBE_RADIUS = EDGE_TUBE_RADIUS_WHEN_TUBED

    render_run(history, out_dir, stem, size=size, threads=threads, cut=cut,
               zoom=zoom, mode=mode, vote=vote, rule=rule)


if __name__ == "__main__":
    main()
