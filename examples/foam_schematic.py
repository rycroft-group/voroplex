"""
Schematic figure for the anisotropic-foam section.

Draws the three directions that are easy to conflate:

  1. the direction of a wall's NORMAL, n_f          (what the fabric tensor measures)
  2. the direction of the WALL itself, t_f          (perpendicular to n_f in 2D)
  3. the direction in which a CELL is ELONGATED     (perpendicular to the dominant normal)

The key, counter-intuitive relationship -- and the reason a "horizontal
principal axis" produces "vertical walls and vertically elongated cells" --
is that concentrating wall normals along u forces the walls themselves to lie
perpendicular to u, which flattens each cell along u and elongates it in the
perpendicular direction.

Panels:
  (a) one cell with n_f, t_f and the wall weight w_f labelled
  (b) equiaxed cells: normals spread uniformly, A is near-isotropic
  (c) cells elongated along y: normals concentrate along x, so A_xx > A_yy

The tensors quoted in (b) and (c) are computed from the drawn geometry, so the
schematic is self-consistent rather than illustrative only.

Run:
    python examples/foam_schematic.py [output.png]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from foam_figure_style import apply_paper_style


# Same aspect as every other panel in the section, so the three
# schematic images render at equal height beside one another.
PANEL_ASPECT = 1.09


CELL_FACE = "#fdf0e3"
CELL_EDGE = "#8a8a8a"
HILITE = "#c2410c"
NORMAL_C = "#1d4ed8"
TANGENT_C = "#047857"
GEN_C = "#d83b3b"


def hexagon(cx, cy, r, y_scale=1.0):
    """Pointy-top regular hexagon, optionally stretched in y."""
    ang = np.pi / 2.0 + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + r * np.cos(ang),
                            cy + y_scale * r * np.sin(ang)])


def hex_patch(n_cols, n_rows, r, y_scale=1.0):
    """A small honeycomb patch. Returns a list of polygons (each (6, 2))."""
    dx = r * np.sqrt(3.0)
    dy = 1.5 * r * y_scale
    polys = []
    for j in range(n_rows):
        for i in range(n_cols):
            cx = i * dx + (j % 2) * dx / 2.0
            cy = j * dy
            polys.append(hexagon(cx, cy, r, y_scale=y_scale))
    return polys


def fabric_tensor_2d(polys):
    """Length-weighted edge-normal tensor of a set of polygons.

    A = sum_e l_e n_e n_e^T / sum_e l_e

    Each interior wall is shared by two neighbouring cells and so is traversed
    twice; since both traversals contribute the same n n^T (the outer product
    is insensitive to the sign of n), the double counting cancels in the
    normalisation and A is unchanged.
    """
    num = np.zeros((2, 2))
    den = 0.0
    for poly in polys:
        nxt = np.roll(poly, -1, axis=0)
        vec = nxt - poly
        length = np.linalg.norm(vec, axis=1)
        keep = length > 1e-12
        vec, length = vec[keep], length[keep]
        tang = vec / length[:, None]
        norm = np.column_stack([-tang[:, 1], tang[:, 0]])
        for n_e, l_e in zip(norm, length):
            num += l_e * np.outer(n_e, n_e)
            den += l_e
    return num / max(den, 1e-300)


def draw_patch(ax, polys, normal_every=1, normal_len=0.16, show_normals=True):
    for poly in polys:
        ax.fill(poly[:, 0], poly[:, 1], facecolor=CELL_FACE,
                edgecolor=CELL_EDGE, linewidth=1.6, zorder=1)
    if not show_normals:
        return
    seen = set()
    for poly in polys:
        nxt = np.roll(poly, -1, axis=0)
        for k in range(len(poly)):
            v0, v1 = poly[k], nxt[k]
            mid = 0.5 * (v0 + v1)
            key = (round(mid[0], 4), round(mid[1], 4))
            if key in seen:
                continue
            seen.add(key)
            if len(seen) % normal_every:
                continue
            vec = v1 - v0
            L = np.linalg.norm(vec)
            if L < 1e-12:
                continue
            t = vec / L
            # Outward normal. The patch polygons are wound counter-clockwise,
            # for which (-t_y, t_x) points into the cell; panel (a) draws the
            # outward normal, so (b) and (c) must agree with it. A's value is
            # unaffected either way, since it is built from n n^T.
            n = np.array([t[1], -t[0]])
            ax.arrow(mid[0], mid[1], normal_len * n[0], normal_len * n[1],
                     head_width=0.055, head_length=0.055, length_includes_head=True,
                     fc=NORMAL_C, ec=NORMAL_C, linewidth=1.3, zorder=3)


def panel_a(ax):
    """One cell with the normal, the wall direction, and the weight labelled."""
    r = 1.0
    poly = hexagon(0.0, 0.0, r)
    ax.fill(poly[:, 0], poly[:, 1], facecolor=CELL_FACE, edgecolor=CELL_EDGE,
            linewidth=2.2, zorder=1)

    # Generator
    ax.plot(0, 0, "o", color=GEN_C, markersize=12, markeredgecolor="white",
            markeredgewidth=1.4, zorder=5)
    ax.annotate(r"$\mathbf{p}_c$", (0, 0), textcoords="offset points",
                xytext=(-34, -8), fontsize=20, color=GEN_C)

    # Highlight the RIGHT-HAND wall. For a pointy-top hexagon the vertices at
    # -30 deg and +30 deg share an x coordinate, so this wall is exactly
    # vertical -- which makes the normal/tangent distinction easiest to read.
    v0, v1 = poly[4], poly[5]
    ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color=HILITE, linewidth=6.0,
            solid_capstyle="round", zorder=4)

    mid = 0.5 * (v0 + v1)
    t = (v1 - v0) / np.linalg.norm(v1 - v0)      # wall direction, +y
    n = np.array([-t[1], t[0]])
    if n[0] < 0:
        n = -n                                    # outward normal, +x

    ax.arrow(mid[0], mid[1], 0.80 * n[0], 0.80 * n[1],
             head_width=0.14, head_length=0.16, length_includes_head=True,
             fc=NORMAL_C, ec=NORMAL_C, linewidth=3.2, zorder=6)
    ax.text(mid[0] + 0.90 * n[0], mid[1] + 0.90 * n[1] + 0.12,
            r"$\mathbf{n}_f$", fontsize=22, color=NORMAL_C,
            ha="left", va="bottom")

    ax.arrow(mid[0], mid[1], 0.55 * t[0], 0.55 * t[1],
             head_width=0.14, head_length=0.16, length_includes_head=True,
             fc=TANGENT_C, ec=TANGENT_C, linewidth=3.2, zorder=6)
    ax.text(mid[0] - 0.12, mid[1] + 0.62,
            r"$\mathbf{t}_f$", fontsize=22, color=TANGENT_C,
            ha="right", va="bottom")

    # Wall length, bracketed just INSIDE the cell. Placing it outside would run
    # along the outward normal arrow, which starts at the same wall midpoint.
    off = np.array([-0.30, 0.0])
    ax.annotate("", xy=v1 + off, xytext=v0 + off,
                arrowprops=dict(arrowstyle="<->", color=HILITE, linewidth=2.2))
    ax.text(v0[0] - 0.40, 0.0, r"$w_f$", fontsize=22, color=HILITE,
            ha="right", va="center")

    ax.set_xlim(-1.35, 2.30)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_patch(ax, y_scale, n_cols, n_rows, show_elongation):
    """A honeycomb patch with every wall normal drawn.

    The quoted tensor is computed from the cells actually drawn, so the number
    in the panel and the picture above it cannot disagree. Panel descriptions
    live in the LaTeX caption rather than in an in-axes title.
    """
    polys = hex_patch(n_cols=n_cols, n_rows=n_rows, r=0.5, y_scale=y_scale)

    draw_patch(ax, polys, normal_every=1, normal_len=0.17)
    A = fabric_tensor_2d(polys)

    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx = 0.5 * (x0 + x1)
    span_x, span_y = x1 - x0, y1 - y0

    right_pad = 0.16 * span_x
    if show_elongation:
        # Double-headed arrow marking the direction the cells are stretched...
        xarr = x1 + 0.16 * span_x
        ax.annotate("", xy=(xarr, y1), xytext=(xarr, y0),
                    arrowprops=dict(arrowstyle="<->", color=HILITE, linewidth=3.4))
        ax.text(xarr + 0.04 * span_x, 0.5 * (y0 + y1), "cell\nelongation",
                fontsize=17, color=HILITE,
                va="center", ha="left", linespacing=1.15)
        # ...and the principal axis of A, which is perpendicular to it.
        ymid = y0 - 0.16 * span_y
        ax.annotate("", xy=(cx + 0.28 * span_x, ymid),
                    xytext=(cx - 0.28 * span_x, ymid),
                    arrowprops=dict(arrowstyle="<->", color=NORMAL_C, linewidth=3.4))
        ax.text(cx, ymid - 0.05 * span_y,
                r"principal axis of $\mathbf{A}$",
                fontsize=17, color=NORMAL_C,
                va="top", ha="center")
        # Just enough for the arrow and its two-line caption. A wider margin
        # leaves a void that pushes this panel's letter away from its patch.
        right_pad = 0.62 * span_x

    # The tensor entries go BELOW the patch: placing them above collides with
    # the neighbouring panel's text in a three-across layout.
    ax.text(cx, y0 - 0.34 * span_y,
            rf"$A_{{xx}}={A[0,0]:.2f},\;A_{{yy}}={A[1,1]:.2f}$",
            fontsize=18, ha="center", va="top")

    ax.set_xlim(x0 - 0.10 * span_x, x1 + right_pad)
    ax.set_ylim(y0 - 0.52 * span_y, y1 + 0.16 * span_y)
    ax.set_aspect("equal")
    ax.axis("off")
    return A


def save_combined_schematic(panel_paths, out_path):
    """Write the three schematic panels as ONE image with (a)..(c) beneath.

    Composing the row from the rendered panels rather than with LaTeX
    subfigures packs them much closer together -- each panel's blank margin is
    cropped first -- so every panel is drawn larger for the same total width,
    and the letters are sized independently of the axis fonts.
    """
    from compose_figure_row import compose_grid

    # Panel (a) shows ONE cell where (b) and (c) show a patch of many, so at
    # equal panel height its single hexagon dwarfs theirs. Drawing it shorter
    # puts the cells themselves at a comparable size across the row.
    # The letter size is left to compose_grid, which derives it from the
    # figure width so that it prints at the same size as in the other figures.
    return compose_grid(panel_paths, out_path, ncols=3, labels="abc",
                        panel_height=3.6, gap=0.10, label_gap=0.02,
                        scales=(0.66, 1.0, 1.0))


def main(out_dir):
    """Write the schematic as three standalone panels.

    The paper composes them with LaTeX subfigures, which supply the (a)/(b)/(c)
    labels, so no letters are drawn here -- matching how every other figure in
    the section is assembled.
    """
    from foam_figure_style import save_panel

    apply_paper_style()
    out_dir = Path(out_dir)
    written = []

    # (a) one cell with the three directions labelled
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    panel_a(ax)
    written.append(save_panel(fig, out_dir, "a_wall_directions",
                              pad_aspect=PANEL_ASPECT))

    # (b) equiaxed cells
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    A_iso = panel_patch(ax, y_scale=1.0, n_cols=4, n_rows=4,
                        show_elongation=False)
    written.append(save_panel(fig, out_dir, "b_equiaxed",
                              pad_aspect=PANEL_ASPECT))

    # (c) cells elongated along y
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    A_ani = panel_patch(ax, y_scale=2.2, n_cols=4, n_rows=3,
                        show_elongation=True)
    written.append(save_panel(fig, out_dir, "c_elongated",
                              pad_aspect=PANEL_ASPECT))

    # ...and the same three panels as one composed row, which is what the
    # paper includes.
    save_combined_schematic(written, out_dir / "combined_row.png")

    for p in written:
        print(f"  wrote {p}")
    print(f"  (b) equiaxed   A = [[{A_iso[0,0]:.3f}, {A_iso[0,1]:.3f}], "
          f"[{A_iso[1,0]:.3f}, {A_iso[1,1]:.3f}]]")
    print(f"  (c) elongated  A = [[{A_ani[0,0]:.3f}, {A_ani[0,1]:.3f}], "
          f"[{A_ani[1,0]:.3f}, {A_ani[1,1]:.3f}]]")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "examples/schematic"
    main(target)
