"""
Compose several rendered panels into ONE image, with (a), (b), ... beneath.

Used for the two figures in Sec. 5.3 whose panels are not produced by a single
matplotlib call: the three-part schematic, and the six ParaView renders of the
optimized 3D foams. Both were previously assembled with LaTeX subfigures,
which forces wide, fixed gutters and gives the letters the caption font size.
Assembling them here instead lets the panels be packed tightly and the letters
be drawn as large as the figure needs, in the same Times face as everything
else in the section.

Each input is trimmed of its uniform background border, so a render with a lot
of empty margin does not push its neighbours apart. Panels are then scaled to a
common height and laid out on a grid.

Usage:
    python examples/compose_figure_row.py OUT.png IMG1 IMG2 ... [--ncols=3]
                                          [--labels=abcdef] [--gap=0.05]
                                          [--row-gap=0.04] [--label-gap=0.02]

All gaps are fractions of the panel height.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from foam_figure_style import apply_paper_style


# The composed image is included at \textwidth, so its letters are scaled by
# (textwidth / figure width). Sizing them from that keeps the printed letters
# the same across figures of different widths, which hand-picked source sizes
# do not: a three-panel row is scaled down far less than a five-panel one.
TEXTWIDTH_IN = 6.5
LABEL_PRINTED_PT = 13.0


def load_trimmed(path, tol=0.02, pad_frac=0.0):
    """Read an image, flatten onto white, and crop its uniform border.

    The background colour is taken from the top-left pixel, so this works for
    a white matplotlib export, a grey ParaView viewport and a transparent
    screenshot alike. `tol` is the per-channel difference, on a 0..1 scale, at
    which a pixel counts as content rather than background.
    """
    img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0

    corner = img[0, 0]
    ink = np.abs(img - corner).max(axis=2) > tol
    # Fully transparent pixels are background whatever their stored RGB is.
    ink &= img[..., 3] > 0.02

    if ink.any():
        rows = np.flatnonzero(ink.any(axis=1))
        cols = np.flatnonzero(ink.any(axis=0))
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        if pad_frac:
            pr = int(round(pad_frac * (r1 - r0)))
            pc = int(round(pad_frac * (c1 - c0)))
            r0, r1 = max(0, r0 - pr), min(img.shape[0], r1 + pr)
            c0, c1 = max(0, c0 - pc), min(img.shape[1], c1 + pc)
        img = img[r0:r1, c0:c1]

    # Flatten any remaining transparency onto white so the composed figure has
    # one uniform background.
    alpha = img[..., 3:4]
    return img[..., :3] * alpha + (1.0 - alpha)


def compose_grid(image_paths, out_path, ncols=None, labels=None,
                 panel_height=3.4, gap=0.05, row_gap=0.04, label_gap=0.015,
                 label_fontsize=None, dpi=220, tol=0.02,
                 scales=None, column_labels=False):
    """Lay trimmed panels out on a grid and write one image.

    Args:
        image_paths: panel images, in reading order.
        out_path: destination PNG.
        ncols: panels per row (default: all on one row).
        labels: e.g. "abcdef"; pass None or "" for no letters.
        panel_height: height of one panel, in inches, before labels.
        gap: horizontal space between panels, as a fraction of panel_height.
        row_gap: vertical space between a row's letters and the next row.
        label_gap: space between a panel and its letter.
        label_fontsize: size of the "(a)" text. Left as None it is derived
            from the figure width so the letters print at LABEL_PRINTED_PT.
        column_labels: label the COLUMNS once, beneath the whole grid, rather
            than labelling every panel. Use when the rows are two views of the
            same three cases, so the letters identify the case and the caption
            explains what each row shows.
        scales: optional per-panel multiplier on the panel height. Equal
            heights are the right default for panels showing the same kind of
            object, but a panel whose content is a single large object next to
            panels showing many small ones needs to be drawn smaller for the
            objects themselves to read at a comparable size.
    """
    apply_paper_style()

    panels = [load_trimmed(p, tol=tol) for p in image_paths]
    n = len(panels)
    ncols = n if ncols is None else int(ncols)
    nrows = (n + ncols - 1) // ncols

    labels = list(labels) if labels else []
    has_labels = bool(labels)

    scales = [1.0] * n if scales is None else list(scales)
    if len(scales) != n:
        raise ValueError(f"scales has {len(scales)} entries for {n} panels")

    # Panel rectangles are snapped to whole output pixels. Without this the
    # two rows of a grid land on the pixel grid with a fractional offset, and
    # the same object rendered twice resamples slightly differently -- a
    # sub-pixel misregistration along every silhouette edge.
    def snap(inches):
        return round(inches * dpi) / dpi

    heights = [snap(panel_height * s) for s in scales]
    widths = [snap(h * (p.shape[1] / p.shape[0]))
              for h, p in zip(heights, panels)]

    gap_in = snap(gap * panel_height)
    label_gap_in = snap(label_gap * panel_height)
    row_gap_in = snap(row_gap * panel_height)

    rows = [list(range(r * ncols, min((r + 1) * ncols, n)))
            for r in range(nrows)]

    # On a multi-row grid the columns are given a common width, so panels
    # line up vertically even when two screenshots of the same object were
    # cropped to slightly different aspect ratios.
    col_widths = [max(widths[i] for idx in rows for i in idx
                      if i - idx[0] == c)
                  for c in range(ncols)] if nrows > 1 else None

    def slot_width(i, idx):
        return widths[i] if col_widths is None else col_widths[i - idx[0]]

    row_widths = [sum(slot_width(i, idx) for i in idx)
                  + gap_in * (len(idx) - 1) for idx in rows]

    fig_w = max(row_widths)

    if label_fontsize is None:
        label_fontsize = LABEL_PRINTED_PT * fig_w / TEXTWIDTH_IN
    # Room for one line of letters. A point is 1/72 in; 1.15x leading is ample
    # because the letters are set on a single line.
    label_strip_in = snap(label_fontsize / 72.0 * 1.15 + label_gap_in) \
        if has_labels else 0.0

    # A row is as tall as its tallest panel, plus the strip for its letters.
    # With column labels there is only one strip, under the bottom row.
    strips = [label_strip_in if (not column_labels or r == nrows - 1) else 0.0
              for r in range(nrows)]
    row_panel_h = [max(heights[i] for i in idx) for idx in rows]
    bands = [h + s for h, s in zip(row_panel_h, strips)]
    fig_h = sum(bands) + row_gap_in * (nrows - 1)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Figure coordinates measure up from the bottom; rows read downwards.
    band_top = fig_h
    for r, idx in enumerate(rows):
        # Rows with fewer panels than the widest are centred.
        cursor = 0.5 * (fig_w - row_widths[r])
        band_bottom = band_top - bands[r]
        label_y = band_bottom + strips[r] - label_gap_in
        last_row = r == nrows - 1
        for c, i in enumerate(idx):
            w, h = widths[i], heights[i]
            slot = slot_width(i, idx)
            x0 = cursor + 0.5 * (slot - w)
            # Panels shorter than the row are centred within it, so a row of
            # unequal panels reads as one band rather than a ragged line.
            y0 = band_bottom + strips[r] + 0.5 * (row_panel_h[r] - h)
            ax = fig.add_axes([x0 / fig_w, y0 / fig_h,
                               w / fig_w, h / fig_h])
            ax.imshow(panels[i], interpolation="antialiased")
            ax.set_axis_off()
            letter = c if column_labels else i
            if has_labels and letter < len(labels) and \
                    (last_row or not column_labels):
                fig.text((cursor + 0.5 * slot) / fig_w, label_y / fig_h,
                         f"({labels[letter]})", fontsize=label_fontsize,
                         ha="center", va="top")
            cursor += slot + gap_in
        band_top = band_bottom - row_gap_in

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # No tight bounding box: the layout above is already exact, and trimming
    # would silently undo the deliberate margins.
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}  ({nrows}x{ncols}, {n} panels)")
    return out_path


def _flag(prefix, default):
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return a[len(prefix):]
    return default


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(positional) < 2:
        raise SystemExit(__doc__)

    out_path, image_paths = positional[0], positional[1:]
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing input image(s):\n  " + "\n  ".join(missing))

    ncols = _flag("--ncols=", None)
    compose_grid(
        image_paths, out_path,
        ncols=int(ncols) if ncols else None,
        labels=_flag("--labels=", "abcdefghij")[:len(image_paths)],
        panel_height=float(_flag("--panel-height=", "3.4")),
        gap=float(_flag("--gap=", "0.05")),
        row_gap=float(_flag("--row-gap=", "0.04")),
        label_gap=float(_flag("--label-gap=", "0.015")),
        label_fontsize=(float(_flag("--label-size=", "")) or None
                        if _flag("--label-size=", "") else None),
        tol=float(_flag("--tol=", "0.02")),
        scales=([float(s) for s in _flag("--scales=", "").split(",")]
                if _flag("--scales=", "") else None),
    )


if __name__ == "__main__":
    main()
