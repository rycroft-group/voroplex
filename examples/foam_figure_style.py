"""
Shared figure styling for the anisotropic-foam examples (Sec. 5.3 of the paper).

Design rules encoded here:

  * No titles inside the axes. Each subplot carries a bold "(a)", "(b)", ...
    label instead, and the description lives in the LaTeX figure caption.
  * Fonts are large. A figure authored at ~15 in wide and included at
    \\textwidth (~6.5 in) is scaled by ~0.43, so a 20 pt source label prints
    at ~8.6 pt -- comparable to a caption. Anything smaller than this in the
    source becomes unreadable on the page.
  * Target-vs-achieved tensors are shown as a SINGLE difference heatmap
    (A_tgt - A) rather than two separate matrices.

Import from the example scripts, which are in the same directory:

    from foam_figure_style import apply_paper_style, panel_label, ...
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Source-figure font sizes. See the module docstring for why these are large.
FONT_AXIS_LABEL = 20
FONT_TICK = 17
FONT_LEGEND = 17
FONT_PANEL_LABEL = 24
FONT_MATRIX_CELL = 18
# Letters drawn beneath a hand-composed row of panels. Larger than
# FONT_PANEL_LABEL because such a row is included at \textwidth as a single
# image, so it is scaled down further than an individual panel would be.
FONT_ROW_LABEL = 32


def apply_paper_style():
    """Set global rcParams for paper-quality foam figures. Call once before
    creating a figure.

    Everything is set in a Times face to match the body text of the paper:
    axis labels, tick numbers, legends and any annotation. Mathematical text
    uses the STIX set, which is metrically designed to sit alongside Times.
    Nothing is emboldened -- bold is reserved for vectors and matrices, which
    are typeset by LaTeX in the caption and body rather than drawn here.
    """
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_AXIS_LABEL,
        "axes.titlesize": FONT_AXIS_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.4,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
    })


def panel_label(ax, letter, dx=-0.16, dy=1.04, is_3d=False):
    """Put a bold "(a)"-style label above the top-left corner of `ax`.

    Panels are lettered and described in the caption rather than titled
    inside the figure.
    """
    # Not emboldened: bold is reserved for vectors and matrices.
    kwargs = dict(fontsize=FONT_PANEL_LABEL, va="bottom", ha="left")
    if is_3d:
        # A 3D axes draws its content inset within its bounding box, so the
        # default offsets would push the label far from the plot. Axes3D also
        # needs text2D for figure-relative placement.
        ax.text2D(0.02, 0.90, f"({letter})", transform=ax.transAxes, **kwargs)
    else:
        ax.text(dx, dy, f"({letter})", transform=ax.transAxes, **kwargs)


def draw_tensor_difference(ax, A_target, A_achieved, axis_labels,
                           cbar=True, fig=None, cell_fontsize=None):
    """Single heatmap of the signed mismatch D = A_tgt - A.

    Condenses what used to be two separate "target A" / "achieved A" heatmaps
    into one panel. The colour scale is diverging and symmetric about zero, so
    a perfectly matched tensor renders as a uniform white block. Each entry is
    annotated with its signed difference.

    Args:
        ax: target axes.
        A_target, A_achieved: (d, d) ndarrays.
        axis_labels: e.g. ["x", "y"] or ["x", "y", "z"].
        cbar: whether to attach a colourbar (needs `fig`).
        fig: parent figure, required when cbar=True.

    Returns the image handle.
    """
    D = np.asarray(A_target, dtype=float) - np.asarray(A_achieved, dtype=float)
    d = D.shape[0]
    # Symmetric scale so that white == exact match. The scale must track the
    # ACTUAL magnitude of the mismatch: flooring it (an earlier version pinned
    # it at 1e-3) silently rescales a 1e-9 residual to zero and the panel then
    # reports "+0.00" everywhere, hiding the very result it exists to show.
    vmax = float(np.abs(D).max())
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-12
    im = ax.imshow(D, vmin=-vmax, vmax=vmax, cmap="RdBu_r")

    fs = FONT_MATRIX_CELL if cell_fontsize is None else cell_fontsize
    # When the match is very close, three decimals would print every entry as
    # "+0.000" and the panel would carry no information. Factor out a common
    # power of ten instead: the entries stay short enough to fit inside their
    # cells, and the scale is stated once on the colourbar.
    sci = vmax < 5e-3
    exp = int(np.floor(np.log10(vmax))) if sci else 0
    scale = 10.0 ** exp

    def fmt(v):
        return f"{v / scale:+.2f}" if sci else f"{v:+.3f}"

    for i in range(d):
        for j in range(d):
            val = D[i, j]
            # Pick a legible text colour against the local background.
            shade = abs(val) / vmax
            colour = "white" if shade > 0.6 else "black"
            ax.text(j, i, fmt(val), ha="center", va="center",
                    color=colour, fontsize=fs)

    ax.set_xticks(range(d), labels=axis_labels)
    ax.set_yticks(range(d), labels=axis_labels)
    ax.tick_params(length=0)

    if cbar and fig is not None:
        # Horizontal bar underneath the matrix. A vertical bar on the right
        # collides with the y-axis label of the panel that follows it,
        # especially in the single-row 2D layout.
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          location="bottom", fraction=0.055, pad=0.14)
        cb.ax.tick_params(labelsize=FONT_TICK - 4)
        cb.locator = matplotlib.ticker.MaxNLocator(nbins=3, symmetric=True)
        label = r"$A^{\mathrm{tgt}}_{ij}-A_{ij}$"
        if sci:
            # Tick in the same factored-out units as the cell annotations, so
            # the bar and the matrix cannot disagree.
            cb.formatter = matplotlib.ticker.FuncFormatter(
                lambda v, _pos: f"{v / scale:.0f}")
            label += rf" ($\times 10^{{{exp}}}$)"
        cb.update_ticks()
        cb.set_label(label, fontsize=FONT_AXIS_LABEL - 4, labelpad=2)
    return im


def draw_tensor_absolute(ax, A, axis_labels, cbar=True, fig=None,
                         cell_fontsize=None, label=r"$A_{ij}$"):
    """Heatmap of a fabric tensor's entries on a fixed [0, 1] scale.

    Used in place of `draw_tensor_difference` when the optimization target is
    spatially varying and therefore has no single global tensor to difference
    against. The fixed scale makes isotropy (all diagonal entries 1/3) visually
    recognisable across figures.
    """
    A = np.asarray(A, dtype=float)
    d = A.shape[0]
    im = ax.imshow(A, vmin=0.0, vmax=1.0, cmap="magma")

    fs = FONT_MATRIX_CELL if cell_fontsize is None else cell_fontsize
    for i in range(d):
        for j in range(d):
            val = A[i, j]
            colour = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=colour, fontsize=fs)

    ax.set_xticks(range(d), labels=axis_labels)
    ax.set_yticks(range(d), labels=axis_labels)
    ax.tick_params(length=0)

    if cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          location="bottom", fraction=0.055, pad=0.14)
        cb.ax.tick_params(labelsize=FONT_TICK - 4)
        cb.locator = matplotlib.ticker.MaxNLocator(nbins=3)
        cb.update_ticks()
        cb.set_label(label, fontsize=FONT_AXIS_LABEL - 4, labelpad=2)
    return im


def log_ymin(*series):
    """Lower bound for a semilogy axis that keeps every supplied curve visible.

    Pass every curve that will be drawn; using only one of them clips the
    others when they decay to different plateaus.
    """
    lo = None
    for values in series:
        arr = np.asarray(values, dtype=np.float64)
        pos = arr[arr > 0]
        if len(pos):
            cand = max(float(pos.min()) * 0.8, float(np.finfo(np.float64).tiny))
            lo = cand if lo is None else min(lo, cand)
    return lo if lo is not None else 1e-16


def running_median(values, window=21):
    """Centred running median, with the window shrinking at the ends.

    Used to draw the trend of a loss curve. Optimising over a Voronoi
    tessellation produces a genuinely spiky loss -- every change in the
    tessellation's connectivity kicks it -- and once the loss has decayed to
    the scale of those kicks the raw curve is dominated by them. A median is
    preferred over a mean here because it ignores the isolated upward spikes
    rather than being dragged up by them.
    """
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n == 0:
        return y
    half = max(1, window // 2)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(y[lo:hi])
    return out


def plot_loss_curve(ax, iters, values, color, label, window=21):
    """Draw one loss curve as a faint raw trace plus a bold trend line.

    Showing both keeps the figure honest -- the reader sees the true
    iteration-to-iteration scatter -- while making the underlying trend, and
    in particular whether it has flattened, legible.
    """
    ax.semilogy(iters, values, color=color, alpha=0.22, linewidth=1.0,
                zorder=1)
    ax.semilogy(iters, running_median(values, window=window), color=color,
                linewidth=2.4, label=label, zorder=2)


def style_loss_axes(ax, *series):
    """Common styling for a semilogy loss panel."""
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.set_ylim(bottom=log_ymin(*series))
    ax.grid(True, alpha=0.25)


def new_panel(figsize=(4.4, 3.7), projection=None):
    """Create a single-axes figure for one standalone panel.

    The paper composes its multi-panel figures with LaTeX subfigures, which
    supply the (a)/(b)/(c) labels themselves, so panels written through this
    helper deliberately carry no in-figure letter and no title.
    """
    apply_paper_style()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    return fig, ax


def pad_image_to_aspect(path, aspect):
    """Pad a saved PNG with white until its width/height equals `aspect`.

    Panels are written with a tight bounding box, so their final proportions
    are set by their content: a square plot with no y axis comes out narrower
    than one with a colourbar underneath. Included side by side at a common
    width, they would then render at different heights. Padding equalises the
    aspect ratio without rescaling any content, so the plots themselves stay
    the same size and only the surrounding margin changes.

    Padding is symmetric, and only ever added -- never cropped.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        cur = w / h
        if abs(cur - aspect) < 1e-3:
            return
        if cur < aspect:                 # too narrow: widen
            new_w, new_h = int(round(h * aspect)), h
        else:                            # too wide: heighten
            new_w, new_h = w, int(round(w / aspect))
        canvas = Image.new("RGB", (new_w, new_h), "white")
        canvas.paste(im, ((new_w - w) // 2, (new_h - h) // 2))
        canvas.save(path)


def save_panel(fig, out_dir, name, dpi=220, pad_aspect=None):
    """Write one panel to `out_dir/name.png`, creating the directory.

    `pad_aspect` pads the result to a fixed width/height ratio so that a row
    of panels renders at uniform height; see `pad_image_to_aspect`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if pad_aspect is not None:
        pad_image_to_aspect(path, pad_aspect)
    return path


def hide_shared_y(ax):
    """Blank the y tick labels and y axis label on a panel that shares its
    vertical axis with the panel to its left."""
    ax.set_ylabel("")
    ax.tick_params(labelleft=False)
