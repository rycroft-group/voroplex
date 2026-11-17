"""
Compose the six renders of the optimized 3D foams into one figure.

Assembling them as LaTeX subfigures forces wide gutters, a large vertical gap
between the two rows, and letters set at the caption size. This builds the 3x2
grid directly instead: each render is cropped to its content -- so screenshots
taken at different window sizes still line up -- scaled to a common height, and
labelled (a)..(f) in the same Times face as the rest of the section.

The two rows are two views of the SAME three cases, so the letters label the
columns once, beneath the grid, and the caption says what each row shows:

    lamellar            columnar            radial            <- whole cube
    lamellar cut        columnar cut        radial cut        <- cut away
       (a)                 (b)                 (c)

Usage:
    python examples/make_foam_3d_grid.py [dir] [output.png] [--render] [--rule=R]

`dir` defaults to "examples/5.3(v3)" and must contain
lamellar.png, needle.png, circular.png and the matching *_inside.png.
With --render those six are produced first, from the saved run histories in
`dir`/{uniaxial,needle,circular}/summary_history.npz.
"""

import sys
from pathlib import Path

from compose_figure_row import compose_grid


STEMS = ["lamellar", "needle", "circular"]
# Which run directory each stem is rendered from. The lamellar run lives under
# "uniaxial", the name the optimizer uses for a constant target axis.
RUN_DIR = {"lamellar": "uniaxial", "needle": "needle", "circular": "circular"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else Path("examples/5.3(v3)")
    out = Path(args[1]) if len(args) > 1 else src / "foam_3d_grid.png"

    names = [f"{s}.png" for s in STEMS] + [f"{s}_inside.png" for s in STEMS]
    paths = [src / n for n in names]

    if "--render" in sys.argv:
        from render_foam_pyvista import render_run, RULE_DEFAULT
        rule = RULE_DEFAULT
        for a in sys.argv[1:]:
            if a.startswith("--rule="):
                rule = a[len("--rule="):]
        for stem in STEMS:
            render_run(src / RUN_DIR[stem] / "summary_history.npz", src, stem,
                       rule=rule)

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing render(s):\n  " + "\n  ".join(str(p) for p in missing) +
            "\n\nRe-run with --render to produce them.")

    compose_grid(
        paths, out, ncols=3, labels="abc", column_labels=True,
        panel_height=3.2,
        # Equal gutters: a 3x2 grid of the same object reads as skewed when
        # the rows sit closer together than the columns.
        gap=0.035,
        row_gap=0.035,
        label_gap=0.015,
    )


if __name__ == "__main__":
    main()
