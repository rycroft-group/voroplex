"""
Regenerate a 3D foam summary figure from a saved training history.

Each production run writes `<figure>_history.npz` alongside its PNG. This
script reloads that archive and redraws the figure, so layout and styling can
be iterated on without repeating a ~30 minute optimisation. Only the slice
panel needs geometry, and that is a single tessellation of the stored final
generator positions.

Usage:
    python examples/replot_foam_3d.py <history.npz> [output.png] [--target=MODE]

If the output path is omitted, the figure is written next to the archive, at
the path obtained by stripping the "_history" suffix. The target mode is
inferred from the filename when not given explicitly; it selects the slice axis
and whether the tensor panel shows a difference or an absolute tensor.
"""

import importlib
import sys
from pathlib import Path

import numpy as np

foam3d = importlib.import_module("17_3d_foam_orientation_optimization")


def infer_mode(path):
    """Guess the target mode from the archive's path.

    Production runs are named `<mode>/summary_history.npz`, so the mode is in
    the directory rather than the filename; earlier runs put it in the
    filename. Matching against the whole path covers both, and getting this
    wrong silently picks the wrong slice axis and tensor panel.
    """
    lowered = str(path).lower()
    for mode in ("circular", "needle", "planar", "uniaxial"):
        if mode in lowered:
            return mode
    return "uniaxial"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)

    history_path = Path(args[0])
    if not history_path.exists():
        raise SystemExit(f"No such history archive: {history_path}")

    if len(args) > 1:
        out_path = Path(args[1])
    else:
        stem = history_path.name.replace("_history.npz", ".png")
        out_path = history_path.parent / stem

    mode = foam3d.parse_flag_value("--target=", None) or infer_mode(history_path)
    threads = int(foam3d.parse_flag_value("--threads=", "4"))

    data = np.load(history_path)
    history = {k[len("hist_"):]: data[k] for k in data.files
               if k.startswith("hist_")}

    print(f"Replotting {history_path.name}  (mode={mode})")
    print(f"  iterations : {len(history['loss'])}")
    print(f"  generators : {data['final_points'].shape[0]}")

    tensor_panel = "absolute" if mode == "circular" else "difference"

    foam3d.save_summary_figure_3d(
        initial_points=data["initial_points"],
        final_points=data["final_points"],
        history=history,
        output_path=str(out_path),
        u_target_np=data["u_target"],
        limits=[[0, 1], [0, 1], [0, 1]],
        threads=threads,
        slice_axis=foam3d.get_slice_axis(mode),
    )
    print(f"  wrote      : {out_path}")

    # Optionally also emit every panel as a standalone image for LaTeX
    # subfigure composition.
    panel_dir = foam3d.parse_flag_value("--panel-dir=", None)
    if panel_dir is not None:
        foam3d.save_panels_3d(
            out_dir=panel_dir,
            initial_points=data["initial_points"],
            final_points=data["final_points"],
            history=history,
            A_target_np=data["A_target"],
            A_final_np=data["A_final"],
            u_target_np=data["u_target"],
            limits=[[0, 1], [0, 1], [0, 1]],
            threads=threads,
            slice_axis=foam3d.get_slice_axis(mode),
            tensor_panel=tensor_panel,
        )


if __name__ == "__main__":
    main()
