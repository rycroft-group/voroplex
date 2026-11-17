"""
Redraw the combined 2D row from a cached training result.

The 2D run writes `combined_row_data.npz` beside its panels, holding the
generator positions, the two foam snapshots, the loss history and the target
tensor. This script reloads that archive and re-runs only the drawing code, so
the layout can be tuned without repeating the optimisation.

Usage:
    python examples/replot_combined_2d.py <panel_dir> [output.png]
"""

import importlib
import sys
from pathlib import Path

import numpy as np

foam2d = importlib.import_module("16_2d_foam_orientation_optimization")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)

    panel_dir = Path(args[0])
    archive = panel_dir if panel_dir.suffix == ".npz" else \
        panel_dir / "combined_row_data.npz"
    if not archive.exists():
        raise SystemExit(f"No cache at {archive}. Re-run the 2D script with "
                         f"--panel-dir to create it.")

    data = np.load(archive)
    out_path = Path(args[1]) if len(args) > 1 else \
        archive.parent / "combined_row.png"

    def group(prefix):
        return {k[len(prefix):]: data[k] for k in data.files
                if k.startswith(prefix)}

    foam2d.save_combined_2d(
        out_path,
        data["initial_points"], data["final_points"],
        group("init_"), group("final_"), group("hist_"),
        data["target_A"], data["u_target"],
    )


if __name__ == "__main__":
    main()
