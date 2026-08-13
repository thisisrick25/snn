"""Regenerate the five pilot figures from saved results (no retraining).

Usage (from repo root):
  python -m src.scripts.make_plots [--config configs/pilot.yaml]
"""

from __future__ import annotations

import argparse

import yaml

from src.analysis.io import ensure_dirs, load_conditions
from src.analysis.plotting import make_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate pilot plots from saved results.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    dirs = ensure_dirs(cfg["results_dir"])
    records = load_conditions(dirs["raw"])
    if not records:
        print(f"[make_plots] no results found in {dirs['raw']}; run run_pilot first.")
        return
    paths = make_all_plots(records, dirs["plots"])
    print(f"[make_plots] wrote {len(paths)} figures to {dirs['plots']}")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
