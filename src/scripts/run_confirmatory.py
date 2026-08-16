from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from src.analysis.confirmatory_report import confirmatory_json
from src.analysis.io import ensure_dirs, load_conditions, save_summary_csv
from src.analysis.mediation import confirmatory_mediation_analysis


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("dead_network") != "True"]


def _results_dir(config_path: Path) -> str:
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    return str(cfg["results_dir"])


def _ensure_summary(results_dir: str) -> Path:
    """Return the summary.csv path, rebuilding it from raw/*.json when absent.

    run_full_study.sh may skip every already-complete cell, so run_pilot (the
    only writer of summary.csv) never runs; without this rebuild the confirmatory
    step would abort on a missing summary even though the raw records exist.
    """
    dirs = ensure_dirs(results_dir)
    summary_path = Path(dirs["metrics"]) / "summary.csv"
    if not summary_path.exists():
        records = load_conditions(dirs["raw"])
        if records:
            save_summary_csv(dirs["metrics"], records)
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run confirmatory mechanism-moderated mediation analysis.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--mediator", choices=["cka", "cosine"], default="cka")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    summary_path = _ensure_summary(_results_dir(Path(args.config)))
    if not summary_path.exists():
        print(f"[confirmatory] no summary at {summary_path} and no raw records; run the pilot first.")
        return 1

    rows = _load_rows(summary_path)
    primary = confirmatory_mediation_analysis(rows, mediator=args.mediator, n_boot=args.n_boot, random_state=args.random_state)
    cosine = None
    if args.mediator == "cka":
        cosine = confirmatory_mediation_analysis(rows, mediator="cosine", n_boot=args.n_boot, random_state=args.random_state + 1)
    payload = confirmatory_json(primary, cosine)
    output_path = summary_path.parent / "confirmatory.json"
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"[confirmatory] wrote {output_path}")
    for setting, result in primary.settings.items():
        raw_flags = result.estimates.diagnostics.get("flags", [])
        flags = [str(item) for item in raw_flags] if isinstance(raw_flags, list) else []
        if result.bootstrap.high_failure_rate:
            flags.append("high_bootstrap_failure_rate")
        print(f"[confirmatory] {setting}: n={result.n_rows}, seeds={result.n_seed_clusters}, flags={flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
