"""Run the exploratory mediation analysis over saved pilot results.

Loads the per-condition summary (activity, CKA overlap, forgetting), fits the
activity -> overlap -> forgetting mediation model, prints a report, and saves
results/pilot/metrics/mediation.json.

This is an EXPLORATORY screen at pilot scale (few conditions x few seeds), not a
confirmatory test. Its job is to indicate whether overlap plausibly mediates the
sparsity-forgetting link or whether the two merely co-vary with activity; the
confirmatory version needs more seeds (full study).

Usage (from repo root):
  python -m src.scripts.run_mediation [--config configs/pilot.yaml]
                                      [--dataset mnist|cifar10]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import yaml

from src.analysis.mediation import mediation_analysis


def _load_rows(summary_path: str) -> list[dict]:
    with open(summary_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _verdict(res) -> str:
    mediated = res.indirect_ci_low > 0 or res.indirect_ci_high < 0
    shrinks = abs(res.direct_effect_c_prime) < abs(res.total_effect_c)
    if mediated and shrinks:
        return "CONSISTENT WITH MEDIATION (indirect effect nonzero; direct effect shrinks)"
    if mediated:
        return "PARTIAL: indirect effect nonzero but direct effect did not shrink"
    return "NO EVIDENCE of mediation beyond activity co-variation (indirect CI includes 0)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory mediation analysis over pilot results.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--dataset", default=None, help="filter to one dataset (mnist|cifar10)")
    parser.add_argument("--arch", default=None, help="filter to one arch (mlp|conv_snn)")
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    metrics_dir = os.path.join(cfg["results_dir"], "metrics")
    summary_path = os.path.join(metrics_dir, "summary.csv")
    if not os.path.isfile(summary_path):
        print(f"[mediation] no summary at {summary_path}; run the pilot first.")
        return

    rows = _load_rows(summary_path)
    rows = [r for r in rows if r.get("dead_network", "False") != "True"]
    if args.dataset is not None:
        rows = [r for r in rows if r.get("dataset", "mnist") == args.dataset]
    if args.arch is not None:
        rows = [r for r in rows if r.get("arch", "mlp") == args.arch]
    if len(rows) < 3:
        print(f"[mediation] only {len(rows)} live conditions; need >= 3.")
        return

    activity = np.array([float(r["mean_observed_activity"]) for r in rows])
    overlap = np.array([float(r["overlap_cka"]) for r in rows])
    forgetting = np.array([float(r["mean_forgetting"]) for r in rows])

    res = mediation_analysis(activity, overlap, forgetting, n_boot=args.n_boot)

    print(f"\n=== Mediation: activity -> overlap(CKA) -> forgetting ===")
    print(f"EXPLORATORY screen (n = {res.n} conditions; pilot scale, not confirmatory)")
    print(f"  total effect   c  (activity->forgetting)      = {res.total_effect_c:+.3f}")
    print(f"  a-path         a  (activity->overlap)          = {res.a_path:+.3f}")
    print(f"  b-path         b  (overlap->forgetting|activity)= {res.b_path:+.3f}")
    print(f"  direct effect  c' (activity->forgetting|overlap)= {res.direct_effect_c_prime:+.3f}")
    print(f"  indirect       a*b                              = {res.indirect_effect:+.3f}")
    print(f"                 95% bootstrap CI = [{res.indirect_ci_low:+.3f}, {res.indirect_ci_high:+.3f}]")
    print(f"  proportion mediated (a*b / c)                   = {res.proportion_mediated:.3f}")
    print(f"  VERDICT: {_verdict(res)}\n")

    out_path = os.path.join(metrics_dir, "mediation.json")
    payload = {
        "exploratory": True,
        "note": "pilot-scale screen, not confirmatory; needs more seeds for a confirmatory estimate",
        "dataset_filter": args.dataset,
        "arch_filter": args.arch,
        "n": res.n,
        "total_effect_c": res.total_effect_c,
        "a_path": res.a_path,
        "b_path": res.b_path,
        "direct_effect_c_prime": res.direct_effect_c_prime,
        "indirect_effect": res.indirect_effect,
        "indirect_ci": [res.indirect_ci_low, res.indirect_ci_high],
        "proportion_mediated": res.proportion_mediated,
        "verdict": _verdict(res),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[mediation] saved -> {out_path}")


if __name__ == "__main__":
    main()
