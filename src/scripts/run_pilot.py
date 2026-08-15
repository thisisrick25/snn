"""Pilot runner: naive Split-MNIST LIF-SNN sparsity sweep.

The pilot sweeps a FIXED set of global firing thresholds. Spike activity is a
MEASURED OUTCOME of each threshold, not a target we calibrate toward: we set the
threshold, run continual learning, and record whatever activity results. This is
both simpler and more honest than the old calibrate-to-a-target-activity design,
because observed activity drifts under continued training and the analysis has
always keyed on observed activity anyway.

For every (seed, threshold) condition:
  1. seed everything
  2. build Split-MNIST loaders
  3. build a model with the fixed threshold and freeze it
  4. run naive sequential continual learning over all 5 tasks
  5. compute forgetting/accuracy metrics
  6. extract hidden-layer-2 representations per task on a FIXED probe subset
  7. compute mean pairwise overlap (CKA + cosine)
  8. flag degenerate (dead-network) conditions
  9. persist a ConditionRecord as JSON

Usage (from repo root):
  python -m src.scripts.run_pilot [--config configs/pilot.yaml] [--seeds 0]
                                  [--thresholds 5.0] [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys


def _resolve_thread_cap() -> int | None:
    for i, arg in enumerate(sys.argv):
        if arg == "--threads" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--threads="):
            return int(arg.split("=", 1)[1])
    return None


# Thread-limiting env vars must be set BEFORE torch is imported (which happens
# transitively via the src.* imports below), otherwise the underlying BLAS/OpenMP
# pools are already sized to all cores. A CLI --threads value takes precedence;
# otherwise we defer to torch.set_num_threads in main() using the config value.
_cli_threads = _resolve_thread_cap()
if _cli_threads and _cli_threads > 0:
    os.environ["OMP_NUM_THREADS"] = str(_cli_threads)
    os.environ["MKL_NUM_THREADS"] = str(_cli_threads)

import yaml

import torch

from src.training.seeds import set_seed
from src.data.split_mnist import build_split_mnist
from src.data.split_cifar import build_split_cifar
from src.models.lif_snn import build_model
from src.models.conv_snn import build_conv_model
from src.training.continual import run_naive_sequential
from src.analysis.metrics import compute_metrics
from src.analysis.representations import fixed_subset_loader, extract_representation
from src.analysis.overlap import mean_pairwise_overlap
from src.analysis.io import (
    ConditionRecord,
    ensure_dirs,
    save_condition,
    load_conditions,
    save_summary_csv,
)

# A condition is treated as a degenerate "dead network" (excluded from mechanism
# analysis) when the network essentially stops firing and accuracy collapses to
# chance. These are pre-declared data-quality boundaries, not usable data points.
_DEAD_ACTIVITY = 1e-3      # mean observed activity at/below this = effectively silent
_CHANCE_ACCURACY = 0.55    # binary tasks; final accuracy at/below this = ~chance


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_device(cli_device: str | None, cfg: dict) -> str:
    value = cli_device if cli_device is not None else cfg.get("device", "auto")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _mean(xs: list[float]) -> float:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else float("nan")


_INPUT_DIM = {"mnist": 784, "cifar10": 3072}


def build_dataset(cfg: dict, tasks, batch_size: int, conv: bool = False):
    """Return (train_loaders, test_loaders) for the configured dataset.

    ``download=True`` so the dataset is fetched on first use; torchvision skips
    the download when the cache already exists. ``conv=True`` makes the CIFAR
    loader keep images as [3, 32, 32] (unflattened) for the conv frontend.
    """
    name = cfg.get("dataset", "mnist")
    root = cfg["data_root"]
    if name == "mnist":
        return build_split_mnist(root=root, tasks=tasks, batch_size=batch_size, download=True)
    if name == "cifar10":
        return build_split_cifar(root=root, tasks=tasks, batch_size=batch_size, download=True, conv=conv)
    raise ValueError(f"unknown dataset {name!r}; expected 'mnist' or 'cifar10'")


def run_condition(cfg: dict, seed: int, condition: float, *, probe_samples: int,
                  epochs: int, device: str = "cpu") -> ConditionRecord:
    timesteps = int(cfg["timesteps"])
    lr = float(cfg["lr"])
    batch_size = int(cfg["batch_size"])
    tasks = [tuple(t) for t in cfg["tasks"]]
    arch = cfg.get("arch", "mlp")
    is_conv = arch == "conv_snn"

    # The MLP path derives fc1 input size from the dataset; the conv frontend
    # fixes its own feature dim so input_dim is irrelevant there.
    if not is_conv:
        cfg["input_dim"] = _INPUT_DIM[cfg.get("dataset", "mnist")]

    # 1-2. seed + data (conv keeps CIFAR images unflattened as [3, 32, 32])
    set_seed(seed)
    train_loaders, test_loaders = build_dataset(cfg, tasks, batch_size, conv=is_conv)

    # 3. build model for this sparsity condition. In kwta_window mode `condition`
    # is a target activity fraction (the builder derives k per layer); in threshold
    # mode it is the global firing threshold. set_threshold only applies to the
    # latter (in kwta mode the threshold is left at its default and k gates firing).
    model = build_conv_model(cfg, condition) if is_conv else build_model(cfg, condition)
    if cfg.get("sparsity_mode", "threshold") == "threshold":
        model.set_threshold(condition)
    model.to(device)

    # 4. naive sequential CL
    result = run_naive_sequential(
        model, train_loaders, test_loaders,
        epochs=epochs, lr=lr, timesteps=timesteps, device=device,
    )

    # 5. metrics
    metrics = compute_metrics(result.accuracy_matrix)

    # 6-7. representations + overlap on a fixed probe subset per task
    reps = []
    for j in range(len(test_loaders)):
        probe = fixed_subset_loader(test_loaders[j].dataset, probe_samples, batch_size)
        reps.append(extract_representation(model, probe, task_id=j, device=device))
    overlap_cka = mean_pairwise_overlap(reps, measure="cka")
    overlap_cosine = mean_pairwise_overlap(reps, measure="cosine")

    # 8. degenerate-condition flag
    mean_activity = _mean(result.observed_activity_per_task)
    dead_network = (
        mean_activity <= _DEAD_ACTIVITY
        or metrics.final_avg_accuracy <= _CHANCE_ACCURACY
    )

    return ConditionRecord(
        seed=seed,
        threshold=condition,
        observed_activity_per_task=result.observed_activity_per_task,
        mean_observed_activity=mean_activity,
        accuracy_matrix=result.accuracy_matrix,
        final_avg_accuracy=metrics.final_avg_accuracy,
        per_task_forgetting=metrics.per_task_forgetting,
        mean_forgetting=metrics.mean_forgetting,
        overlap_cka=overlap_cka,
        overlap_cosine=overlap_cosine,
        dead_network=dead_network,
        dataset=cfg.get("dataset", "mnist"),
        arch=arch,
        train_losses=result.train_losses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Split-MNIST LIF-SNN sparsity pilot.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="override seeds (default: from config)")
    parser.add_argument("--conditions", type=float, nargs="*", default=None,
                        help="override the swept sparsity conditions: kwta target "
                             "fractions in kwta_window mode, thresholds in threshold mode")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None,
                        help="alias of --conditions (kept for back-compat)")
    parser.add_argument("--probe-samples", type=int, default=512,
                        help="fixed probe subset size per task for overlap")
    parser.add_argument("--quick", action="store_true",
                        help="1 epoch/task for a fast dry run")
    parser.add_argument("--threads", type=int, default=None,
                        help="cap CPU threads (default: config num_threads; 0 = all cores)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None,
                        help="compute device (default: config device; auto = cuda if available)")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None,
                        help="benchmark dataset (default: config dataset)")
    parser.add_argument("--arch", choices=["mlp", "conv_snn"], default=None,
                        help="model architecture (default: config arch; conv_snn needs cifar10)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    print(f"[cfg] dataset = {cfg.get('dataset', 'mnist')}", flush=True)

    if args.arch is not None:
        cfg["arch"] = args.arch
    print(f"[cfg] arch = {cfg.get('arch', 'mlp')}", flush=True)

    device = resolve_device(args.device, cfg)
    print(f"[cfg] device = {device}", flush=True)

    n_threads = args.threads if args.threads is not None else int(cfg.get("num_threads", 0))
    if n_threads and n_threads > 0:
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass
        print(f"[cfg] torch threads capped at {n_threads}", flush=True)
    seeds = args.seeds if args.seeds is not None else [int(s) for s in cfg["seeds"]]

    mode = cfg.get("sparsity_mode", "threshold")
    cli_conditions = args.conditions if args.conditions is not None else args.thresholds
    if cli_conditions is not None:
        conditions = cli_conditions
    elif mode == "kwta_window":
        conditions = [float(f) for f in cfg["kwta_fractions"]]
    else:
        conditions = [float(t) for t in cfg["thresholds"]]
    label = "frac" if mode == "kwta_window" else "theta"
    print(f"[cfg] sparsity_mode = {mode}", flush=True)
    epochs = 1 if args.quick else int(cfg["epochs_per_task"])

    dirs = ensure_dirs(cfg["results_dir"])

    for seed in seeds:
        for condition in conditions:
            print(f"[run] seed={seed} {label}={condition:.3g} epochs={epochs}", flush=True)
            record = run_condition(
                cfg, seed, condition,
                probe_samples=args.probe_samples,
                epochs=epochs,
                device=device,
            )
            path = save_condition(dirs["raw"], record)
            flag = " DEAD" if record.dead_network else ""
            print(
                f"  -> obs_act={record.mean_observed_activity:.3f} "
                f"final_acc={record.final_avg_accuracy:.3f} "
                f"forget={record.mean_forgetting:.3f} "
                f"cka={record.overlap_cka:.3f}{flag} saved={os.path.basename(path)}",
                flush=True,
            )

    records = load_conditions(dirs["raw"])
    csv_path = save_summary_csv(dirs["metrics"], records)
    print(f"[done] {len(records)} conditions; summary -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
