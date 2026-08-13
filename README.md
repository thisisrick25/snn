# Spike Sparsity and Catastrophic Forgetting in Continual-Learning SNNs

Research code investigating whether spike sparsity reduces catastrophic forgetting in
continual-learning spiking neural networks (SNNs), and whether the effect is mediated by
reduced representational overlap between tasks. This repository contains the **pilot**: a
naive Split-MNIST LIF-SNN sparsity sweep used to screen the mechanism before the full study.

## Setup

Requires Python 3.11+ (developed on 3.14).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux
python -m pip install -r requirements.txt
```

`requirements.txt` pins a **CUDA (GPU) build** of torch (`torch==2.13.0+cu132`,
`torchvision==0.28.0+cu132`) via an `--extra-index-url` to the PyTorch cu132 wheel index.
This targets recent NVIDIA GPUs (including Blackwell, e.g. RTX 50-series). The install is
~700 MB.

**CPU-only install** (no NVIDIA GPU, or you prefer CPU): edit `requirements.txt` — remove the
`--extra-index-url` line and change the pins back to `torch==2.12.0` and `torchvision==0.27.0`.
The pilot runs fine on CPU (see the thread-cap note below).

The MNIST dataset downloads automatically on the first run into `./data`. If it is already
present, the pilot runs offline (`download=False`).

### Verifying the GPU (after a CUDA install)

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(2000,2000,device='cuda'); print((x@x).sum().item())"
```

The matmul must complete without error. If it prints a `no kernel image for sm_120` (or
similar) error, the wheel lacks kernels for your GPU — install a nightly cu132/cu130 build
from `https://download.pytorch.org/whl/nightly/`.

## Running the pilot

Run everything from the **repository root** (imports are package-qualified `src.*`).

```powershell
# Full sweep: all seeds x all thresholds from configs/pilot.yaml
python -m src.scripts.run_pilot

# Regenerate the 5 figures from saved results
python -m src.scripts.make_plots
```

Results are written to `results/pilot/`:

- `results/pilot/raw/` — one JSON per `(seed, threshold)` condition (`seed{S}_theta{T}.json`)
- `results/pilot/metrics/summary.csv` — one row per condition
- `results/pilot/plots/` — the 5 figures

### `run_pilot` flags

| Flag | Default | Meaning |
|---|---|---|
| `--config PATH` | `configs/pilot.yaml` | Config file to load. |
| `--seeds S [S ...]` | config `seeds` (`0 1 2`) | Random seeds to run. |
| `--thresholds T [T ...]` | config `thresholds` | Fixed global firing thresholds to sweep. Spike activity is a **measured outcome** of each threshold, not a target. Higher threshold = sparser firing (theta 1.5 ≈ 35% active, 64 ≈ near-dead). |
| `--probe-samples N` | `512` | Fixed probe subset size per task used for representational-overlap (CKA/cosine). |
| `--quick` | off | 1 epoch/task instead of 10 — fast smoke test, not for real results. |
| `--threads N` | config `num_threads` (`2`) | Cap CPU threads torch uses. `0` = all cores (fastest). See below. |
| `--device {cpu,cuda,auto}` | config `device` (`auto`) | Compute device. `auto` uses the GPU if a CUDA torch build sees one, else CPU. The run prints the resolved device (`[cfg] device = ...`). |

### Examples

```powershell
# Fast end-to-end smoke test (1 seed, 1 threshold, 1 epoch/task)
python -m src.scripts.run_pilot --seeds 0 --thresholds 5.0 --quick --probe-samples 128

# A subset sweep on one seed
python -m src.scripts.run_pilot --seeds 0 --thresholds 1.5 5.0 16.0 32.0

# Full run using all cores (fastest, but will saturate the machine)
python -m src.scripts.run_pilot --threads 0

# Force the GPU (with a CUDA torch build installed)
python -m src.scripts.run_pilot --device cuda
```

### GPU vs CPU

With the CUDA torch build (the default in `requirements.txt`), `device: auto` picks the GPU
automatically — no flag needed. The `--threads` cap only matters on CPU. The model here is
small (`784 → 256 → 256`), so the GPU speedup is modest (roughly 3–10x); its bigger benefit is
moving load off the CPU so the desktop stays responsive during a run.

## Keeping your machine usable during a run

The CPU-only torch build will otherwise grab **every core** for the LIF matmuls and can
freeze the desktop. Two independent controls:

1. **Thread cap** (default). `configs/pilot.yaml` sets `num_threads: 2`, so the plain
   `python -m src.scripts.run_pilot` already caps torch at 2 threads. Override per-run with
   `--threads 1` (most responsive) or `--threads 0` (all cores, fastest). Edit the config
   key to change the default.

2. **Lower OS priority** (Windows/PowerShell), independent of thread count:

   ```powershell
   Start-Process python -ArgumentList '-m','src.scripts.run_pilot' -PriorityClass BelowNormal
   ```

## Pilot design (short version)

- **Data:** Split-MNIST, 5 binary tasks `(0,1) (2,3) (4,5) (6,7) (8,9)`, task-incremental (per-task heads).
- **Model:** feedforward LIF-SNN, `784 → 256 → 256`, snntorch `Leaky` neurons, `beta ≈ 0.9512`, `T = 25` timesteps, rate/direct input encoding, spike-count readout.
- **Sparsity knob:** one global firing threshold shared by both LIF layers, fixed per condition (no calibration). Activity is measured, not targeted.
- **CL method:** naive sequential (no replay/regularization) — this is a correlation-**screening** pilot only.
- **Metrics:** accuracy matrix, final average accuracy, per-task forgetting, mean forgetting, observed activity, representational overlap (linear CKA + cosine).
- **Note:** under this configuration the network fires at most ~38% of neurons (an activity ceiling); very high thresholds produce a degenerate "dead network" that is auto-flagged (`dead_network`) and excluded from mechanism analysis.

## Project documents

- `RESEARCH_IDEA_REFINED.md` — the refined proposal (research questions, hypotheses H1–H4, confound controls, mediation model).
- `EXPERIMENT_PROTOCOL.md` — the two-stage pilot → full-study protocol.
- `RESEARCH_REPORT.md` — comprehensive standalone deep-dive, including pilot findings.
- `RESEARCH_JOURNAL.md` — dated, append-only research log.
- `RELATED_WORK_REFERENCES.md` — annotated reference list and novelty positioning.
