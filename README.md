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

## Running the full study

The pilot is a single-mechanism screen. The **full study** runs the complete matrix
across two settings, three sparsity mechanisms, and three confound controls, then fits the
confirmatory mediation analysis. It is configured by `configs/full_study.yaml`
(`results_dir: ./results/full_study`, `seeds: 0..8` = 9 seeds).

### The matrix (12 runs)

| Block | Runs | What |
|---|---|---|
| Core | 6 | 2 settings `{mnist/mlp, cifar10/conv_snn}` × 3 mechanisms `{kwta_window, activity_reg, threshold}` |
| Controls | 6 | 2 settings × 3 confound controls `{update_norm, activation_dropout, block_freeze}`, each on a dense (`--sparsity-mode threshold`) reference |

Each run sweeps its mechanism's whole grid across all 9 seeds, so the expected JSON count
per cell is:

| Cell | Conditions × seeds | Expected JSONs |
|---|---|---|
| `kwta_window` | 4 fractions × 9 | 36 |
| `activity_reg` | 4 targets × 9 | 36 |
| `threshold` | 9 thetas × 9 | 81 |
| each control | 1 × 9 | 9 |

After all runs, `run_confirmatory` fits the analysis over the accumulated `summary.csv`.

### Launch

The wrappers run all 12 cells sequentially and then the confirmatory analysis. Both are
**skip-aware** (see Resume below), so re-running them only fills in what is missing.

```powershell
# Windows
.\run_full_study.ps1               # all cores
.\run_full_study.ps1 -Threads 4    # cap CPU threads to keep the desktop usable
```

```bash
# Linux / Kaggle
bash run_full_study.sh             # all cores
bash run_full_study.sh 4           # optional thread cap
```

Or run any single cell directly (all output goes to `results/full_study/` via the config's
`results_dir`; there is **no `--results-dir` flag**):

```powershell
python -m src.scripts.run_pilot --config configs/full_study.yaml --dataset mnist --arch mlp --sparsity-mode kwta_window
python -m src.scripts.run_pilot --config configs/full_study.yaml --dataset cifar10 --arch conv_snn --sparsity-mode threshold --control block_freeze
python -m src.scripts.run_confirmatory --config configs/full_study.yaml
```

Results land in `results/full_study/`:

- `raw/` — one JSON per `(seed, condition)`, named
  `seed{S}_{dataset}_{arch}_{mechanism}_{control}_act{observed:.3g}.json`
- `metrics/summary.csv` — one row per condition (rebuilt from the **whole** `raw/` dir at the
  end of every run, so it always reflects all accumulated conditions)
- `metrics/confirmatory.json` — the confirmatory mediation output

## Resume mechanism

Long runs (especially `cifar10/conv_snn`) can outlast a machine session or a Kaggle time
limit. The study is designed to resume cleanly with **zero wasted recomputation**, at three
layers:

**1. Per-condition durability.** `run_pilot` writes one JSON to `results/<results_dir>/raw/`
*immediately* after each `(seed, condition)` finishes, and rebuilds `summary.csv` from the
entire `raw/` directory at the end. Accumulation is order-independent: interrupted runs never
lose already-finished conditions, and partial runs merge cleanly.

**2. Per-condition skip (`--resume`, default ON).** Before training each `(seed, condition)`,
`run_pilot` checks whether that condition is already on disk and skips it if so, printing
`[skip] seed=… θ=… already done`. This gives true **mid-mechanism** resume — a cell killed
after seed 3 of 9 resumes at seed 4, not seed 0.

The filename encodes the *observed* activity (unknown before the run finishes), **not** the
swept condition, so the skip check does not rely on the filename. Instead `_already_done()`
globs every raw JSON sharing the run's prefix
(`seed{S}_{dataset}_{arch}_{mechanism}_{control}_*.json`), reads the `threshold` field stored
inside each (which equals the swept condition value), and skips when it matches within `1e-9`.

Pass `--no-resume` to force recomputation (overwrites existing JSONs for those conditions).

**3. Per-cell skip (wrappers).** `run_full_study.ps1` / `run_full_study.sh` count the raw
JSONs for each cell and skip launching a cell whose count already meets the expected total
(table above), printing `[skip] … complete (have/expected)`. So a fully-finished mechanism
costs nothing on a re-run, while a partially-finished one is relaunched and its `--resume`
guard fills only the missing seeds.

### Resuming on Kaggle

Kaggle wipes `/kaggle/working` between sessions and caps each session (~9 h), so resume
requires **persisting `raw/` yourself**:

1. **Persist between sessions.** Save `results/full_study/raw/` as a Kaggle *Dataset* (or via
   a notebook *Commit*), and at the start of each new session restore those JSONs back into
   `results/full_study/raw/` before running. Without this, there is nothing to resume from.
2. **One cell per session.** Run a single mechanism at a time — especially each
   `cifar10/conv_snn` mechanism — to stay under the session limit.
3. **Install without pinning CUDA.** `pip install torch torchvision snntorch pyyaml numpy`
   (do **not** pin `+cu132`; the code is device-agnostic via `device: auto`). Set
   `export PYTHONPATH=$PWD`. Datasets auto-download via torchvision into `./data`.
4. Re-run `bash run_full_study.sh` each session; the skip layers ensure only missing work
   runs, and finish with `python -m src.scripts.run_confirmatory --config configs/full_study.yaml`.

### Important: regenerate the `activity_reg` arm

An earlier bug made `activity_reg` results **seed-invariant** (identical across all 9 seeds,
so effective *n* = 1). Root cause: the λ-calibration step
(`src/training/activity_calibration.py`) re-seeds to `seeds[0]` and left the RNG parked on
that state, so the real `activity_reg` model always trained from the seed-0 RNG. The fix
re-applies `set_seed(seed)` after calibration and before model construction in
`run_pilot.run_condition`. The `threshold` and `kwta_window` arms never used calibration and
are unaffected.

Any `activity_reg` JSONs produced before the fix are invalid. Delete them and regenerate:

```bash
rm results/full_study/raw/*_activity_reg_*.json     # PowerShell: Remove-Item results/full_study/raw/*_activity_reg_*.json
python -m src.scripts.run_pilot --config configs/full_study.yaml --dataset mnist   --arch mlp      --sparsity-mode activity_reg
python -m src.scripts.run_pilot --config configs/full_study.yaml --dataset cifar10 --arch conv_snn --sparsity-mode activity_reg
```

## Project documents

- `RESEARCH_IDEA_REFINED.md` — the refined proposal (research questions, hypotheses H1–H4, confound controls, mediation model).
- `EXPERIMENT_PROTOCOL.md` — the two-stage pilot → full-study protocol.
- `RESEARCH_REPORT.md` — comprehensive standalone deep-dive, including pilot findings.
- `RESEARCH_JOURNAL.md` — dated, append-only research log.
- `RELATED_WORK_REFERENCES.md` — annotated reference list and novelty positioning.
