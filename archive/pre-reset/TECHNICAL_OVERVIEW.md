# Technical overview

This document explains the research project in a direct technical way. It is for someone who wants to understand what the project tests, how the pilot experiment works, what the current results show, and what should happen next.

## 1. Problem

Catastrophic forgetting happens when a model learns a new task and loses performance on tasks it learned earlier.

Example:

1. A model learns Task A.
2. The same model is trained on Task B.
3. Performance on Task A drops because the new training changed parameters that were useful for Task A.

This matters in continual learning because the model receives tasks sequentially. It cannot assume that all data is available at once.

## 2. Research idea

This project tests whether spike sparsity in spiking neural networks (SNNs) can reduce catastrophic forgetting.

The main idea is simple:

> If fewer neurons are active for each task, then different tasks may use less-overlapping internal representations. Less overlap may reduce interference when the model learns new tasks.

The current pilot does not claim that SNNs solve catastrophic forgetting. It tests a narrower question:

> In a controlled Split-MNIST LIF-SNN setting, does changing the firing threshold affect forgetting and representation overlap?

## 3. Hypothesis

The intended mechanism is:

```text
Higher LIF firing threshold
        -> lower spiking activity
        -> fewer active hidden units
        -> lower task-representation overlap
        -> less interference during new-task learning
        -> lower catastrophic forgetting
```

There is also a downside:

```text
Too little activity
        -> too few active hidden units
        -> lower representational capacity
        -> lower accuracy
```

So the expected pattern is not "more sparsity is always better." The expected pattern is a tradeoff.

| Regime | Expected behavior |
|---|---|
| Too dense | More overlap, more interference, more forgetting |
| Moderate sparsity | Lower overlap, better retention |
| Too sparse | Not enough active capacity, lower accuracy |

## 4. Pilot experiment

The pilot experiment is intentionally small. It tests the core mechanism before adding more datasets, baselines, or continual learning methods.

| Component | Pilot setup |
|---|---|
| Dataset | Split-MNIST |
| Continual learning setting | Task-incremental learning |
| Model | LIF-SNN only |
| Training method | Naive sequential learning |
| Sparsity control | LIF firing threshold |
| Nominal activity targets | 1%, 10%, 20%, 40%, 80% |
| Seeds | 3 |
| Epochs per task | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |

Split-MNIST uses five binary tasks:

1. Task 1: digits 0 vs 1
2. Task 2: digits 2 vs 3
3. Task 3: digits 4 vs 5
4. Task 4: digits 6 vs 7
5. Task 5: digits 8 vs 9

The model learns these tasks one after another. After each task, it is evaluated on all tasks seen so far. This creates an accuracy matrix that shows how much old-task performance changes as new tasks are learned.

## 5. Model

The pilot uses a feedforward LIF-SNN implemented with `snntorch`.

Architecture:

```text
Flattened MNIST image, 784 inputs
        -> 256 LIF hidden units
        -> 256 LIF hidden units
        -> task-specific 2-class output head
```

Key parameters:

| Parameter | Value |
|---|---|
| Neuron type | Leaky integrate-and-fire (LIF) |
| Membrane time constant | `tau_mem = 20 ms` |
| Reset potential | `V_reset = 0.0` |
| Resting potential | `V_rest = 0.0` |
| Time steps | 25 |
| Hidden representation used for overlap | `h2_mean`, the average spike activity of the second hidden layer |

The LIF threshold is the experimental control. A higher threshold means neurons need stronger input to spike.

## 6. Codebase flow

The pilot code is organized around a single execution path.

| File | Role |
|---|---|
| `requirements.txt` | Lists Python dependencies |
| `.venv/` | Local Python 3.12 environment |
| `src/data.py` | Downloads MNIST, builds Split-MNIST tasks, creates train/test loaders and calibration batch |
| `src/model.py` | Defines `LIFNet`, the feedforward LIF-SNN |
| `src/sparsity.py` | Calibrates firing threshold and records achieved activity |
| `src/train.py` | Runs naive sequential training and collects hidden representations |
| `src/metrics.py` | Computes accuracy, forgetting, BWT, spike stats, and energy proxy |
| `src/overlap.py` | Computes cosine overlap, PCA subspace overlap, and linear CKA |
| `src/run_pilot.py` | Orchestrates the full pilot and writes results |
| `src/plots.py` | Reads results and generates figures |

Execution flow:

```text
src/run_pilot.py
    -> src/data.py
    -> src/model.py
    -> src/sparsity.py
    -> src/train.py
    -> src/metrics.py
    -> src/overlap.py
    -> results/metrics.csv
    -> results/runs/*.json

src/plots.py
    -> results/metrics.csv
    -> results/runs/*.json
    -> results/*.png
```

## 7. Metrics

The pilot measures both performance and mechanism.

### Performance metrics

| Metric | Meaning |
|---|---|
| Accuracy matrix `A[i,j]` | Accuracy on task `j` after training task `i` |
| Final average accuracy | Average final accuracy across all tasks |
| Mean forgetting | Average drop from best old-task accuracy to final old-task accuracy |
| Backward transfer (BWT) | How later tasks affect earlier-task performance |

### Sparsity and energy metrics

| Metric | Meaning |
|---|---|
| Trained activity | Fraction of hidden units active after training |
| Spike rate | Average spikes per neuron per time step |
| Total spike count | Total hidden spikes over evaluation |
| Energy proxy | `spike_count x synaptic_operations` |

The energy proxy is not a hardware energy measurement. It is only a computational estimate.

### Representation-overlap metrics

| Metric | Meaning |
|---|---|
| Cosine overlap | Similarity between task-mean hidden representations |
| PCA subspace overlap | Overlap between principal representation subspaces |
| Linear CKA | Scale-invariant representation similarity |

## 8. How to run the experiment

Run these commands from the repo root:

```powershell
cd C:\Users\swapn\code\snn
$env:PYTHONUTF8="1"; .venv\Scripts\python.exe src\run_pilot.py
$env:PYTHONUTF8="1"; .venv\Scripts\python.exe src\plots.py
```

The experiment writes:

| Output | Meaning |
|---|---|
| `results/metrics.csv` | One summary row per run |
| `results/runs/*.json` | Per-run details, including accuracy matrices |
| `results/fig_forgetting_vs_activity.png` | Forgetting vs achieved activity |
| `results/fig_accuracy_vs_activity.png` | Accuracy vs achieved activity |
| `results/fig_retention_curves.png` | Per-task retention curves |
| `results/fig_overlap_vs_activity.png` | Representation overlap vs achieved activity |
| `results/fig_overlap_vs_forgetting.png` | Overlap vs forgetting |

## 9. Current pilot results

The pilot ran 15 configurations: 5 threshold targets times 3 seeds.

| Target activity | Trained activity | Final accuracy | Mean forgetting | Cosine overlap | PCA overlap |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.654 ± 0.029 | 0.942 ± 0.007 | 0.068 ± 0.009 | 0.976 | 0.701 |
| 0.10 | 0.443 ± 0.037 | 0.977 ± 0.001 | 0.026 ± 0.002 | 0.945 | 0.676 |
| 0.20 | 0.367 ± 0.012 | 0.952 ± 0.023 | 0.056 ± 0.029 | 0.943 | 0.647 |
| 0.40 | 0.357 ± 0.030 | 0.876 ± 0.016 | 0.149 ± 0.020 | 0.980 | 0.594 |
| 0.80 | 0.462 ± 0.009 | 0.735 ± 0.040 | 0.319 ± 0.050 | 0.993 | 0.526 |

The best pilot condition was target `0.10`:

- Final accuracy: `0.977 ± 0.001`
- Mean forgetting: `0.026 ± 0.002`

The worst pilot condition was target `0.80`:

- Final accuracy: `0.735 ± 0.040`
- Mean forgetting: `0.319 ± 0.050`

## 10. Correlation summary

These correlations use all 15 runs and are keyed on achieved trained activity.

| Relationship | Correlation |
|---|---:|
| Trained activity vs mean forgetting | `r = -0.068` |
| Trained activity vs final accuracy | `r = +0.066` |
| Trained activity vs cosine overlap | `r = +0.326` |
| Trained activity vs PCA overlap | `r = +0.439` |
| Cosine overlap vs mean forgetting | `r = +0.756` |
| PCA overlap vs mean forgetting | `r = -0.873` |
| PCA overlap vs trained activity | `r = +0.439` |

## 11. Interpretation

The pilot supports continuing the project, but the claim must be narrowed.

What worked:

- The target `0.10` condition had the best accuracy and lowest forgetting.
- The target `0.80` condition had the worst accuracy and highest forgetting.
- PCA subspace overlap tracked forgetting strongly, with `r = -0.873`.

What did not work cleanly:

- Nominal target activity did not map cleanly to post-training achieved activity.
- The manipulated variable is better described as threshold strength, not precise activity percentage.
- Cosine overlap disagreed with PCA overlap. Cosine overlap vs forgetting had `r = +0.756`, while PCA overlap vs forgetting had `r = -0.873`.

The safest current interpretation is:

> Increasing the LIF firing threshold, which suppresses dense spiking behavior, reduced catastrophic forgetting on Split-MNIST under naive sequential training. This reduction co-varied with PCA-subspace overlap between task representations.

## 12. What not to claim yet

Do not claim:

1. SNNs solve catastrophic forgetting.
2. Sparsity always improves continual learning.
3. LIF results generalize to all SNN models.
4. Spike-count energy proxies prove real hardware energy efficiency.
5. The experiment precisely controlled achieved activity.

## 13. Next steps

The next technical step is to improve activity control before expanding the experiment.

Recommended order:

1. Fix activity control so nominal activity better matches post-training achieved activity.
2. Treat PCA subspace overlap as the stronger mechanism signal for now, because cosine overlap gave the opposite trend.
3. Repeat the pilot with better activity control.
4. Add WTA and activity regularization only after the threshold-control result is understood.
5. Add baselines after the mechanism is clearer: MLP, ConvNet, Conv-SNN, Replay, EWC, SI, LwF.
6. Update the paper only after the stronger experiment confirms or revises the current claim.

## 14. Short summary

The pilot suggests that threshold strength matters for forgetting in a LIF-SNN on Split-MNIST. The best current condition was target `0.10`, which produced high final accuracy and low forgetting. PCA subspace overlap supports the proposed mechanism, but cosine overlap does not. The main limitation is that threshold calibration did not precisely control achieved activity after training, so the next experiment should fix activity control before expanding the study.
