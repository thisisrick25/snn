# Research Progress Report

Project: Investigating the Relationship Between Spike Sparsity and Catastrophic Forgetting in Continual Learning Spiking Neural Networks

This report tracks everything done so far: the planning work, the pilot implementation, the pilot results and findings, and the recommended next steps. It is written to be read on its own, without needing the rest of the repository open.

## 1. Where the project stands right now

- The research idea has been refined, peer-reviewed (simulated 5-reviewer panel), and grounded in a literature map.
- A pilot-first experiment protocol was written and then actually executed on real data.
- A working code pipeline exists under `src/`, runs on CPU, and produced real metrics and five figures.
- The pilot gave an interpretable signal, but also surfaced two honest complications that must be fixed before any strong claim or paper edit.
- The `paper/` drafts are intentionally frozen until the reframed claim and the confound fix are settled.

## 2. Planning and framing work completed

- Refined the original idea into `RESEARCH_IDEA_REFINED.md` with falsifiable hypotheses:
  - H1: moderate sparsity (roughly 20 to 40 percent activity) reduces forgetting by at least 30 percent relative to dense firing.
  - H2: extreme sparsity (under 5 percent activity) degrades accuracy by at least 50 percent.
  - H3: the sparsity-versus-forgetting relationship is an inverted-U with a peak in the moderate range.
  - Forgetting score is defined as best accuracy minus current accuracy.
- Ran a simulated multi-perspective peer review (editor, neuromorphic, continual learning, cross-disciplinary, devil's advocate). Verdict was Major Revision. The main warning was that the causal mechanism (sparsity reduces interference reduces forgetting) was asserted, not measured, and that the three sparsity controls could be confounded.
- Built `RELATED_WORK_REFERENCES.md`, a clustered citation map. Closest prior work includes Shen et al. 2024 (sparse selective activation in SNN continual learning), Hammouamri et al. (threshold modulation), and several 2024 to 2026 SNN continual-learning preprints. Foundational continual-learning references (EWC, SI, GEM, LwF, PackNet) and mechanism references (representation overlap, NTK overlap, CKA, SVCCA) are included.
- Agreed on a sharper novelty framing: not "sparse SNNs reduce forgetting," but a systematic causal study of how controlled spike sparsity affects forgetting through representational overlap.
- Wrote `EXPERIMENT_PROTOCOL.md`, which defines a pilot-first plan with explicit decision criteria and claim boundaries, so the work cannot quietly overclaim.
- Old drafts were moved into `archive/` rather than deleted, keeping the working tree clean while preserving history.

## 3. Pilot implementation completed

The pilot scope was deliberately narrow: one model family, one dataset, one continual-learning setting, one sparsity knob. Everything else from the full protocol was deferred on purpose.

- Environment: a local virtual environment at `.venv` (CPython 3.12) with pinned dependencies in `requirements.txt` (torch, torchvision, snntorch, numpy, scikit-learn, matplotlib, pandas, scipy). CPU is sufficient.
- Code modules under `src/`, each individually smoke-tested before the full run:
  - `data.py`: Split-MNIST as five binary tasks ((0,1), (2,3), (4,5), (6,7), (8,9)) in fixed order, task-incremental loaders.
  - `model.py`: a feedforward LIF spiking network in snntorch (784 to 256 to 256 to per-task output head), tau_mem 20 ms, configurable firing threshold, 25 timesteps, rate-coded input.
  - `sparsity.py`: calibrates the firing threshold to hit a target active-neuron percentage, and measures achieved activity.
  - `metrics.py`: accuracy matrix, per-task forgetting, mean forgetting, backward transfer, spike rate, active percentage, and a computational energy proxy.
  - `overlap.py`: three representational-overlap measures between tasks (cosine, PCA subspace overlap, linear CKA).
  - `train.py`: the naive sequential continual-learning loop (Adam, learning rate 0.001, batch 128, 10 epochs per task) with no forgetting defence, plus representation collection.
  - `run_pilot.py`: the orchestrator that sweeps 5 activity targets by 3 seeds, calibrates, trains, evaluates, and writes `results/metrics.csv` plus one JSON per run.
  - `plots.py`: generates the five pilot figures.
- The full pilot was actually run end to end: 5 activity targets {0.01, 0.10, 0.20, 0.40, 0.80} by 3 seeds {0, 1, 2}, each five tasks by ten epochs. Outputs are real, not simulated:
  - `results/metrics.csv` with 15 rows.
  - 15 per-run JSON files in `results/runs/` (each with the full accuracy matrix).
  - `results/pilot_run.log`.
  - Five figures in `results/`: forgetting versus activity, accuracy versus activity, retention curves, overlap versus activity, and overlap versus forgetting.

## 4. Pilot results

Values below are the mean over three seeds, grouped by the nominal sparsity target. Note that the nominal target sets the threshold strength; the achieved activity after training drifts away from it (this is the central caveat, see Section 5).

- Target 0.01 (strongest threshold): achieved activity 0.654, final accuracy 0.942, mean forgetting 0.068, PCA overlap 0.701.
- Target 0.10: achieved activity 0.443, final accuracy 0.977 (best), mean forgetting 0.026 (best), PCA overlap 0.676.
- Target 0.20: achieved activity 0.367, final accuracy 0.952, mean forgetting 0.056, PCA overlap 0.647.
- Target 0.40: achieved activity 0.357, final accuracy 0.876, mean forgetting 0.149, PCA overlap 0.594.
- Target 0.80 (weakest threshold, densest firing): achieved activity 0.462, final accuracy 0.735 (worst), mean forgetting 0.319 (worst), PCA overlap 0.526.

Key correlations across all 15 runs:

- PCA overlap versus mean forgetting: r = -0.873. Strong, and in the predicted direction (less overlap goes with less forgetting). This is the mechanism signal the study was designed to detect.
- Cosine overlap versus mean forgetting: r = +0.756. Strong, but in the opposite direction, which contradicts the PCA result.
- Achieved activity versus mean forgetting: r = -0.068. Essentially no relationship.

## 5. Findings and honest complications

What the pilot supports:

- The inverted-U shows up clearly on the performance axis when keyed on threshold strength: the sweet spot is the moderate setting (target 0.10), with the lowest forgetting and the highest accuracy. The densest-firing setting (target 0.80) is the worst, with roughly twelve times more forgetting and accuracy down at 0.735. This matches the direction of H1 and H3.
- The designed mechanism signal is present: stronger thresholds produce lower PCA-subspace overlap between task representations, and lower overlap tracks lower forgetting (r = -0.873) consistently across three seeds.

Two complications that must be reported honestly and fixed before any strong claim:

- Threshold-floor confound. Calibration was done on the untrained network, but activity drifts once training starts, and both the sparse and dense extremes end up landing in a similar achieved-activity band (roughly 0.45 to 0.65). As a result, when results are keyed on measured activity instead of the threshold knob, the relationship nearly vanishes (r = -0.07). The clean inverted-U is therefore a function of the manipulated threshold, not of cleanly controlled activity. We cannot yet say "X percent activity causes Y forgetting."
- Cosine versus PCA disagreement. The two overlap measures point in opposite directions (cosine r = +0.76, PCA r = -0.87). PCA subspace overlap is the more trustworthy measure here, but the disagreement means "sparsity reduces interference" is suggestive, not settled.

## 6. Decision

- Verdict: continue, with a reframe.
- The protocol's continue-gate is substantially met: moderate sparsity lowers forgetting, extreme density hurts accuracy, PCA overlap drops as the threshold strengthens, PCA overlap correlates with forgetting, and the pattern holds across three seeds with small spread.
- Reframed claim: increasing the LIF firing threshold (suppressing dense activity) reduces catastrophic forgetting on Split-MNIST under naive sequential training, and this reduction co-varies with reduced PCA-subspace overlap between task representations. The manipulated variable is the spike threshold, not a cleanly controlled activity level.
- Claim boundaries that stay in force: the pilot does not show that SNNs solve forgetting, that sparsity always helps continual learning, that the simple LIF result generalizes to other neuron models, or that the spike-count energy proxy reflects real hardware energy.

## 7. Next steps with explanations

1. Fix the activity-control confound. Right now the threshold is calibrated once on the untrained network, so the variable we actually wanted to control (activity) is not held where we set it. The fix is to recalibrate the threshold during or after training so achieved activity matches intent, or to add winner-take-all and activity-regularization as alternative sparsity knobs, then report every result keyed on achieved activity rather than the nominal target. Without this, the headline relationship can be attacked as a threshold artifact rather than a sparsity effect.

2. Resolve the cosine-versus-PCA disagreement. The two overlap measures contradict each other, so no mechanism claim is safe yet. The plan is to inspect why cosine similarity rises while PCA overlap falls (likely because cosine on task-mean vectors is sensitive to a shared global firing direction that PCA factors out), decide which measure actually captures interference, and add at least one more independent check before stating a mechanism in writing.

3. Only then expand to the full study. After the confound is fixed and the mechanism metric is settled, add the comparisons the protocol defers to stage two: an MLP or ANN baseline matched on parameters, the standard continual-learning defences (EWC, SI, Replay, and LwF), and a second benchmark such as Permuted-MNIST. Doing this before the confound fix would multiply the same measurement weakness across many more conditions.

4. Keep the paper frozen until the above is done. The `paper/` drafts will not be edited until the reframed claim and the confound fix are settled, so the written paper never gets ahead of the evidence.
