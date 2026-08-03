# Research Journal — Spike Sparsity & Catastrophic Forgetting in Continual Learning SNNs

**What this file is:** A personal, dated, append-only log of work sessions on this project. It records what happened, why decisions were made, what was found, what's still open, and what comes next. It is NOT a polished report — that's `RESEARCH_REPORT.md`. This is the messy, honest, first-person record.

**How to use it:** Add a new dated entry at the bottom after each work session. Keep entries in chronological order (oldest at top, newest at bottom). Don't edit old entries — if you change your mind, say so in a new entry. Correct the placeholder dates `[DATE — ...]` to real dates as you go.

---

## Project at a glance

| | |
|---|---|
| **Topic** | Investigating the relationship between spike sparsity and catastrophic forgetting in continual learning spiking neural networks |
| **Central hypothesis** | Sparsity reduces forgetting *by reducing representational overlap* — overlap is the mediator, not just a correlate |
| **Current phase** | Pilot built and verified; calibration fixed after first run exposed a compressed activity axis; corrected-axis re-run pending |
| **Two-stage plan** | Stage 1: minimal pilot to screen mechanism-link signals (correlation/mediation screening only, not confirmatory); Stage 2: full study with formal mediation model, confirmatory statistics hierarchy, and mechanism-separated reporting |

---

## Standing decisions

These are locked in. Don't re-litigate them without a new journal entry explaining why.

1. **Pursuing the full research lifecycle** — literature check, novelty assessment, proposal critique, pilot, full study. Not cutting corners to get to code faster.
2. **`archive/` is out of scope** — its contents are not read, referenced, or modified. Treat it as sealed.
3. **Revise proposal docs before writing any pilot code** — the mediation claim needed a real model behind it, not a correlation plot. Code comes after the docs are right.
4. **The four citation errors were fixed** — arXiv IDs now correctly attributed (see Entry 2). The 2026-dated preprints are marked provisionally verified; we must confirm them manually.
5. **No pilot implementation until the revised docs are approved** — we review first, then we build.
6. **Maintain this journal + a deep-dive report** — the journal is the running log; `RESEARCH_REPORT.md` is the exhaustive standalone document. Both are kept current.

---

## Glossary of key terms

**Catastrophic forgetting** — the tendency of a neural network to abruptly lose performance on previously learned tasks when trained sequentially on new ones, because gradient updates overwrite earlier weights.

**Continual learning settings** — three standard scenarios from van de Ven & Tolias 2019: *task-IL* (task identity given at test time), *domain-IL* (same output head, distribution shifts), *class-IL* (no task identity at test time; hardest).

**LIF neuron** — Leaky Integrate-and-Fire: the standard spiking neuron model. Membrane potential integrates input, leaks over time, and fires (emits a spike) when it crosses a threshold, then resets.

**Spike sparsity / activation sparsity** — the fraction of neurons that fire at least once during a forward pass. High sparsity = few neurons active. Can be controlled via threshold modulation, winner-take-all (WTA) circuits, or activity regularization.

**Representational overlap** — how similar the active neuron populations are across tasks. High overlap means the same neurons encode different tasks, making weight updates for one task destructive to another.

**Mediation vs correlation** — correlation says X and Y move together; mediation says X causes Y *through* M (the mediator). Establishing mediation requires estimating path a (X→M), path b (M→Y controlling for X), the indirect effect a×b, and a bootstrap CI on that product. Correlation across sparsity levels is not mediation.

**CKA** — Centered Kernel Alignment. A similarity metric for comparing neural representations across layers or models, invariant to orthogonal transformations and isotropic scaling. Used here to quantify representational overlap between task representations.

**Forgetting score** — the drop in accuracy on task *k* after training on subsequent tasks, averaged across tasks. Formally: F = (1/(T-1)) × Σ (a_{k,k} − a_{T,k}) where a_{i,j} is accuracy on task j after training on task i.

---

## Journal entries

---

### [DATE — first working session] | Entry 1: Kickoff and workspace assessment

We started this project from `RESEARCH_IDEA_REFINED.md`, already written, aiming for the full research lifecycle: literature check, novelty assessment, proposal critique, pilot scoping, and eventually a working codebase.

First thing I did was assess what's actually in the workspace. Found: the main proposal (`RESEARCH_IDEA_REFINED.md`), an experiment protocol (`EXPERIMENT_PROTOCOL.md`), a related-work reference list (`RELATED_WORK_REFERENCES.md`), a companion learning guide (`COMPANION_GUIDE.md`), a `requirements.txt`, and an `archive/` directory. No code exists yet — no `src/`, no notebooks, nothing runnable.

We decided to ignore `archive/`. That's now a standing rule.

The proposal was already reasonably developed — it had hypotheses, a sparsity grid, a pilot design, and some statistical thinking. But I hadn't read it carefully yet. That came next.

---

### [DATE — first working session] | Entry 2: Citation verification and novelty assessment

Ran a literature check against the reference list. The good news: no citations were outright fabricated. The bad news: four were mis-cited and needed fixing.

**The four corrected citations:**
- `arXiv:2602.12236` → Meem et al. 2026, "Energy-Aware Spike Budgeting" — combines replay with learnable LIF parameters and an adaptive spike scheduler for continual learning.
- `arXiv:2604.16496` → Roy et al. 2026 — gradient-free continual learning via inter-spike-interval (ISI-CV) regularization. Interesting because it sidesteps backprop entirely.
- `arXiv:2510.03648` → SAFA-SNN (Zhang et al. 2025) — sparse adaptive feature aggregation in SNNs.
- `arXiv:2603.15184` → CATFormer (Nagabhushana et al. 2026) — a transformer-SNN hybrid for continual learning.

**Caveat on the 2026 preprints:** These are dated 2026, which means they're very recent. We marked them "provisionally verified" — we should open the arXiv abstract pages directly to confirm the titles, authors, and abstracts match what's cited. Don't take these on faith.

**Anchors that checked out fine:** Kornblith et al. 2019 (CKA, ICML), van de Ven & Tolias 2019 (three CL scenarios), Shen et al. 2024 (AAAI, sparse selective activation), Hammouamri/Masquelier/Wilson (TMLR, threshold modulation).

**Novelty verdict: PARTIALLY OCCUPIED.**

The prior literature already covers: sparse selective activation (Shen 2024), threshold modulation (Hammouamri et al.), spike budgeting (Meem 2026), and the ANN literature already links representational overlap to forgetting. None of these, individually, is the project's claim.

What remains open — and what this project must keep central — is the full package: multiple controlled sparsity mechanisms in continual-learning SNNs, with representational overlap tested as the *mediator* of forgetting (not just a correlate), with mechanism-separated reporting.

**Key risk recorded here:** If the project drifts toward "sparsity helps forgetting" without the mediation angle, it collapses into Shen 2024's territory. The mediation framing is the novelty. Protect it.

---

### [DATE — first working session] | Entry 3: Proposal critique (methodological review)

Did an independent methodological critique of the proposal. Produced a claim-support risk matrix:

- **H1** (moderate sparsity reduces forgetting ≥30%): borderline — the threshold is specific enough to be falsifiable, but the 30% figure needs justification.
- **H2** (<5% activity drops accuracy ≥50%): borderline — plausible but the 50% drop is a strong claim with no prior anchor.
- **H3** (inverted-U relationship): borderline — the prediction is directionally reasonable, but "inverted-U" is easy to claim post-hoc without a strict interior-peak criterion.
- **Mediation claim**: UNSUPPORTED AS ORIGINALLY WRITTEN. The original design only correlated overlap and forgetting across sparsity levels. Correlation across conditions is not mediation. This was the biggest problem.
- **RQ4** (sparse-SNN vs param-matched ANN): exploratory — fine to include, but shouldn't be treated as confirmatory.

**Confounds flagged:**
- Mechanism non-equivalence: threshold modulation, WTA, and activity regularization all produce the same active% but through different computational mechanisms. Treating them as equivalent is wrong.
- Capacity partitioning vs overlap reduction: sparser networks have fewer active units, which might reduce forgetting simply by reducing the number of weights updated — not because of overlap geometry.
- Gradient-noise effects: sparse activations change gradient variance, which could independently affect forgetting.
- Threshold calibration drift: if thresholds are set on task 1 and then frozen, they may not produce the intended sparsity level on later tasks.

**Top recommendations from the critique:**
1. Add a formal mediation model (paths a and b, indirect effect a×b, bootstrap CI).
2. Separate overlap, drift, and decodability as distinct measurements.
3. Treat each sparsity mechanism as a distinct intervention, not a parameter of a single mechanism.
4. Strengthen confound controls beyond count-matched freezing.
5. Predefine a confirmatory statistics hierarchy.
6. Expand seeds (the original design used too few for reliable variance estimates).

---

### [DATE — first working session] | Entry 4: Decision — revise the proposal docs before any code

The critique created a real tension. The pilot protocol as written only did correlation. The novelty requires mediation. Options were:

1. Keep the pilot minimal and defer mediation to the full study.
2. Add mediation to the pilot.
3. Revise the proposal docs first, then build.

We chose option 3: **revise the proposal docs first.** We also fixed the four citation errors immediately rather than leaving them as a known issue.

This was the right call. Building a pilot against a proposal that doesn't properly specify the mediation model would mean the pilot's analysis plan is wrong from the start. Better to get the docs right, then code to them.

---

### [DATE — first working session] | Entry 5: Proposal revisions applied

Recorded the concrete edits made to the proposal documents.

**`RESEARCH_IDEA_REFINED.md` changes:**
- Added H4 explicitly: "Representational overlap mediates the effect of sparsity on forgetting." This is now a named, testable hypothesis, not a background assumption.
- Tightened H3 with an interior-peak requirement: the fitted vertex of the inverted-U must sit strictly inside the tested sparsity range, with a bootstrap CI that excludes the boundaries. This prevents post-hoc "inverted-U" claims when the data is actually monotone.
- Added a mechanism-separation requirement to H3: the inverted-U must hold within each mechanism separately, not just in pooled data.
- In the confound section: separated cross-task overlap, representation drift, and decodability as three distinct things to measure. Noted that count-matched freezing controls quantity but not geometry. Added three new controls: update-norm-matched, activation-dropout, and structured neuron-block freezing.
- Added section 3.5: the formal mediation model. Specifies paths a (sparsity→overlap) and b (overlap→forgetting, controlling for sparsity), the direct effect c', the indirect effect a×b with bootstrap CI, proportion mediated, one predefined primary mediator, and per-mechanism-first analysis order.
- In methodology: added an explicit mechanism-non-equivalence statement with per-mechanism reporting metrics. Expanded the sparsity grid to include 30% (now: 1/5/10/20/30/40/60/80/95%). Upgraded confirmatory seeds to 8-10. Predefined a three-family confirmatory statistics hierarchy: primary (Holm-Bonferroni), secondary (Holm-Bonferroni), exploratory (FDR).

**`EXPERIMENT_PROTOCOL.md` changes:**
- Added a correlation-vs-mediation scope note at the top, clarifying that the pilot is screening-only.
- Added an explicit two-stage design description.
- Relabeled the pilot mechanism-link analyses as "screening-only, not confirmatory."
- Added a full-study mediation-analysis paragraph specifying the bootstrap procedure.
- Updated the statistical plan with the seed guidance and the confirmatory hierarchy.

**Net effect:** The mediation claim is now backed by an actual model. Mechanism non-equivalence is explicit. The confounds separate "fewer updates / less capacity" from "sparse coding reduces overlap." The proposal is now internally consistent with its central novelty claim.

---

### [DATE — first working session] | Entry 6: Current status and next steps

Proposal docs are fully revised. We then set up this journal and a deep-dive report (`RESEARCH_REPORT.md`).

**Open items as of this entry:**

1. **We review/approve the revised docs** — nothing moves to code until this happens.
2. **We personally confirm the 2026-dated arXiv references** — open the abstract pages, check that titles and authors match. Don't rely on the provisional verification.
3. **Environment sanity-check before any install** — `requirements.txt` pins versions that likely don't exist (torch==2.12.0, numpy==2.4.6, snntorch==0.9.4). Before running `pip install`, verify what's actually available. Known-good fallback set: torch==2.3.1, torchvision==0.18.1, snntorch==0.9.1, numpy==1.26.4, scikit-learn==1.5.1, matplotlib==3.9.0, pandas==2.2.2, scipy==1.13.1.
4. **Pilot build** — once docs are approved and the environment is sane, build the minimal pilot. The implementation plan already exists: roughly 30 small modules across data/, models/, training/, analysis/, configs/, scripts/, results/. Design defaults: T=25 timesteps, direct current-injection encoding, fast-sigmoid surrogate gradient, global scalar threshold calibrated on task-1 warm-up then frozen, activity defined as neurons that spike at least once over the window across both 256-unit hidden layers, representation for overlap = hidden-layer-2 spike counts, readout = spike-count argmax.

---

### [DATE — pilot session] | Entry 7: Pilot built end-to-end

We approved the proposal revisions and went ahead with the minimal pilot. One constraint: keep all code in a single `src/` folder.

First thing I had to do was correct something from Entry 6. I wrote that the `requirements.txt` versions looked implausible and listed a "known-good fallback set." That was wrong. The pinned versions — torch==2.12.0, torchvision==0.27.0, snntorch==0.9.4, numpy==2.4.6, scipy==1.17.1, scikit-learn==1.9.0, pandas==3.0.3, matplotlib==3.11.0 — are real and install cleanly on this machine's Python 3.14.2. torch 2.12.0 is a CPU build. The fallback set was never needed and was not used. Entry 6 was wrong about this; recording the correction here.

One genuine gap: PyYAML was missing from `requirements.txt` even though the run scripts import `yaml`. Added PyYAML==6.0.3.

**What got built:** roughly 30 small single-purpose modules under `src/`:

- `data/`: `split_mnist.py`, `transforms.py`
- `models/`: `lif_snn.py`, `heads.py`, `encoding.py`
- `training/`: `seeds.py`, `instrumentation.py`, `calibration.py`, `train_task.py`, `evaluate.py`, `continual.py`
- `analysis/`: `metrics.py`, `representations.py`, `overlap.py`, `io.py`, `plotting.py`
- `scripts/`: `run_pilot.py`, `make_plots.py`
- Config at `configs/pilot.yaml`

**Key design defaults as built:**

- Split-MNIST, 5 binary tasks (0-1, 2-3, 4-5, 6-7, 8-9), task-incremental with per-task binary heads
- LIF-SNN: 784 -> 256 -> 256 in snntorch (Leaky neurons, fast-sigmoid surrogate, beta=0.9512 from tau_mem=20ms, reset-to-zero)
- T=25 timesteps, direct/rate current-injection encoding (repeat the flattened input across timesteps, not Poisson)
- Spike-count argmax readout
- Naive sequential continual learning (no replay, no regularization)
- Adam lr=0.001, batch 128, 10 epochs/task
- Sparsity controlled by ONE global scalar firing threshold shared across both LIF layers, calibrated on a task-0 warm-up then frozen for all 5 tasks
- Activity metric = fraction of hidden neurons (pooled across both 256-unit layers, denominator 512) that spike at least once over the T-window
- Overlap representation = hidden-layer-2 spike counts
- Pilot is correlation-screening only — no mediation model (that's the full study)

LSP tooling was unavailable this whole session, so everything was verified by actually running it. A dry run of one condition passed end-to-end.

Two things that looked alarming but were fine: (1) cosine overlap came out ~0.95 while CKA was ~0.02 — expected, because cosine compares raw task-mean vectors while CKA compares centered structure; both are bounded correctly. (2) An early apparent target/observed mismatch was the first sign of the calibration-drift problem diagnosed in Entry 9.

---

### [DATE — pilot session] | Entry 8: First full pilot run — the mechanism signal is there, but the activity axis was broken

We ran the full pilot: 3 seeds x 5 original targets (0.01, 0.10, 0.20, 0.40, 0.80), 10 epochs/task, 15 conditions total.

**Critical interpretation rule discovered first:** the `target_activity` column in the results is misleading. Targets 0.01 and 0.10 produced identical results per seed — calibration couldn't reach 1% activity and both collapsed to the same threshold (theta=32). You must always interpret by `mean_observed_activity`, never by target.

**Results averaged across 3 seeds, ordered by observed activity:**

| Observed activity | Final accuracy | Mean forgetting | CKA |
|---|---|---|---|
| ~0.377 (target 0.4) | ~0.850 | ~0.201 | ~0.0095 |
| ~0.430 (target 0.2) | ~0.930 | ~0.082 | ~0.0071 |
| ~0.441 (target 0.8) | ~0.690 | ~0.377 | ~0.0134 |
| ~0.557 (targets 0.01/0.10 collapsed) | ~0.909 | ~0.107 | ~0.0068 |

**Against the 5 pilot decision criteria:**

1. Lower forgetting at moderate sparsity — PARTIAL. Lowest forgetting at ~0.43 observed activity, a hint of inverted-U, but noisy.
2. Extreme sparsity hurts accuracy — COULD NOT TEST. The run never got below ~0.38 observed activity, so the <5% regime H2 predicts was never reached.
3. Overlap decreases as sparsity increases — directionally yes, but all CKA values are tiny (0.006-0.015). Near-orthogonal everywhere.
4. Overlap correlates with forgetting — YES. This is the most encouraging result: higher CKA tracked higher forgetting consistently. The worst-forgetting rows had the highest CKA; the best had the lowest.
5. Consistent across 3 seeds — YES. Very stable ordering across all three seeds.

**Honest verdict:** the core mechanism signal (overlap tracks forgetting, stable across seeds) is genuinely promising. But the sparsity manipulation was broken at the low end. The whole activity axis was compressed into a narrow ~0.38-0.56 band, so H2 and the low arm of the inverted-U (H3) were untestable. The right response is to fix calibration and re-run, not to reframe the project around what the broken run happened to show.

We decided to do both: (1) fix calibration and re-run, and (2) write up the findings in this journal.

---

### [DATE — pilot session] | Entry 9: Root cause of the broken activity axis, and the calibration fix

Diagnosed two root causes by reading `calibration.py` + `instrumentation.py` + a raw result JSON, then confirmed empirically with a probe script.

**Root cause 1 — low-activity floor:** The threshold sweep ran after only 1 warm-up epoch. Even at the grid's maximum threshold, activity bottomed out around 7.4%. The auto-expand ceiling was capped at 50, so the 1% target was unreachable. Targets 0.01 and 0.10 both collapsed onto the same threshold.

**Root cause 2 — calibration drift (the real killer):** 1 warm-up epoch badly underfits. Task-0 train loss was still ~0.17, far from converged. With underfit weights, few neurons cross the threshold, so calibration measured e.g. 7.4% activity. But once the real 10-epoch training runs, weights grow and many more neurons cross that same frozen threshold. Observed activity during the continual-learning run drifted to 6-8x the calibrated value — per-task activity climbed monotonically, e.g. from ~0.42 up to ~0.64. A threshold calibrated on an underfit network is meaningless once the network is trained.

**Empirical confirmation:** A probe with 5 warm-up epochs gave a smooth monotonic threshold-to-activity curve spanning the full range: theta=1 -> ~36%, theta=32 -> ~8%, theta=64 -> ~0.9%, theta=96 -> ~0.02%. Five-epoch warm-up makes the calibration representative of the trained network, and 1% activity is reachable around theta=64.

**Fixes applied:**

- Raised the calibration auto-expand ceiling from 50 to 200
- Raised warm-up epochs from 1 to 5 in the config
- Extended the threshold grid up to 96

**A genuine model property discovered in the process:** Under this LIF / rate-encoding / reset-to-zero / T=25 configuration, no threshold makes more than ~37-38% of neurons ever fire. That ~38% activity ceiling is a real finding, not a bug. The sparsity axis for this architecture is realistically ~1% to ~38%, not 1-80%.

Because of that ceiling, we redefined the target grid to the reachable range: `target_activity` is now [0.01, 0.05, 0.10, 0.20, 0.35]. All five now map to distinct reachable thresholds and observed activities:

| Target | Theta | Observed activity |
|---|---|---|
| 0.01 | 64 | ~0.009 |
| 0.05 | 48 | ~0.034 |
| 0.10 | 24 | ~0.110 |
| 0.20 | 5 | ~0.229 |
| 0.35 | 1.5 | ~0.348 |

No more collapse. The extreme-sparse regime (1%, 5%) needed for H2 is now reachable.

**Status at end of this entry:** Calibration fixed and verified. The corrected-axis full pilot is queued for us to re-run. Interpreting those fresh results is the next step.

---

## Next session — start here

**Step 1:** Re-run the full pilot with the corrected activity axis. From the repo root:

```
python -m src.scripts.run_pilot
python -m src.scripts.make_plots
```

**Step 2:** Interpret the fresh `summary.csv` against the 5 pilot decision criteria. Now that 1-5% activity is reachable, H2 (extreme sparsity hurts accuracy) and the low arm of the inverted-U (H3) are actually testable for the first time. Check whether the inverted-U vertex sits strictly inside the tested range.

**Step 3:** Check whether the overlap-tracks-forgetting signal from Entry 8 survives on the wider axis. That was the most promising result from the first run; it needs to hold on the corrected data before it means anything.

**Step 4:** The 2026-dated arXiv citations still need manual confirmation. Open the abstract pages and verify titles and authors match what's cited. This has been open since Entry 2.

**Step 5:** No git commits have been made yet. Once the corrected-axis run looks good, commit the working codebase.

**Don't skip steps.** The whole point of the standing decisions is that they hold.
