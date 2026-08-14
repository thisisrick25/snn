# Research Journal — Spike Sparsity & Catastrophic Forgetting in Continual Learning SNNs

**What this file is:** A personal, dated, append-only log of work sessions on this project. It records what happened, why decisions were made, what was found, what's still open, and what comes next. It is NOT a polished report — that's `RESEARCH_REPORT.md`. This is the messy, honest, first-person record.

**How to use it:** Add a new dated entry at the bottom after each work session. Keep entries in chronological order (oldest at top, newest at bottom). Don't edit old entries — if you change your mind, say so in a new entry. Correct the placeholder dates `[DATE — ...]` to real dates as you go.

---

## Project at a glance

| | |
|---|---|
| **Topic** | Investigating the relationship between spike sparsity and catastrophic forgetting in continual learning spiking neural networks |
| **Central hypothesis** | Sparsity reduces forgetting *by reducing representational overlap* — overlap is the mediator, not just a correlate |
| **Current phase** | Phase C complete: formal mediation analysis run on 18 k-WTA conditions (Split-MNIST). Both H3 (inverted-U) and the overlap->forgetting mediation mechanism (H4) are unsupported at pilot scale — two negative screens. Project continues toward a fuller study (more seeds, harder benchmark, decoupled activity range) |
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

### [DATE — pilot re-run + refactor session] | Entry 10: Corrected-axis re-run results, and the Rank1 refactor to fixed-threshold design

Two things happened this session, in order.

#### Part A: The corrected-axis full pilot re-run

The user ran the re-run themselves. 21 conditions total: 3 seeds x 7 target levels (0.01, 0.05, 0.10, 0.20, 0.35, 0.40, 0.80 — the old and new target grids merged). The cardinal rule from Entry 8 still applies: always interpret by `mean_observed_activity`, never by nominal target.

**The dead-network boundary (target 0.01):** All three seeds hit threshold=64. Mean observed activity was exactly 0.0 across all seeds. Final accuracy was 0.5163 — chance. Forgetting was 0.0, CKA was 0.0, loss was frozen at ln(2) ≈ 0.693, zero spikes. This is a degenerate cliff, not a gradient. The network simply never fires. It's excluded from mechanism analysis — there's nothing to analyze.

**Live results, per-seed-averaged, for the remaining six target levels:**

| Nominal target | Observed activity | Final accuracy | Mean forgetting | CKA |
|---|---|---|---|---|
| 0.05 | ~0.452 | ~0.839 | ~0.168 | ~0.0078 |
| 0.10 | ~0.559 | ~0.964 | ~0.038 | ~0.0064 |
| 0.20 | ~0.430 | ~0.935 | ~0.076 | ~0.0072 |
| 0.35 | ~0.405 | ~0.792 | ~0.249 | ~0.0104 |
| 0.40 | ~0.384 | ~0.850 | ~0.180 | ~0.0095 |
| 0.80 | ~0.441 | ~0.690 | ~0.386 | ~0.0134 |

Cosine similarity ran ~0.97-0.99 throughout — near-orthogonal everywhere, consistent with Entry 8.

**Against the three key questions:**

*H2 (extreme sparsity hurts accuracy):* CONFIRMED, but degenerately. The only condition that reached extreme sparsity (target 0.01, theta=64) was the dead-network cliff. Accuracy dropped to chance, which technically confirms H2, but there's no gradient to characterize — it's a binary on/off, not a smooth decline. The shape of the accuracy drop through the sparse regime remains uncharacterized.

*H3 (inverted-U interior peak):* NOT ESTABLISHED. The observed activity axis is still compressed (~0.38-0.56 for the live conditions) and non-monotonic with nominal target — target 0.10 gave the highest observed activity (~0.559) while target 0.80 gave ~0.441. That's calibration drift still at work: the frozen threshold set during warm-up doesn't hold once training runs. You can't fit a clean inverted-U to data where the x-axis is scrambled. This is the same problem as Entry 8, just with more conditions.

*Mechanism link (overlap-CKA tracks forgetting):* SURVIVES. This is the strongest result from the session. The pattern is monotonic and consistent across all three seeds: lowest CKA (~0.006-0.007) paired with lowest forgetting (0.03-0.11); highest CKA (~0.013-0.015) paired with worst forgetting (0.32-0.44). The mediation link the novelty rests on held on the wider axis. That's meaningful.

**Verdict:** Continue. The mechanism signal is real and reproducible. But the sparsity axis is still not under proper control — the frozen-threshold-after-warmup design lets activity drift during continual learning, so the x-axis of every plot is unreliable. That's the thing to fix.

#### Part B: The Rank1 refactor — dropping calibration entirely

The user noted the codebase was getting more complex. A read-only Oracle complexity assessment was run before touching anything.

**Oracle verdict (high confidence):** The frozen-threshold calibration machinery was accidental complexity. It was ceremony fighting a design that doesn't work. Calibrated activity doesn't transfer once training continues — it drifts, as Entries 8 and 9 both showed. And the analysis already keys on observed activity anyway, so the calibration target was never the thing being analyzed. The machinery existed to hit a number that was immediately abandoned.

**Decision (Rank1, Oracle's minimal-regret pick):** Drop the calibration layer entirely. Sweep a fixed set of firing thresholds directly. Spike activity becomes a measured outcome, not a calibration target. Simpler code and more scientifically honest — the threshold is the independent variable, activity is what you measure.

**Changes made and verified:**

- Deleted `src/training/calibration.py`
- Rewrote `src/scripts/run_pilot.py` to sweep fixed thresholds with no warmup, no calibration, no rebuild. Auto-flags degenerate "dead network" conditions: mean activity <= 1e-3 OR final accuracy <= 0.55
- Replaced the config keys `target_activity`, `warmup_epochs`, and `threshold_grid` with a single `thresholds` list: `[1.5, 3.0, 5.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0]`
- New `ConditionRecord` schema keyed on `threshold` + `dead_network` flag instead of `target_activity`
- Fixed plotting for the new schema
- Cleared stale incompatible result files

Verified end-to-end with a `--quick` single-condition run: saved correctly, dead flag correct, all 5 plots regenerated.

**Empirical facts now baked into the config:**

The ~38% activity ceiling from Entry 9 is confirmed. Under this LIF / rate-encoding / reset-to-zero / T=25 setup, no threshold makes more than ~37-38% of neurons ever fire. The reachable sparsity axis is ~1-38%, not 1-80%. The trained-network threshold-to-activity map:

| Threshold | Approx. observed activity |
|---|---|
| 1.5 | ~35% |
| 3.0 | ~30% |
| 5.0 | ~23% |
| 8.0 | ~17% |
| 16.0 | ~14% |
| 24.0 | ~11% |
| 32.0 | ~8% |
| 48.0 | ~3.4% |
| 64.0 | ~0.9% (near-dead) |

These are measured from the trained network, not from a warm-up proxy. That's the whole point.

**Status at end of this entry:** Codebase is cleaner and the design is more honest. The full pilot on the fixed-threshold design is the immediate next step.

---

### [DATE — kwta session] | Entry 11: Fixed-threshold sweep failed; switched to k-WTA

Two things happened this session, in order.

#### Part A: The fixed-threshold pilot — a decisive negative result

The user ran the full fixed-threshold pilot: 27 conditions, 3 seeds x 9 thresholds [1.5, 3, 5, 8, 16, 24, 32, 48, 64], 10 epochs/task.

The threshold utterly failed to control spike activity. Interpreting by measured `mean_observed_activity` (the cardinal rule from Entry 8 — never the nominal threshold), the observed activity clustered in a narrow ~0.33-0.58 band, non-monotonically with the threshold. A few examples: theta=1.5 gave ~0.35, theta=16 gave ~0.57, theta=48 gave ~0.38. Then at theta=64 it cliffed to 0 across all three seeds: dead network, zero spikes, training loss frozen at ln(2) ≈ 0.693, chance accuracy ~0.516.

So the behaviour is bimodal: either "~35-58% active" or "dead", with no stable moderate or low (5-30%) activity regime anywhere in between.

This is a genuine finding, not a bug. Calibration had already been removed in the Entry 10 / Rank1 refactor, so there's no calibration artifact to blame. What this proves is something more fundamental: with a single frozen global firing threshold, spike activity is not a controllable variable in a trained SNN. As the network trains, weights grow until activity re-saturates into that band regardless of the threshold you set. The threshold controls the dead/alive boundary, not the activity level within the alive regime.

**Consequences for the hypotheses:**

- H3 (inverted-U with interior peak): not assessable. The activity axis is a narrow blob plus a dead point. There's nothing to fit a curve to.
- H2 (extreme sparsity hurts accuracy): only showed up as the degenerate dead cliff. That's technically a confirmation, but it's a binary on/off, not the smooth decline H2 predicts.
- Mechanism signal (CKA tracking forgetting): went weak and noisy. CKA range was a tiny 0.0060-0.0095, not cleanly tracking forgetting. This is worth being honest about: the more encouraging CKA signal in earlier runs was riding on a narrow, confounded activity range. The apparent signal was real within those runs, but it wasn't robust to the design change. That partially undercuts the earlier optimism.

The right response is to fix the mechanism, not to reframe the project around what the broken design happened to show.

#### Part B: The fix — switching to k-winner-take-all (k-WTA)

**Decision:** adopt hard k-WTA so spike activity is set directly as a dial (active fraction ≈ k/N) instead of hoping a threshold produces it. This isn't a new direction — top-k/WTA was always a planned mechanism in the full study (see `RESEARCH_IDEA_REFINED.md` section 4.2, mechanism non-equivalence). We're promoting it early rather than inventing something new.

**Design (from an Oracle consult):** the key design question is what "top-k" means over a T-step window. Per-timestep top-k does not control the over-window "active if it spiked at least once across T=25 steps" metric, because different neurons can win on different timesteps and the union blows up. The chosen design instead scores each neuron by its summed membrane potential over a no-gradient pass, picks the top-k per layer, and then gates both spikes and membrane by that fixed winner mask at every timestep. This guarantees the over-window active fraction per layer is at most k/width. It's an upper bound, not exact equality — some winners may not cross threshold — so we still analyse by measured activity and treat the configured fraction as the controlled condition. Gradients still flow through the kept neurons; the mask is detached so top-k is a routing decision, not a differentiable parameter.

**Implementation:** added a `sparsity_mode` config key (`'kwta_window'` vs the kept `'threshold'` fallback) and `kwta_fractions: [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]`. The model, config, and runner were all updated. The old threshold path is preserved as a selectable fallback, not deleted.

**Verification:** a quick run confirmed it works. Target fraction 0.05 produced measured activity 0.050; target 0.20 produced 0.198. Measured activity now tracks the dial, monotonically, spanning the controlled low range. That's exactly what the frozen threshold could never do.

One cost: k-WTA is roughly 2x slower per condition because of the extra no-grad scoring pass. Worth noting for planning the full run.

**Status at end of this entry:** The sparsity mechanism is now under genuine control. The full k-WTA pilot is the immediate next step.

---

### [DATE — kwta results session] | Entry 12: First controlled-axis pilot results — k-WTA works, H3 does not

The full k-WTA pilot ran cleanly. 18 conditions: 3 seeds (0, 1, 2) x 6 target activity fractions (0.01, 0.05, 0.10, 0.20, 0.30, 0.40), 10 epochs/task, naive sequential Split-MNIST, LIF-SNN 784->256->256. About 6 minutes per condition, no dead-network conditions anywhere in the sweep.

#### The main technical result: the axis is finally controlled

This is the thing that matters most from this run. Seed-averaged measured activity tracks the target fraction almost exactly and monotonically across all three seeds:

| Target fraction | Observed activity (seed avg) |
|---|---|
| 0.01 | 0.012 |
| 0.05 | 0.050 |
| 0.10 | 0.101 |
| 0.20 | 0.198 |
| 0.30 | 0.280 |
| 0.40 | 0.327 |

That's a clean ~1% to ~33% controlled range. The frozen-threshold design (Entries 8-11) could never do this — it collapsed everything into a ~0.33-0.58 blob regardless of what threshold you set. k-WTA delivers a real sparsity dial. The top target (0.40) lands at ~0.33 observed rather than 0.40 because k-WTA sets an upper bound on who may fire, and not all winners actually cross threshold. That's expected and fine — we still analyse by measured activity.

#### H2 (extreme sparsity hurts accuracy): weakly supported, soft trend not a cliff

Seed-averaged final accuracy rises from ~0.77 at ~1% activity to ~0.88 at ~33% activity. Forgetting falls from ~0.27 to ~0.15 over the same range. The sparsest condition (~1%) has the lowest accuracy and highest forgetting, but the network still learns at 1% — unlike the frozen-threshold dead-cliff where it sat at chance. So extreme sparsity degrades performance gently; it doesn't collapse it. H2 is supported in direction but the effect is a gradient, not the sharp drop the hypothesis implied.

#### H3 (inverted-U with interior peak): not supported

This is the most important scientific result, and it's a negative one. Accuracy is roughly monotonically increasing with activity (~0.77 to ~0.88) and forgetting is roughly monotonically decreasing (~0.27 to ~0.15) across the tested 1-33% range. There is no interior optimum. Denser is simply better here, at least on Split-MNIST with this LIF setup.

The proposal's central prediction — that moderate sparsity (~20-40%) would be a sweet spot, with performance degrading on both sides — does not appear in this data. That's worth being honest about. It doesn't kill the project, but it does challenge the framing. Whether an inverted-U ever appears may depend on the benchmark: Split-MNIST is easy enough that a dense network can brute-force it, and the forgetting pressure may be too mild to reveal a genuine tradeoff.

#### Mechanism (does representational overlap track forgetting?): present, correctly signed, but activity-confounded

The relationship is there and in the right direction. Seed-averaged CKA falls from ~0.016 at low activity to ~0.009 at high activity, while forgetting also falls (0.27 to 0.15). Higher activity gives both lower overlap and lower forgetting — lower overlap co-occurring with less forgetting is exactly the direction the mediation hypothesis predicts.

But this is a correlation/screening result, not a mediation result, and it's activity-confounded. Overlap and forgetting both co-vary with activity, so this run cannot separate genuine mediation from mere co-variation. That separation is what the full study's formal mediation model is for. The signal is encouraging but it would be wrong to call it evidence of mediation yet.

One other note: `overlap_cosine` stayed high (~0.73-0.99) throughout, as in earlier runs. CKA is the meaningful metric here; cosine similarity on raw spike-count vectors is too coarse to track the geometry.

#### Net verdict

The k-WTA switch is a technical success. There is now a genuinely controlled sparsity axis, which was the prerequisite for asking any of the scientific questions properly. Scientifically the run is mixed and informative: H3's inverted-U did not materialize (denser was monotonically better on this benchmark), H2 is a soft degradation rather than a cliff, and the overlap-forgetting link is present and correctly signed but still activity-confounded.

The honest read: the original "moderate sparsity is optimal" framing is challenged on Split-MNIST, and the mechanism question is now cleanly posed for the full study's mediation analysis. A benchmark harder than Split-MNIST may be needed to see whether an inverted-U ever appears — Split-MNIST may simply be too easy for the forgetting pressure to create a genuine tradeoff between sparsity and capacity.

---

### [DATE — reframing session] | Entry 13: The H3 decision and reframing the project around the mechanism

No new experiment ran this session. This entry records a decision and a docs reframing.

#### The situation

The k-WTA pilot (Entry 12) gave a clean, controlled result on Split-MNIST: accuracy rose and forgetting fell monotonically as activity increased over the ~1-33% range. There was no inverted-U. The "moderate sparsity 20-40% is optimal" prediction (H3) was not supported. That forced a decision about what to do with H3 and, more broadly, how to frame the project going forward.

#### The decision: cover all three paths

Rather than picking one response to the negative H3 result, we decided to pursue all three simultaneously.

**Path 1 — accept the negative result.** On Split-MNIST with genuinely controlled activity, there is no sparsity sweet spot. Denser is monotonically better. This is a clean corrective finding, especially against the prior archived attempt whose apparent inverted-U existed only in threshold space and vanished when keyed on achieved activity (the correlation of achieved activity with forgetting was about -0.07, essentially nothing). That earlier apparent signal was an artifact of the broken calibration design, not a real effect. Documenting this honestly is part of the contribution.

**Path 2 — test a harder benchmark.** Split-MNIST is probably too easy: the network still learns reasonably well at ~1% activity, so there is no accuracy cost to trade off against forgetting pressure. An inverted-U requires a genuine capacity-sparsity tradeoff, and Split-MNIST may not impose one. Split-CIFAR-10 is the next candidate. H3 is retained as an open question for that harder benchmark rather than discarded.

**Path 3 — pivot the headline to the mechanism.** The overlap->forgetting relationship (H4, mediation) is now the project's primary contribution. On the controlled k-WTA axis, CKA and forgetting moved together in the predicted direction: CKA fell from about 0.016 to 0.009 as activity rose, and forgetting fell alongside. But both co-vary with activity, so this screening run cannot separate genuine mediation from co-variation. The formal mediation model in the full study is the primary deliverable. That's what makes the project novel — not the sparsity sweet spot, which didn't survive contact with a properly controlled axis.

#### What actually changed this session (docs only, no code)

**`RESEARCH_IDEA_REFINED.md`:** Added a post-pilot "Framing update" paragraph to the research statement. Prepended a "Status (post-pilot)" note to each of H2, H3, and H4. H2's note records that it is weakly supported as a soft gradient rather than the predicted cliff — the network still learns at ~1% activity, no 50% collapse. H3's note records that it is not supported on Split-MNIST (monotonic, not inverted-U) and is retained for the harder-benchmark test. H4's note retitles it "H4 (PRIMARY)" and records that the precondition is met but mediation is still untested.

**`RESEARCH_REPORT.md`:** Updated the status line and added a "Framing update (post-pilot)" paragraph to the abstract so the report now leads with the mechanism and records the H3 negative result. Section 11.8 already held the k-WTA data; no changes were needed there.

#### The plan from here

Three phases, in order.

Phase A is this docs reframing. Done.

Phase B is adding Split-CIFAR-10 to the pilot. The minimal approach is a dataset toggle in the config: input dimension goes from 784 to 3072 (flatten the 32x32x3 image), keeping the same k-WTA mechanism and fraction sweep. A small convolutional SNN frontend is worth adding later only if the plain flatten-to-MLP-LIF is too weak to show any signal at all. The goal is to find out whether real capacity pressure makes an inverted-U appear.

Phase C is the formal mediation model. This is the thing that separates genuine overlap-mediation from activity co-variation. It requires estimating path a (sparsity->overlap), path b (overlap->forgetting, controlling for sparsity), the indirect effect a×b, and a bootstrap CI on that product. That's the full study's job, and it's now the headline deliverable.

One honest tension worth recording: the project's original headline — a sparsity sweet spot — did not survive contact with a properly controlled activity axis. That's a real result. The mechanism question it sharpens is arguably the more interesting contribution anyway, but it's worth being clear-eyed that the framing shifted because the data forced it, not because we planned it this way from the start.

---

### [DATE — mediation session] | Entry 14: The mediation model — the mechanism did not survive its first test

Phase C was the formal test of H4: does representational overlap actually mediate the effect of activity on forgetting, or do both just co-vary with activity? This entry records what was built, what the numbers said, and what to make of it.

#### What was built

Two new files: `src/analysis/mediation.py` and `src/scripts/run_mediation.py`. The implementation is numpy-only OLS plus a percentile bootstrap — no new dependency added to the project. The model runs on the 18 k-WTA conditions from Entry 12 (6 activity fractions x 3 seeds, Split-MNIST). All three variables — activity, overlap (CKA), and forgetting — are standardized before estimation so the path coefficients are comparable.

The model estimates five quantities: total effect c (activity -> forgetting), path a (activity -> overlap), path b (overlap -> forgetting, controlling for activity), direct effect c' (activity -> forgetting controlling for overlap), and the indirect effect a*b with a 95% percentile bootstrap CI. Results saved to `results/pilot/metrics/mediation.json`.

#### Why this was necessary

The raw k-WTA pilot (Entry 12) showed CKA overlap and forgetting both falling as activity rises. That's the right direction for the mediation story. But both quantities co-vary with activity, so a raw correlation cannot tell genuine mediation apart from spurious co-variation. The mediation model is the tool for separating them: it estimates path b — the association between overlap and forgetting *after conditioning on activity* — and if that path is near zero or wrong-signed, the apparent overlap-forgetting link is explained by the shared dependence on activity, not by overlap doing any causal work.

#### The result

Standardized estimates, mediator = overlap_cka:

- Total effect c = -0.483 (more activity -> less forgetting, as expected)
- Path a = -0.805 (more activity -> less overlap, strong and expected)
- Path b = -0.159 (weak, and the wrong sign)
- Direct effect c' = -0.611
- Indirect effect a*b = +0.128, 95% bootstrap CI [-0.586, +0.994]
- Proportion mediated = -0.265

The b-path is the critical number. Conditional on activity, more overlap associates with slightly *less* forgetting — the opposite of what the hypothesis requires. The indirect effect is positive (because a is negative and b is negative, their product is positive, meaning the mediation pathway would *increase* forgetting, not reduce it), and the bootstrap CI is wide and straddles zero by a large margin. The proportion mediated is negative, which is nonsensical and confirms the model found no coherent mediation structure.

Verdict: no evidence of mediation beyond activity co-variation.

#### What it means

The apparent overlap-tracks-forgetting signal from Entry 12 is explained by both quantities co-varying with activity. Once forgetting is conditioned on activity, overlap adds nothing — the b-path is weak and wrong-signed, and the indirect effect CI includes zero with room to spare.

This is a second negative result stacking on the H3 inverted-U non-result from Entry 12. Because Entry 13 made the mechanism the project's headline contribution, the honest current status is that both the "moderate-sparsity sweet spot" (H3) and the "overlap mediates forgetting" mechanism (H4) are unsupported at pilot scale on Split-MNIST.

That's worth sitting with. The project's original framing (H3 sweet spot) didn't survive a properly controlled activity axis. The reframing around the mechanism (H4 mediation) was the response. Now the mechanism hasn't survived its first formal test either. Two negative screens in a row.

#### The caveats that soften it

These are real, not spin.

n=18 is tiny. The study is severely underpowered for a mediation analysis, where the indirect effect (a product of two estimated paths) needs substantially more data than either path alone. The b-path estimate is noisy enough that the true value could be anywhere in a wide range.

All three variables are tightly coupled to activity. Activity, overlap, and forgetting are nearly collinear in this dataset — the CKA range is only about 0.009 to 0.016, a minuscule spread. When the mediator has almost no variance independent of the predictor, the b-path is very hard to estimate reliably. This is a near-collinearity problem, not necessarily a true null.

This was always framed as an exploratory screen, not a confirmatory test. The pilot was designed to screen for signals worth pursuing, not to confirm or disconfirm the mechanism. A negative screen means "don't count on this signal yet," not "the mechanism is false."

A full study with 8-10 seeds, a wider and more decoupled activity range, and a harder benchmark (Split-CIFAR) could still reveal mediation. The negative result sharpens the design requirements: we need to decouple activity from overlap (so the b-path has something to work with), more seeds (so the indirect effect estimate has lower variance), and capacity pressure (so forgetting is large enough to detect partial mediation).

#### Net position

The pilot has done its job. It screened two signals — the inverted-U and the mediation mechanism — and found neither on Split-MNIST at this scale. That's useful information. The full study design is now clearer: harder benchmark, more seeds, a wider and less collinear activity range, and a pre-registered confirmatory mediation test rather than an exploratory screen.

---

## Next session — start here

**Immediate:** The user's Split-CIFAR-10 run may change the picture. Under real capacity pressure, forgetting is larger and the activity-overlap-forgetting relationship may be less collinear. Check those results first before drawing any further conclusions about the mechanism.

**Full study design:** The mediation analysis needs more seeds (8-10, not 3), a wider and more decoupled activity range (so overlap has variance independent of activity), and a harder benchmark. The confirmatory mediation test should be pre-registered, not exploratory. Plan this before writing any new code.

**Git:** Phase B and Phase C code (the mediation files and any CIFAR additions) have not been committed yet. Do a proper git commit of all Phase B+C work before the next experiment run.

**Citations:** The 2026-dated arXiv preprints (Meem et al. 2026, Roy et al. 2026, Nagabhushana et al. 2026) still need manual confirmation — open the abstract pages and verify titles and authors match what's cited. Open since Entry 2, still unresolved.

**Still open from earlier entries:**

- A stray `REPORT.md` file needs to be triaged (check whether it's a duplicate of `RESEARCH_REPORT.md` or something else, and decide whether to keep or delete it).
