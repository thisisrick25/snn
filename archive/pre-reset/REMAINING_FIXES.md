# Remaining fixes worklist

Worklist for hardening `RESEARCH_IDEA_REFINED.md` before submission. Fixes #1–#4 are already applied to that file. This document covers the remaining unapplied fixes (#5–#9), in priority order.

Suggested order to tackle: **#7 first** (trivial, highest credibility risk), then **#8**, then **#5**, **#6**, **#9**.

---

## Already applied (for reference — do not redo)

- **Fix #1** — Capacity/plasticity controls (Section 3.3: matched task-A mastery, representation-space forgetting via linear probe + CKA, count-matched dense baseline).
- **Fix #2** — Ablation logic corrected (Section 3.2: manipulate sparsity *during* task-B training, fixed task-A evaluation mask).
- **Fix #3** — Added harder CL setting (Section 4.3: class-incremental alongside task-incremental).
- **Fix #4** — Statistical rigor (Section 4.3 Statistical analysis subsection + Section 8.3 H3 interior-peak requirement: N≥5 seeds, mean±std, Cohen's d, Holm-Bonferroni/FDR correction).

---

## Fix #7 — Remove fabricated / unverifiable / future-dated citations

**Priority: HIGHEST. Trivial to apply. Fabricated citations sink a submission fastest.**

**Where:** Section 6 (key references list) and Section 10 (numbered references).

**Actions:**
1. **Remove `Mascoli et al. (2022)`** — currently Section 10 ref item 7 ("Mascoli et al. (2022), SNN continual learning, citation to be verified"). Unverifiable; treat as fabricated. Also appears in the Section 6 key-references list — remove it there too.
2. **Remove `arXiv:2507.18139`** ("Spike sparsity in SNNs") — currently Section 10 ref item 10. Unverifiable.
3. **Fix or remove `arXiv:2602.12236`** ("Energy-Aware Spike Budgeting for CL in SNNs") — dated Feb 2026. Real paper but future-dated. Either correct the date to the actual publication date or remove it.
4. **Keep** `Kornblith et al. (2019)` (CKA, ref item 6) — legitimate, added during Fix #1.
5. Renumber the Section 10 list after removals.

**Done when:** no "citation to be verified" placeholders remain, no future-dated entries remain, and every reference resolves to a real, findable work.

---

## Fix #8 — Add verified prior art and position against it

**Priority: HIGH. This is the novelty-defense fix.**

**Where:** Section 6 (related work / key references) and Section 10 (references).

**Context — novelty positioning:** The claim "moderate spike sparsity (20–40%) reduces forgetting in SNNs" is plausibly novel, but adjacent work exists. Frame the contribution as a **mechanistic study of how spike sparsity relates to representational overlap, and whether an intermediate sparsity regime balances plasticity vs stability**. Do NOT frame as "sparsity reduces forgetting in general" or "first SNN-CL paper to use sparse representations" — both are refutable.

**Add these 9 verified citations and position against each:**

| Citation | arXiv | Why it matters / how to position |
|---|---|---|
| **HLOP-SNN** | 2402.11984 | **MOST dangerous overlap.** Hebbian orthogonal projection reduces interference via subspace orthogonality — same mechanism family as this proposal's claim. Must explicitly distinguish: this work studies *activity sparsity* as the lever and overlap as the mediator, not an explicit orthogonal-projection constraint. |
| SOR-SNN / Adaptive Reorganization | 2309.09550 | Sparse pathways for CL. Distinguish from passive activity sparsity. |
| Active Dendrites (TTFS) | 2404.19419 | Dendritic/temporal coding for CL. |
| Bayesian-CL (SNN) | 2208.13723 | Bayesian approach to SNN continual learning. |
| DSD-SNN | 2308.04749 | Dynamic structure development in SNNs. |
| TACOS | 2409.00021 | Task-agnostic continual SNN learning. |
| SCA-SNN | 2411.05802 | Sparse / context-adaptive SNN CL. |
| Columnar SNN | 2506.17169 | Columnar architecture for CL. |
| Compressed Latent Replays | 2407.03111 | Replay-based SNN CL. |

**Done when:** Each is cited, the related-work section explicitly states how this proposal differs (especially vs HLOP-SNN), and the framing avoids the two refutable novelty claims above.

---

## Fix #5 — Pin SNN hyperparameters

**Priority: MEDIUM. Needed for reproducibility and to make the sparsity metric well-defined.**

**Where:** Section 4.1 (SNN model definition) and Section 5.3 (energy metric).

Current SNN spec already has LIF basics: `tau_mem=20ms, V_thresh=1.0, V_rest=0, V_reset=0`. Add:

1. **Surrogate gradient** — specify `atan` or `fast-sigmoid`, and state the slope/scale parameter (e.g. surrogate slope = 25 for fast-sigmoid). Required because BPTT through spikes needs a defined surrogate.
2. **Timesteps T** — pin a value (e.g. `T = 25`) and state whether the activity/sparsity metric (% active neurons) is measured **per-timestep** or **averaged over T**. This matters: T interacts with the sparsity metric, and an undefined T makes "% active" ambiguous.
3. **Input encoding** — name the scheme (rate coding / latency-temporal coding / direct current injection) and justify the choice. Affects both accuracy and the meaning of spike counts.
4. **Energy proxy** — replace the hand-wavy `spike_count × synaptic_operations` with **#SynOps × E_SynOp**, using a **cited per-operation energy figure**, compared against the ANN's **MACs × E_MAC** with its own cited figure. Without cited per-op figures the energy claim is not defensible.

**Done when:** A reader can reproduce the SNN exactly (surrogate + slope, T, encoding) and the energy comparison rests on cited per-operation costs for both SNN SynOps and ANN MACs.

---

## Fix #6 — Compute-match (FLOP-match) ANN vs SNN, not just param-match

**Priority: MEDIUM. Defends the RQ4 efficiency comparison.**

**Where:** Section 4.1 (models) / RQ4 and the efficiency-related metrics.

**Problem:** The current ANN-vs-SNN comparison is **parameter-matched only** (~260K params). An efficiency or accuracy claim can be confounded by differing compute budgets — equal parameters does not mean equal FLOPs, especially since SNNs run over T timesteps.

**Action:** Add a **FLOP/compute-matched** comparison condition so that RQ4 (sparse-SNN vs ANN) controls for compute budget, not just parameter count. Report both param-matched and compute-matched comparisons; note where conclusions differ.

**Done when:** RQ4 reports results under both param-matched and compute-matched conditions, and the efficiency claim is stated relative to a defined compute budget.

---

## Fix #9 — Reframe H1 as an explicit mediation hypothesis

**Priority: MEDIUM. Strengthens the causal/mechanistic story.**

**Where:** Hypotheses section and Section 8.1 (H1 evidence).

**Current:** H1 tests two endpoints (sparsity level → forgetting reduction) and separately measures representational overlap, but does not formally tie them as a mediation chain.

**Action:** State H1 as an explicit mediation hypothesis:

> sparsity ↑ → representational overlap ↓ → forgetting ↓

and **test the mediation** (representational overlap as the mediator between sparsity and forgetting) — e.g. a formal mediation analysis showing the sparsity→forgetting effect is carried by the reduction in overlap — rather than only correlating the two endpoints. This connects directly to the Section 3.1 overlap metrics and the Section 3.3.2 representation-space measures.

**Done when:** H1 is written as a mediation chain, and Section 8.1 lists the mediation test (not just endpoint correlations) as required evidence.

---

## Cross-cutting note

Several fixes touch the same sections (Section 4.1: #5, #6; Section 10 references: #7, #8; Section 8.1 / hypotheses: #9). If applying multiple in one pass, do **#7 then #8** together (both touch references), and **#5 then #6** together (both touch Section 4.1 models).
