# Research Project: Spike Sparsity and Catastrophic Forgetting in SNNs

## Comprehensive Review Synthesis and Structured Action Plan

---

## Part I: Research Summary

### Original Research Question

**Title:** Investigating the Relationship Between Spike Sparsity and Catastrophic Forgetting in Continual Learning Spiking Neural Networks

**Primary RQ:** Does increasing spike sparsity reduce catastrophic forgetting in spiking neural networks?

**Secondary RQs:**
1. How does spike sparsity affect the retention of old training tasks?
2. What sparsity level gives the best tradeoff between accuracy, forgetting, and energy consumption?
3. Do sparse SNNs outperform equally sized ANNs in continual learning?

**Core Hypothesis:** Moderately sparse SNNs will exhibit less forgetting due to reduced interference between task representations. Extremely sparse SNNs will underperform due to insufficient representational capacity. There exists an optimal sparsity region ("Too Dense → Optimal → Too Sparse").

**Novelty Claim:** Spike Sparsity → Representation Drift → Catastrophic Forgetting (vs. standard ANN vs. SNN comparisons)

---

## Part II: Five-Reviewer Panel Assessment

### Overall Verdict: Major Revision Required

---

### Reviewer 1: Editor-in-Chief (Methodology & Scope)

| Dimension | Assessment |
|---|---|
| **Recommendation** | Minor Revision |
| **Feasibility** | Achievable in typical timeframe |
| **Computational Cost** | LOW — MLP + LIF on MNIST-scale is CPU-feasible in hours |

**Strengths:**
1. Clear, well-structured RQ hierarchy with primary/secondary distinctions
2. Novel focus on spike sparsity as a mechanistic driver of forgetting
3. Testable hypothesis with predicted inverted-U relationship
4. Multi-method comparison (4 CL strategies × multiple sparsity levels) provides systematic coverage
5. Energy proxy metric bridges neuromorphic engineering relevance

**Critical Issues:**
1. **[Major]** Sparsity manipulation via 3 different mechanisms (threshold, WTA, activity regularization) may confound comparisons — are these equivalent? Without equivalence mapping, "10-80% activity levels" across methods may not be comparable
2. **[Major]** MLP baseline is unusual for SNN work — most CL literature uses ResNet/VGG. Need justification or additional controls
3. **[Major]** CL task protocol not specified (Split CIFAR-10/100? Permuted MNIST?) — critical for reproducibility
4. **[Major]** Forgetting score definition needs specification (standard F = A_final - A_after_learning or variant?)
5. **[Minor]** LIF parameters (τ_mem, V_thresh) should be reported for reproducibility
6. **[Minor]** Statistical testing not mentioned

**Required for Acceptance:**
- Key citations: Kirkpatrick et al. (2017), Lopez-Paz & Ranzato (2017), Zenke et al. (2017), Pfeiffer & Pfeil (2018)
- Clarify task protocol with specific dataset and splitting strategy

---

### Reviewer 2: Neuromorphic Computing Expert

| Dimension | Assessment |
|---|---|
| **Expertise Fit** | Strong with neuromorphic computing literature |
| **SNN Modeling** | LIF appropriate but potentially oversimplified |

**SNN-Specific Issues:**

1. **LIF Model Simplicity** — [Major] — Vanilla LIF lacks reset dynamics and refractoriness. Consider Izhikevich, Hindmarsh-Rose, or adaptive LIF to establish whether sparsity-forgetting relationships generalize beyond simplified neurons.

2. **Missing Membrane Time Constant** — [Major] — τ_mem directly controls spike timing and intrinsic firing patterns. Without this specification, reproducibility is compromised.

3. **No Leakage-Bias Interaction Analysis** — [Minor] — Sparsity changes alter input currents and effective integration window. Interaction between sparsity mechanisms and LIF leakage not discussed.

**Sparsity Approach Evaluation:**

| Mechanism | Assessment | Notes |
|---|---|---|
| Threshold control | Appropriate but underspecified | Need threshold values and membrane dynamics relationship |
| WTA | Appropriate | Missing: k-winners, soft/hard WTA, decay timing |
| Activity regularization | Most flexible | Need regularization weight schedule across tasks |

**Biophysical Credibility:** Partially addressed. "Human brains exhibit sparse activity" is invoked but does not reference established neuroscience (olfactory bulb, hippocampal place cells, inhibitory circuits).

**Technical Recommendations:**
1. Use adaptive LIF or justify why vanilla LIF suffices
2. Report all LIF parameters explicitly
3. Connect sparsity manipulation to biological circuit mechanisms

---

### Reviewer 3: Continual Learning Expert

| Dimension | Assessment |
|---|---|
| **Expertise Fit** | Strong with continual learning literature |

**CL Methodology Issues:**

1. **Missing Gradient Projection Methods** — [Major] — PackNet, Progressive Neural Networks, Hard Attention to Task are fundamental CL baselines missing from evaluation.

2. **No Learning without Forgetting (LwF)** — [Major] — LwF and variants are standard CL baselines absent from this work.

3. **Parameter Isolation Methods** — [Minor] — SupSup or CrowdMate not included.

**Baseline Adequacy:**

| Baseline | Assessment | Notes |
|---|---|---|
| Naive | Appropriate | Standard worst-case baseline; correctly included |
| Replay | Appropriate | Must specify buffer size and memory per task; hidden state replay for SNNs |
| EWC | Appropriate | Implementation concerns around Fisher matrix estimation and λ sweep |
| Synaptic Intelligence | Appropriate | Online version (o-Si) should be used |

**Forgetting Measurement:**
- "Forgetting score = best - current" is standard Lopez-Paz "forgetting measure" but captures only **backward transfer**
- Missing: forward transfer metrics, "remembering score" (current accuracy after task T_k on task k)
- Missing evaluation metrics: average accuracy across tasks, agreement score/confusion matrix, sparsity-energy-accuracy tradeoff curve

**Task Sequence Concerns:**
- Under-specified: number of tasks, task similarity structure, task boundaries, task ordering
- Missing: task-incremental vs class-incremental vs domain-incremental setting

**Recommendations:**
1. Add PackNet or similar parameter isolation baseline
2. Add LwF or SNN variant if one exists
3. Report task similarity analysis
4. Specify exact task sequence protocol

---

### Reviewer 4: Cross-Disciplinary (ML + Neuroscience)

| Dimension | Assessment |
|---|---|
| **Cross-Disciplinary Contribution** | Moderate |

**Theoretical Grounds:**
- Core sparsity-forgetting claim: **Partial** — Plausible intuition but underdeveloped theory. Link between sparsity and interference reduction is asserted without citation to established theory (sparse coding theory, efficient coding hypotheses, complementary learning systems theory).
- Evidence: ANN-vs-SNN comparison is methodologically sound but causal mechanism is **asserted, not demonstrated**.

**Biological Credibility:**
- Work does **not** genuinely connect to neural learning mechanisms — [Superficial]
- "Human brains exhibit sparse activity" is a **shallow invocation**. Real brains achieve sparsity through inhibitory circuits (OLM cells, SOM interneurons), homeostatic plasticity, and metabolic constraints.
- No reference to experimental neuroscience findings on sparse coding.

**Interdisciplinary Value:**

| Field | Contribution | Notes |
|---|---|---|
| ML | Moderate | Systematic empirical study |
| Neuroscience | Minimal | Superficial biological framing without actual neuroscience |
| Bridge? | No | Uses neuroscience as motivation but does not inform it |

**Theory-Literature Gap:**
- Missing: computational work on representational sparsity in ANNs (Liu et al. on dead neurons, Evci et al. on lottery tickets) and relationship to interference
- Missing: sparse coding in biological systems

**Recommendations:**
1. Engage deeply with neuroscience literature or acknowledge the gap
2. Reference established sparsity theory (efficient coding, sparse coding)
3. Consider what this work contributes to **both** fields

---

### Reviewer 5: Devil's Advocate (Argument Challenges)

**Core Claim Challenges:**

**1. "Moderately sparse = less forgetting via reduced interference"**

| Aspect | Assessment |
|---|---|
| Strongest Counter | Causal chain is circular. Interference is never directly measured—only forgetting. Forgetting could arise from: (a) consolidated vs. unconsolidated representations, (b) learning rate confounds, (c) temporal correlation masking |
| Weakness | Paper likely acknowledges as limitation; ablation studies would weaken this counter |
| Verdict | **Needs More Evidence** — Claim is plausible but mechanism is asserted, not demonstrated |

**2. "Extremely sparse = insufficient capacity"**

| Aspect | Assessment |
|---|---|
| Strongest Counter | This is the **null hypothesis** of standard network capacity theory. Any neural network with too few parameters underperforms. The question is whether SNN spike sparsity introduces capacity constraints beyond what width/depth already capture |
| Weakness | If paper shows sparsity-specific capacity effects beyond neuron count, argument strengthens |
| Verdict | **Unremarkable prediction** — Needs evidence that sparsity effects are distinct from simple capacity reduction |

**3. "Optimal sparsity region exists"**

| Aspect | Assessment |
|---|---|
| Strongest Counter | The inverted-U is a generic pattern appearing across many neural network phenomena (activation functions, dropout rates, learning rates). This may not be specific to SNNs or sparsity |
| Weakness | Pattern may beANN general, not SNN-specific |
| Verdict | **Needs demonstrating SNN-specificity** — Show this pattern is not just a general neural network phenomenon |

**Novelty Challenge:**
- "Spike Sparsity → Forgetting" may be a specific instantiation of known sparsity benefits in ANNs (lottery tickets, sparse networks reducing catastrophic interference)
- Must explicitly differentiate from ANN sparsity literature

**Fatal Flaws:**
- If causal mechanism remains unproven and alternatives (learning rate confounds, random orthogonal representations) not ruled out, research is **indefensible**

**Verdict:** Major Revision Required

---

## Part III: Consolidated Gap Analysis

### Critical Issues (Must Fix)

| # | Issue | Source | Required Action |
|---|---|---|---|
| 1 | **Causal mechanism unproven** | All 5 reviewers | Add ablation studies varying sparsity post-hoc. Directly measure representational similarity (cosine similarity, PCA overlap) across tasks at different sparsity levels |
| 2 | **CL task benchmark not specified** | EIC, CL Expert | Explicitly name benchmark (Split CIFAR-10/100, Permuted MNIST, etc.). Specify # tasks, task similarity structure, ordering |
| 3 | **Sparsity methods confounded** | EIC, Neuromorphic Expert | Map sparsity levels across methods for equivalence OR analyze each mechanism separately with clear interpretation |
| 4 | **Missing foundational CL references** | EIC, CL Expert | Cite Kirkpatrick (2017), Lopez-Paz (2017), Zenke (2017), Pfeiffer & Pfeil (2018) |
| 5 | **Interference not directly measured** | Devil's Advocate | Add explicit interference measurement; do not treat as self-evident |

### Major Issues (Strongly Recommended)

| # | Issue | Source | Required Action |
|---|---|---|---|
| 6 | **ANN baseline non-standard** | EIC | Use ConvNet or justify MLP choice |
| 7 | **Missing CL methods** | CL Expert | Add LwF baseline; consider PackNet/parameter isolation |
| 8 | **LIF model may be too simple** | Neuromorphic Expert | Use adaptive LIF or justify vanilla LIF |
| 9 | **Biological plausibility superficial** | Cross-disciplinary | Connect sparsity to actual neural circuit mechanisms OR acknowledge gap |
| 10 | **Novelty overstated** | Devil's Advocate | Explicitly differentiate from ANN sparsity literature |
| 11 | **Hypotheses vague** | Multiple | Define specific sparsity regimes (1%, 5%, 10%, etc.) with numerical predictions |
| 12 | **Task sequence design incomplete** | CL Expert | Specify task-incremental vs class-incremental vs domain-incremental; task ordering protocol |

### Minor Issues (Consider)

| # | Issue | Recommendation |
|---|---|---|
| 13 | τ_mem not specified | Report all LIF parameters |
| 14 | Statistical testing not mentioned | Specify test (paired t-test, ANOVA, etc.) |
| 15 | Only backward transfer measured | Add forward transfer and average accuracy metrics |
| 16 | Energy proxy formula unclear | Define precisely: spike count × synaptic ops |
| 17 | Replay buffer size not specified | Explicitly state buffer capacity |
| 18 | Missing forward transfer metrics | Add forward transfer measurement |

---

## Part IV: Structured Action Plan

### Phase 1: Theoretical Foundation (Pre-experimental)

- [ ] **T1.1** Add direct interference measurement (cosine similarity, PCA overlap) to mechanistic analysis
- [ ] **T1.2** Add ablation: vary sparsity post-hoc to establish causality
- [ ] **T1.3** Justify sparsity-forgetting mechanism with theory citations (sparse coding theory, efficient coding, complementary learning systems)
- [ ] **T1.4** Differentiate from ANN sparsity literature (lottery tickets, sparse networks)
- [ ] **T1.5** Operationalize hypotheses with specific numerical predictions per sparsity level

### Phase 2: Experimental Design

- [ ] **E2.1** Specify CL benchmark explicitly (name, # tasks, task structure)
- [ ] **E2.2** Ensure sparsity level equivalence across manipulation methods OR analyze each separately
- [ ] **E2.3** Add LwF baseline
- [ ] **E2.4** Consider adaptive LIF OR justify vanilla LIF
- [ ] **E2.5** Use ConvNet baseline OR justify MLP choice
- [ ] **E2.6** Specify task sequence protocol (task-incremental, class-incremental, ordering)

### Phase 3: Methodology Detail

- [ ] **M3.1** Report all LIF parameters (τ_mem, V_thresh, etc.)
- [ ] **M3.2** Specify replay buffer size explicitly
- [ ] **M3.3** Define energy proxy precisely
- [ ] **M3.4** Specify statistical tests
- [ ] **M3.5** Add forward transfer and average accuracy metrics
- [ ] **M3.6** Map sparsity levels across threshold/WTA/regularization for equivalence

### Phase 4: Literature Integration

- [ ] **L4.1** Cite foundational CL papers: Kirkpatrick, Lopez-Paz, Zenke
- [ ] **L4.2** Cite SNN foundational work: Pfeiffer & Pfeil (2018)
- [ ] **L4.3** Reference biological sparsity mechanisms (inhibitory circuits, homeostatic plasticity)
- [ ] **L4.4** Add ANN sparsity literature comparison (lottery tickets, sparse networks)
- [ ] **L4.5** Add neuroscience references on sparse coding in biological systems

### Phase 5: Biological Plausibility (Optional Enhancement)

- [ ] **B5.1** Connect sparsity manipulation to inhibitory circuit mechanisms OR
- [ ] **B5.2** Explicitly acknowledge limitation that work uses neuroscience as motivation without deep biological mechanism

---

## Part V: Revised Paper Structure

Based on all reviews, the revised paper should follow this structure:

```
1. Abstract
   - Challenge → Insight → Contribution format
   - Highlight the "too dense—optimal—too sparse" finding

2. Introduction
   - 1.1 Task and Motivation (CL problem, catastrophic forgetting)
   - 1.2 SNNs as Promising Substrate (sparse, event-driven)
   - 1.3 Research Gap and Challenge (sparsity-forgetting unexplored)
   - 1.4 Research Questions (primary + secondary)
   - 1.5 Contributions (3-4 bullet points)

3. Related Work
   - 2.1 Catastrophic Forgetting in ANNs (regularization, architecture, rehearsal)
   - 2.2 Spiking Neural Networks (LIF, sparsity, neuromorphic hardware)
   - 2.3 Continual Learning in SNNs (existing work)
   - 2.4 Sparsity in Neural Networks (ANN sparsity → connection to this work)

4. Methodology
   - 3.1 Overview and Pipeline
   - 3.2 Models and Architecture (ANN baseline + SNN with LIF parameters)
   - 3.3 Sparsity Manipulation Mechanisms
     * Threshold Adjustment
     * Winner-Take-All Inhibition
     * Activity Regularization
     * Equivalence mapping across methods
   - 3.4 Continual Learning Protocols
   - 3.5 Evaluation Metrics

5. Experiments and Results
   - 5.1 Experimental Setup (datasets, training details, CL benchmark specification)
   - 5.2 Sparsity Manipulation Results
   - 5.3 Comparison with ANNs
   - 5.4 Continual Learning Methods Comparison
   - 5.5 Mechanistic Analysis (INTERFERENCE MEASUREMENT HERE)
     * Synaptic Overlap
     * Representation Drift
     * Ablation Studies

6. Discussion
   - 6.1 Interpretation of Results
   - 6.2 Relationship to ANN Sparsity Literature
   - 6.3 Biological Plausibility (or limitation acknowledgment)
   - 6.4 Limitations

7. Conclusion
   - Summary of Contributions
   - Key Findings
   - Implications
   - Future Work
```

---

## Part VI: Final Assessment

| Dimension | Status | Notes |
|---|---|---|
| **Idea Quality** | ✅ Strong | Novel, testable, timely research question |
| **Literature Coverage** | ✅ Much improved | Now cites foundational CL papers; needs ANN sparsity differentiation |
| **Methodology** | ⚠️ Needs tightening | Sparsity mechanism equivalence unclear; LIF justification needed |
| **Experimental Evidence** | ❓ Unknown | Appears planned but not visible in files |
| **Theoretical Grounding** | ⚠️ Partial | Causal mechanism asserted; interference not directly measured |
| **Biological Plausibility** | ⚠️ Superficial | Uses neuroscience motivation without deep engagement |
| **Readiness for Submission** | ⚠️ Not ready | Mechanistic analysis appears incomplete; experiments may not be run |

### Overall Verdict

**This is a promising research direction with genuine novelty and timely contribution.** The intersection of spike sparsity and catastrophic forgetting is underexplored, and the inverted-U hypothesis is intuitive and testable.

However, significant gaps remain:
1. **Causal mechanism must be proven**, not assumed
2. **Sparsity mechanisms must be equivalent-mapped** or analyzed separately
3. **Biological claims must be either deeply engaged or explicitly qualified**
4. **Novelty must be differentiated** from ANN sparsity literature

**With proper execution of Phase 1-4 above, this research has high potential for meaningful contribution.**

---

*Document generated from:*
- *Original research proposal*
- *Idea analysis (idea-analysis.md)*
- *Draft analysis (draft analysis.md)*
- *Full paper draft (full_paper.md + paper/ folder)*
- *5-Reviewer Panel Assessment (EIC + 4 peer reviewers + Devil's Advocate)*