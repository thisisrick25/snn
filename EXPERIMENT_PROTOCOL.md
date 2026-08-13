# Experiment protocol

This protocol defines the first empirical path for testing whether controlled spike sparsity reduces catastrophic forgetting in continual learning spiking neural networks. The goal is not to show state-of-the-art performance. The goal is to test a narrower causal claim: changing spike sparsity changes representational overlap between tasks, and that overlap predicts forgetting.

## 1. Core claim to test

The main claim is:

> Moderate spike sparsity can reduce catastrophic forgetting in SNNs by reducing representational overlap between sequentially learned tasks.

This claim requires three kinds of evidence:

1. A performance effect: forgetting changes across sparsity levels.
2. A mechanism effect: representational overlap changes across sparsity levels.
3. A link between the two: lower overlap predicts lower forgetting.

**Correlation vs mediation (scope note).** The word "by" in the claim above is a *mediation* claim: sparsity is asserted to reduce forgetting *through* reduced representational overlap. A significant correlation between overlap and forgetting across sparsity levels is a necessary precondition but does **not** by itself establish mediation. The distinction drives a two-stage design:

- **Pilot (this document, §2):** screens only the correlation precondition (evidence types 1-3 above) on the minimal threshold-only naive Split-MNIST setup with 3 seeds. It is feasibility-only and cannot support a mediation claim.
- **Full study (RESEARCH_IDEA_REFINED.md §3.5):** runs the formal mediation model (indirect effect sparsity -> overlap -> forgetting with bootstrap CI, conditional on confounds), across multiple sparsity mechanisms, with 8-10 confirmatory seeds. The final mediation claim rests here, not on the pilot.

The pilot is designed to *falsify early*: if the correlation precondition fails at pilot scale, the mediation hypothesis is not worth the full study.

## 2. Pilot experiment

The pilot should be small enough to run quickly and strict enough to falsify the idea early.

### 2.1 Pilot scope

| Component | Pilot choice | Reason |
|---|---|---|
| Dataset | Split-MNIST | Simple, standard continual learning benchmark |
| Setting | Task-incremental learning | Clear first setting with task labels available at inference |
| Model | LIF-SNN only | Tests the core SNN hypothesis before adding ANN controls |
| Continual learning method | Naive sequential learning | Exposes forgetting without mitigation from replay or regularization |
| Sparsity mechanism | Spike-threshold control | Simplest direct way to change firing activity |
| Sparsity levels | 1%, 10%, 20%, 40%, 80% active neurons | Covers extreme sparse, moderate sparse, and dense-ish regimes |
| Seeds | At least 3 | Required to avoid seed-specific conclusions |

### 2.2 Split-MNIST task sequence

Use five binary tasks:

1. Task 1: digits 0 vs 1
2. Task 2: digits 2 vs 3
3. Task 3: digits 4 vs 5
4. Task 4: digits 6 vs 7
5. Task 5: digits 8 vs 9

Use a fixed task order for the first pilot. Randomized task orders can be added later as a robustness check.

### 2.3 Pilot model

Use a feedforward LIF-SNN implemented in `snntorch`.

Recommended starting configuration:

- Input: flattened MNIST image
- Hidden layer 1: 256 LIF neurons
- Hidden layer 2: 256 LIF neurons
- Output: task-specific binary classifier or shared output with task label provided
- Membrane time constant: tau_mem = 20 ms
- Firing threshold: tuned to achieve target activity level
- Resting potential: V_rest = 0.0
- Reset potential: V_reset = 0.0
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs per task: 10

### 2.4 Sparsity calibration

For each target activity level, tune the firing threshold before the continual learning run.

Target activity is defined as:

```text
active_neuron_percentage = number_of_neurons_that_spike_at_least_once / total_neurons
```

Calibration procedure:

1. Train or warm up on the first task for a short calibration pass.
2. Sweep threshold values.
3. Select the threshold that produces the closest observed activity to the target level.
4. Keep that threshold fixed during the continual learning run.
5. Record the observed activity level, not only the target level.

The analysis should use observed activity whenever possible.

## 3. Measurements

### 3.1 Accuracy and forgetting

After each task, evaluate the model on all tasks learned so far.

Store an accuracy matrix `A`, where `A[i, j]` is the accuracy on task `j` after training task `i`.

Report:

1. Final average accuracy: average accuracy over all tasks after the final task.
2. Per-task forgetting: max accuracy on a task during training minus final accuracy on that task.
3. Mean forgetting: average forgetting over all tasks except the final task.
4. Backward transfer: average effect of later task training on earlier task performance.

### 3.2 Spike sparsity and energy proxy

Report these for each task and sparsity level:

1. Average spike rate per neuron per time step.
2. Percentage of inactive neurons.
3. Percentage of active neurons.
4. Total spike count.
5. Energy proxy: spike_count x synaptic_operations.

The energy proxy is not a hardware energy measurement. It should be described as a computational estimate only.

### 3.3 Representational overlap

Measure hidden representations after each task. Use the same held-out evaluation subset for all sparsity levels and seeds.

Recommended representation sources:

1. Hidden-layer spike counts over the full simulation window.
2. Hidden-layer membrane potentials averaged over time.
3. Final-time hidden state, if the implementation exposes it reliably.

Report at least two overlap measures:

1. Cosine similarity between task-mean representations.
2. PCA subspace overlap between task representations.

Optional additional measure:

3. CKA similarity between hidden representations, especially if cosine similarity is too coarse.

### 3.4 Mechanism link

The key plot should connect sparsity, overlap, and forgetting.

Minimum analyses (pilot):

1. Plot observed activity level vs mean forgetting.
2. Plot observed activity level vs representational overlap.
3. Plot representational overlap vs mean forgetting.
4. Report correlation between overlap and forgetting across sparsity levels and seeds.

At pilot scale these are **screening** analyses only. The pilot establishes whether the correlation precondition holds; it does not attempt to estimate mediation, and its correlation coefficients must not be reported as confirmatory evidence.

**Full-study mediation analysis (not part of the pilot).** In the full study the mechanism link is evaluated with a formal mediation model rather than correlation alone (see RESEARCH_IDEA_REFINED.md §3.5 for the full specification). In brief: with X = observed activity level, M = predefined primary cross-task overlap, Y = mean forgetting, and covariates (sparsity mechanism, task setting, matched task-A mastery, seed as random effect), estimate path a (X -> M), path b (M -> Y | X), the direct effect c' (X -> Y | M), and the indirect effect a*b with a bootstrap CI resampled at the seed/run level; report the proportion mediated. The mechanism claim is supported only if the indirect effect a*b is significantly non-zero after conditioning on covariates. A significant total effect with a non-significant indirect effect indicates sparsity acts through a non-overlap pathway (plasticity or capacity), which would weaken the novelty. Mediation is estimated per sparsity mechanism first and pooled only if per-mechanism estimates agree.

## 4. Pilot decision criteria

Use the pilot to decide whether the project should expand, narrow, or change direction.

### 4.1 Continue to full experiments if

Continue if most of the following are true:

1. Forgetting is lower at moderate sparsity than at dense or near-dense activity.
2. Extreme sparsity hurts accuracy.
3. Representational overlap decreases as sparsity increases.
4. Representational overlap correlates with forgetting.
5. The pattern is visible across at least 3 seeds.

### 4.2 Revise the hypothesis if

Revise if any of the following happen:

1. Sparsity changes spike rate but not forgetting.
2. Sparsity changes forgetting but not representational overlap.
3. The best result occurs only at extreme sparsity.
4. Results differ strongly across seeds.
5. The model fails to learn Split-MNIST reliably.

### 4.3 Stop or reframe if

Reframe the project if the pilot shows no reliable relationship among sparsity, overlap, and forgetting. In that case, the paper may still become a negative result or a study of why spike sparsity alone is insufficient for continual learning.

## 5. Full experiment expansion

Only expand after the pilot produces interpretable results.

### 5.1 Add baselines

Add these models after the LIF-SNN pilot:

1. MLP with matched parameter count.
2. Simplified ConvNet.
3. Optional Conv-SNN if resources allow.

Purpose:

- MLP tests whether the effect is specific to spike sparsity or also appears with ordinary activation sparsity.
- ConvNet checks whether the conclusion survives a more standard image model.
- Conv-SNN checks whether the SNN effect survives convolutional feature extraction.

### 5.2 Add continual learning methods

Add methods in this order:

1. Replay buffer, buffer size = 200.
2. Elastic Weight Consolidation.
3. Synaptic Intelligence.
4. Learning without Forgetting.
5. Optional PackNet or another parameter-isolation method.

Purpose:

- Naive learning shows the raw forgetting effect.
- Replay tests whether sparsity still matters when old examples are available.
- EWC and SI test whether sparsity interacts with parameter-importance regularization.
- LwF tests whether sparsity interacts with distillation-based retention.

### 5.3 Add datasets

Add datasets in this order:

1. Permuted-MNIST.
2. Rotated-MNIST, optional.
3. CIFAR-10 split or CIFAR-100 split, only after MNIST-scale findings are stable.

Purpose:

- Split-MNIST tests class-incremental structure in a simple setting.
- Permuted-MNIST tests domain shifts without changing labels.
- CIFAR tests whether the effect survives harder visual inputs.

### 5.4 Add sparsity mechanisms

After threshold control, add:

1. Winner-take-all sparsity.
2. Activity regularization.

Each mechanism must be calibrated to the same observed active-neuron percentages. Results should be reported separately by mechanism and jointly by observed activity level.

## 6. Figures and tables to produce

### 6.1 Pilot figures

1. Accuracy matrix for each sparsity level.
2. Observed activity level vs final average accuracy.
3. Observed activity level vs mean forgetting.
4. Observed activity level vs representational overlap.
5. Representational overlap vs mean forgetting.

### 6.2 Full experiment figures

1. Heatmap of forgetting across sparsity levels and continual learning methods.
2. Accuracy-forgetting-energy tradeoff curve.
3. Comparison of threshold, winner-take-all, and activity regularization at matched activity levels.
4. ANN vs SNN comparison at matched parameter count and matched activity level where possible.
5. Mechanism summary plot linking activity, overlap, and forgetting.

### 6.3 Tables

1. Model architecture and parameter counts.
2. Dataset and task protocol.
3. Sparsity calibration table showing target activity, observed activity, and threshold value.
4. Final accuracy, forgetting, spike rate, and energy proxy for each condition.
5. Statistical test results.

## 7. Statistical plan

**Seed guidance.** The pilot uses 3 seeds for *feasibility only*; pilot p-values are never reported as confirmatory evidence and exist only to guide design decisions. Confirmatory conditions in the full study use 8-10 seeds; secondary and exploratory conditions use at least 5. Seeds vary initialization, data shuffling, and task order; report mean +/- std with no single-run estimates.

**Confirmatory test hierarchy (full study).** Hypotheses are assigned to predefined families and corrected within each family (mirrors RESEARCH_IDEA_REFINED.md §4.3):

1. **Primary confirmatory family** - H1/H2/H3 and the H4 mediation claim on ONE predefined configuration: Split-MNIST + task-incremental + naive sequential + spike-threshold mechanism. Corrected with Holm-Bonferroni.
2. **Secondary mechanism family** - the same hypotheses re-tested across the winner-take-all and activity-regularization mechanisms. Corrected with Holm-Bonferroni within-family.
3. **Exploratory family** - other datasets (Permuted-MNIST, CIFAR), other CL methods (replay, EWC, SI, LwF, PackNet), the ANN comparison (RQ4), and alternative overlap metrics. Corrected with Benjamini-Hochberg FDR and reported as exploratory, not confirmatory.

**Mechanism separation.** Because spike-threshold, winner-take-all, and activity-regularization produce different kinds of sparsity at the same active-neuron percentage, all confirmatory tests (inverted-U quadratic fits and mediation) are estimated per mechanism first. Curves and mediation estimates are pooled only when the per-mechanism results agree in direction and shape.

Recommended tests:

1. Paired comparisons between moderate sparsity and dense activity for forgetting (report Cohen's d with CI).
2. Quadratic regression for the inverted-U relationship between activity and performance, with a bootstrap CI on the fitted peak location that must exclude the range boundaries (per-mechanism before any pooled fit).
3. Formal mediation model (indirect effect a*b with bootstrap CI) for the overlap-mediates-forgetting claim; correlation/regression between overlap and forgetting is a precondition screen only.
4. Confidence intervals across seeds for all main plots.

All analyses and plots use the OBSERVED active-neuron percentage, never the target. Avoid overclaiming p-values from the pilot. The pilot guides design decisions; the full experiment supports the final claim.

## 8. Implementation checklist

- [ ] Implement or confirm Split-MNIST task loader.
- [ ] Implement LIF-SNN baseline.
- [ ] Add threshold calibration for target activity levels.
- [ ] Log spike counts and active-neuron percentages.
- [ ] Save accuracy matrix after each task.
- [ ] Save hidden representations for overlap analysis.
- [ ] Compute forgetting and final average accuracy.
- [ ] Compute cosine similarity and PCA overlap.
- [ ] Plot sparsity vs forgetting.
- [ ] Plot overlap vs forgetting.
- [ ] Run at least 3 seeds.
- [ ] Decide whether to expand, revise, or reframe.

## 9. Claim boundaries

The first paper should avoid broad claims until the full grid is complete.

Acceptable pilot-stage claim:

> In a controlled Split-MNIST LIF-SNN setting, moderate spike sparsity is associated with lower representational overlap and lower forgetting.

Acceptable full-study claim if results support it:

> Across multiple sparsity controls and continual learning methods, the relationship between spike sparsity and forgetting is non-linear, and representational overlap explains part of this relationship.

Claims to avoid:

1. SNNs solve catastrophic forgetting.
2. Sparsity always improves continual learning.
3. LIF results generalize to all SNN models.
4. Spike-count energy proxies prove hardware energy efficiency.

## 10. Immediate next action

Start with the pilot only:

1. Implement Split-MNIST.
2. Implement the LIF-SNN.
3. Calibrate threshold-based sparsity at 1%, 10%, 20%, 40%, and 80% active neurons.
4. Run naive sequential learning with 3 seeds.
5. Produce the five pilot plots listed in Section 6.1.

Do not add Replay, EWC, SI, LwF, WTA, activity regularization, Permuted-MNIST, or ANN baselines until the pilot produces interpretable results.
