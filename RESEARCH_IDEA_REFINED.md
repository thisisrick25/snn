# Investigating the relationship between spike sparsity and catastrophic forgetting in continual learning spiking neural networks

## Research statement

Catastrophic forgetting is a persistent problem in continual learning. Artificial neural networks often lose performance on earlier tasks after training on new ones. Biological brains, in contrast, learn over long periods while using sparse neural activity. Spiking neural networks (SNNs) are a useful setting for testing whether sparse, event-driven activity can reduce interference between task representations.

This project asks whether spike sparsity reduces catastrophic forgetting in SNNs, where the useful sparsity range lies, and whether the effect is explained by lower representational overlap between tasks.

**Framing update (post-pilot).** An initial controlled-activity pilot on Split-MNIST reshaped the emphasis of this project. Using winner-take-all gating to hold spike activity at fixed, directly-controlled levels (k-WTA, see Section 4.2), the predicted inverted-U in performance did not appear: over the reachable ~1-33% activity range, denser activity was monotonically better (higher accuracy, lower forgetting), with no interior optimum. The "moderate sparsity is a sweet spot" hypothesis (H3) is therefore not supported on Split-MNIST, and is retained as an open question for a harder benchmark (Split-CIFAR) where capacity pressure is real. Consequently the project's primary contribution is now the *mechanism* question (H4/RQ5): whether reduced representational overlap mediates the sparsity-forgetting relationship. The pilot found overlap (CKA) and forgetting move together in the predicted direction, but both co-vary with activity, so a formal mediation model is required to separate genuine mediation from co-variation. That mediation analysis is the study's headline result, not the shape of the sparsity-performance curve. (See RESEARCH_JOURNAL.md Entries 11-13 and RESEARCH_REPORT.md Section 11.8 for the pilot evidence.)

## 1. Research questions

### 1.1 Primary question

RQ1: Does increasing spike sparsity reduce catastrophic forgetting in spiking neural networks?

### 1.2 Secondary questions

RQ2: How does spike sparsity affect retention of previously learned tasks across different continual learning protocols?

RQ3: What sparsity level gives the best tradeoff among:

- Classification accuracy, with a target of at least 90% of baseline performance
- Forgetting rate, with a target of no more than 20% degradation from peak performance
- Energy efficiency, estimated from spike count and synaptic operations

RQ4: Do sparse SNNs at their best sparsity level outperform equally parameterized ANN baselines on continual learning benchmarks?

RQ5: What mechanism links spike sparsity to reduced catastrophic forgetting?

## 2. Hypotheses

### H1: Moderate spike sparsity reduces forgetting

Prediction: SNNs with 20-40% activity will show at least a 30% reduction in forgetting compared with dense baselines.

Mechanism: Sparse activity should reduce overlap between task representations. When fewer neurons are active, fewer weight updates should interfere with representations learned for earlier tasks.

Test: Measure the forgetting score, defined as best_accuracy - current_accuracy, across sparsity levels from 1% to 95%.

### H2: Extreme sparsity harms accuracy

Status (post-pilot): WEAKLY supported on Split-MNIST as a soft gradient, not the predicted cliff. Under controlled k-WTA activity, accuracy declined gently toward the sparse end (about 0.88 at ~33% activity to about 0.77 at ~1%), and the network still learned well at ~1% activity - far from a 50% collapse. The steep drop the prediction anticipated did not occur on Split-MNIST, consistent with the task being easy enough that even ~1% active neurons carry enough capacity. (Note: this contrasts sharply with the failed frozen-threshold approach, where the sparsest setting produced a *dead* network at chance accuracy - that was a threshold artifact, not a genuine capacity limit; see Section 4.2 and RESEARCH_JOURNAL.md Entry 11.)

Prediction (original): SNNs with less than 5% activity will show at least a 50% accuracy drop compared with dense baselines.

Mechanism: If too few neurons are active, the network may not have enough capacity to represent task-specific features.

Test: Measure classification accuracy at 1%, 5%, and 10% activity.

### H3: The sparsity-performance relationship is non-linear

Status (post-pilot): NOT SUPPORTED on Split-MNIST. On the controlled-activity k-WTA pilot (3 seeds x 6 target activity fractions spanning observed ~1-33%), continual-learning performance was monotonic, not inverted-U: accuracy increased with activity (about 0.77 at ~1% activity up to about 0.88 at ~33%) and forgetting decreased (about 0.27 down to about 0.15), with no interior peak. Denser activity was simply better across the reachable range. This is a genuine negative result against the sweet-spot prediction, and it corrects a prior-attempt artifact in which an apparent inverted-U existed only in firing-threshold space and vanished when keyed on achieved activity (see RESEARCH_JOURNAL.md Entry 12 and the archive comparison).

Prediction (original, retained for the harder-benchmark test): Continual learning performance will follow an inverted-U pattern, with the best performance around 20-40% activity.

Open question: The inverted-U may still appear where capacity pressure is real. Split-MNIST is easy enough that a two-hidden-layer LIF network has ample capacity even at ~1% activity, so there is no accuracy penalty for extreme sparsity to trade against. H3 is therefore retained as a test for a harder benchmark (Split-CIFAR), not as a claim about Split-MNIST.

Test: Fit a quadratic regression to accuracy as a function of sparsity and test for non-linearity with an F-test at p < 0.05.

Interior-peak requirement: A significant quadratic term is necessary but not sufficient. The fitted vertex must lie strictly inside the tested activity range (inside (5%, 95%), ideally within 20-40%), with a bootstrap confidence interval on the peak location that excludes the range boundaries (see Section 8.3). A curve that merely flattens or bends at an extreme can also produce a significant quadratic coefficient and does not count as support for an inverted-U.

Mechanism-separation requirement: The inverted-U must be established per sparsity mechanism (threshold, winner-take-all, activity regularization) before any pooled curve is reported. Mechanistically different interventions are not pooled unless their individual curves agree (see Section 4.2).

### H4 (PRIMARY): Representational overlap mediates the sparsity-forgetting relationship

This is now the project's headline hypothesis. With the inverted-U (H3) not supported on Split-MNIST, the central contribution is no longer the shape of the sparsity-performance curve but the *mechanism*: whether reduced representational overlap is what links sparsity to reduced forgetting.

Prediction: The effect of spike sparsity on forgetting is at least partially mediated by reduced representational overlap between tasks. That is, increasing sparsity reduces overlap (path a), and lower overlap predicts lower forgetting conditional on sparsity (path b), such that the indirect effect (a x b) is significantly different from zero.

Status (post-pilot): TESTED at pilot scale, NOT SUPPORTED. The raw precondition held - CKA overlap and forgetting moved together in the predicted direction (CKA fell from about 0.016 to about 0.009 as activity rose, and forgetting fell alongside it) - but a formal exploratory mediation model (Section 3.5; n = 18 conditions, 3 seeds x 6 activity fractions, numpy OLS with seed-level bootstrap) found no evidence of mediation beyond activity co-variation. Standardized paths: total effect c = -0.483; a-path (activity -> overlap) = -0.805; b-path (overlap -> forgetting | activity) = -0.159, which is weak and wrong-signed for the hypothesis; direct effect c' = -0.611; indirect effect a x b = +0.128 with a 95% bootstrap CI of [-0.586, +0.994] that straddles zero. Conditional on activity, overlap adds nothing: the apparent overlap-forgetting link is explained by both quantities co-varying with activity, not by overlap mediating the effect. This is a second negative screening result (alongside H3), and it directly undercuts the "central novelty" framing below. Caveats keep it from being fatal: n = 18 is tiny and underpowered, the three variables are near-collinear (all tightly coupled to activity, so the b-path is hard to estimate), and the CKA range is minuscule (about 0.009-0.016). Mediation therefore remains the study's primary open question, deferred to the full study (8-10 seeds, a decoupled activity range, and a harder benchmark such as Split-CIFAR); it is no longer a claim the pilot supports.

Rationale: This is the intended central novelty of the project, though the pilot screen above did not yet support it. Prior SNN continual-learning work already establishes that sparse activation, threshold modulation, and spike budgeting can reduce forgetting; what remains under-characterized is *why*. Establishing overlap as a mediator (rather than merely a correlate) is what would distinguish this study from occupied territory (e.g., Shen et al. 2024) - but that hinges on a properly powered mediation test that the pilot could not deliver. Correlation across sparsity levels is not mediation.

Test: A formal mediation model estimating the indirect effect with a bootstrap confidence interval, conditional on confounds (see Section 3.5). A significant negative correlation between overlap and forgetting is a necessary precondition but is not, by itself, accepted as evidence of mediation.

## 3. Testing the proposed mechanism

A lower forgetting score would not, by itself, show that sparsity caused lower interference. The study therefore measures the proposed mechanism directly instead of treating the accuracy curve as enough evidence.

### 3.1 Representational overlap

- Cosine similarity: Measure cosine similarity between hidden-state activations for task A and task B across sparsity levels.
- PCA overlap: Project hidden representations onto principal components and measure overlap between task subspaces.
- Expected pattern: Higher sparsity should reduce cosine similarity and PCA overlap. Those reductions should line up with lower forgetting.

### 3.2 Ablation study

Interference is created *while* task B is being learned, not after. The ablation therefore manipulates sparsity during task-B training rather than masking the network after task A.

- During-task-B manipulation: Train task A to a fixed mastery, then increase or decrease the active-unit sparsity *during* task-B training, which is when weight updates from task B can overwrite task-A structure.
- Fixed evaluation mask: Keep the task-A evaluation mask fixed across all conditions, so that the measurement probe for task-A accuracy does not change when the training-time sparsity changes. This ensures any observed difference reflects altered interference during learning, not a different readout at test time.
- Control condition: Compare against networks where the task-B-time sparsity is left at its default level (no manipulation), with the same fixed task-A evaluation mask.
- Why not post-hoc masking: Masking the network after task A changes which units the readout reads from (a readout effect) without changing the interference that occurred during task-B learning. Manipulating sparsity during task B, with a fixed evaluation mask, isolates the interference mechanism the hypothesis is about.
- Expected pattern: If sparsity has a causal role in reducing interference, increasing active-unit overlap during task-B training should increase task-A forgetting, and increasing sparsity during task-B training should reduce it, with the task-A evaluation mask held constant.

### 3.3 Capacity and plasticity controls

A lower forgetting score at moderate sparsity could arise for reasons unrelated to reduced interference. Two confounds must be ruled out before any causal claim:

- Capacity confound: A sparser network may simply learn *less* of task A, so there is less to forget. Lower forgetting would then reflect weaker initial learning, not better retention.
- Plasticity confound: With fewer active neurons, fewer weights are updated during task B. Lower forgetting would then reflect *fewer updates*, a trivial plasticity effect, rather than sparse coding reducing representational interference.

The study controls for both as follows.

#### 3.3.1 Matched task-A mastery

- Requirement: Report task-A peak accuracy (immediately after training on task A, before any task-B training) at every sparsity level.
- Analysis rule: Forgetting comparisons across sparsity levels are only valid conditional on equal task-A mastery. Where peak accuracies differ, report forgetting as a function of task-A mastery (for example, by stratifying or covarying on peak accuracy) rather than comparing raw forgetting scores at unequal starting points.
- Expected pattern: If sparsity reduces interference, lower forgetting should persist at moderate sparsity even after task-A mastery is matched.

#### 3.3.2 Forgetting in representation space

Output-level accuracy conflates two distinct effects: degradation of the internal representation and reshuffling of the readout. The study measures forgetting in representation space to separate them.

- Linear probe protocol: After training task A, freeze the feature extractor and train a linear probe on each task. After training task B, re-probe with the same frozen-then-probed protocol. A drop in probe accuracy indicates the *representation itself* degraded, independent of the output head.
- Centered Kernel Alignment (CKA): Compute CKA (Kornblith et al. 2019) between task-A hidden representations measured before and after task-B training. High CKA with low output accuracy indicates "representation preserved, readout reshuffled"; low CKA indicates genuine representational forgetting.
- Expected pattern: If sparsity reduces representational interference, moderate sparsity should show higher pre/post-B CKA and smaller probe-accuracy drops than dense baselines.
- Three distinct quantities must be reported separately and not conflated:
  1. Cross-task representational overlap: similarity between the representations of *different* tasks (task A vs task B), measured at a fixed layer. This is the proposed mediator (see Section 3.5). High cross-task overlap implies competing tasks share coding subspace.
  2. Representation drift: change in task A's *own* representation before vs after task-B training (e.g., pre/post-B CKA on task-A inputs). This measures how much task A's internal code was overwritten.
  3. Decodability: linear-probe accuracy on frozen features. This measures whether task information remains linearly recoverable regardless of drift. A representation can drift substantially yet remain decodable, or stay stable yet lose decodability; these are not interchangeable.

#### 3.3.3 Count-matched dense baseline

- Control: Construct a dense network in which a random subset of weights is frozen during task B, matched in count to the number of weights the sparse network leaves unupdated at each sparsity level.
- Logic: This isolates "fewer weights updated" (a plasticity effect) from "sparse distributed coding reduces overlap" (the proposed mechanism).
- Expected pattern: If the count-matched dense-frozen network forgets as little as the sparse network, the effect is attributable to fewer updates, not to sparse coding. If the sparse network forgets less than the count-matched dense-frozen control, this supports a genuine sparse-coding mechanism.
- Limitation: This control matches the *quantity* of unupdated weights but not the *geometry* of which weights are updated or how large those updates are. The following controls (3.3.4-3.3.6) address geometry and update magnitude, which count-matching alone leaves open.

#### 3.3.4 Update-norm-matched control

- Motivation: Sparse activation reduces not only how many parameters are updated but also the *magnitude* of gradients flowing through inactive units. Lower forgetting could therefore reflect smaller/noisier weight updates rather than sparse coding reducing overlap.
- Control: Match the total per-task update norm (sum of squared weight deltas, or per-layer update norm) between the sparse network and a dense comparison network, e.g. by scaling the dense network's learning rate or gradient norm so that its cumulative update magnitude equals the sparse network's at each sparsity level.
- Logic: Isolates "smaller effective updates" from "sparse coding reduces overlap."
- Instrumentation: Log per-task and per-layer gradient norms, update norms, and cross-task gradient cosine similarity for every run.
- Expected pattern: If forgetting differences vanish once update norm is matched, the mechanism is update-magnitude driven, not overlap driven.

#### 3.3.5 Activation-dropout control

- Motivation: Sparsity mechanisms deactivate units. A dense network with random unit dropout can reproduce the same fraction of active units without any learned, input-dependent sparse code.
- Control: Apply random per-unit dropout to a dense network calibrated to match the *observed* active-neuron percentage of the sparse network at each level.
- Logic: Isolates "fewer units active" (a stochastic capacity effect) from "structured, input-dependent sparse coding."
- Expected pattern: If activation-dropout at matched active% reproduces the forgetting reduction, the effect does not require a learned sparse code. If the sparse network still forgets less, this supports structured sparse coding.

#### 3.3.6 Structured (neuron-block) freezing control

- Motivation: The count-matched control (3.3.3) freezes a *random* subset of weights. Sparse coding may instead protect *contiguous, neuron-aligned* subnetworks (capacity partitioning), which random freezing does not emulate.
- Control: Freeze structured neuron-aligned blocks (whole units / channels) during task B, matched to the effective unupdated-unit count of the sparse network.
- Logic: Distinguishes capacity partitioning (protecting whole neurons) from distributed sparse coding (reducing overlap across shared neurons).
- Expected pattern: Comparing random-weight freezing (3.3.3), structured-neuron freezing (3.3.6), and true sparsity separates "which units are protected" from "how the code is distributed."

### 3.4 Learning-rate control

- Confound check: Keep the main learning rate fixed across sparsity levels.
- Sensitivity analysis: Repeat key experiments with learning rates of 0.001, 0.0001, and 0.00001.

### 3.5 Formal mediation model

Correlating sparsity, overlap, and forgetting across sparsity levels does not establish mediation. To test H4 (Section 2), the study fits a formal mediation model that estimates the indirect effect of sparsity on forgetting *through* representational overlap, conditional on confounds.

Primary mediator (predefined): cross-task representational overlap between task A and task B measured at the primary layer *during task-B training*. This single quantity is the confirmatory mediator. CKA, cosine similarity, PCA overlap, and linear-probe-based measures serve as robustness/sensitivity checks, not as additional confirmatory mediators.

Variables:
- Treatment X = observed activity level (active-neuron percentage; observed, never target).
- Mediator M = primary cross-task overlap (as above).
- Outcome Y = forgetting (mean forgetting score F over prior tasks).
- Covariates/controls: sparsity mechanism, task setting (task-IL vs class-IL), task-A mastery (matched peak accuracy / loss / probe accuracy), with seed as a random effect.

Model:
- Path a: regress M on X + mechanism + task setting + task-A mastery + (1 | seed).
- Path b and direct effect: regress Y on M + X + mechanism + task setting + task-A mastery + (1 | seed). Coefficient on M = path b; coefficient on X = direct effect c'.
- Indirect (mediated) effect = a x b, reported with a bootstrap confidence interval (resampling at the seed/run level).
- Report: path a, path b, direct effect c', indirect effect a x b with bootstrap CI, and the proportion of the total effect that is mediated.

Decision rule: H4 is supported only if the indirect effect a x b is significantly different from zero (bootstrap CI excludes zero) after conditioning on the covariates above. A significant total effect with a non-significant indirect effect indicates sparsity acts through a non-overlap pathway (e.g., reduced plasticity or capacity), which would weaken the central novelty claim.

Mechanism separation: The mediation model is estimated per sparsity mechanism first. Mechanisms are pooled only if their individual a, b, and indirect effects are consistent (see Section 4.2).

## 4. Methodology

### 4.1 Models

#### ANN baseline: Multi-layer perceptron

- Architecture: Input, hidden layer with 256 units, hidden layer with 256 units, output layer
- Activation: ReLU
- Parameter count: About 260K, matched to the SNN where possible
- Rationale: CNNs are standard for image tasks, but an MLP makes it easier to isolate sparsity effects without convolutional feature sharing. A simplified ConvNet can be used as a secondary validation model.

#### SNN baseline: Leaky integrate-and-fire network

- Library: snntorch
- Neuron model: LIF
- Membrane time constant: tau_mem = 20 ms
- Firing threshold: V_thresh = 1.0
- Resting potential: V_rest = 0.0
- Reset potential after spike: V_reset = 0.0
- Rationale: LIF is the simplest standard SNN model and is suitable for an initial test of spike sparsity. Adaptive LIF or Izhikevich neurons can be used later to test whether the findings hold with richer neuron dynamics.

### 4.2 Sparsity manipulation

The three sparsity mechanisms may not be equivalent, so each will be calibrated to the same observed activity metric: percentage of active neurons.

| Mechanism | How sparsity is controlled | Target activity levels |
|---|---|---|
| Spike threshold | Higher theta produces fewer spikes | 1%, 5%, 10%, 20%, 30%, 40%, 60%, 80%, 95% |
| Winner-take-all | Only the top-k neurons fire | Matched to the same activity levels |
| Activity regularization | A penalty discourages high activity | Regularization strength tuned to the same activity levels |

Mechanism non-equivalence (critical): The three mechanisms are distinct interventions, not interchangeable ways of reaching "the same sparsity." Matching them on a single scalar (percentage of active neurons) does not make them equivalent, because they differ in *which* neurons are silenced and *how* the surviving code is distributed:

- Spike threshold raises the firing bar uniformly; whichever neurons exceed it fire, so the active set is input-driven and can be highly overlapping across inputs.
- Winner-take-all (top-k) enforces a hard cardinality per step; exactly k neurons fire regardless of input magnitude, producing competition and often more decorrelated codes.
- Activity regularization applies a soft penalty during training; sparsity emerges as a learned equilibrium and may concentrate in particular units.

Because these produce different *kinds* of sparsity at the same active-neuron percentage, results are analyzed per mechanism first. No pooled sparsity-forgetting or sparsity-overlap curve is reported unless the individual per-mechanism curves agree in shape and direction. This requirement is stated for H3 (Section 2) and for the mediation model (Section 3.5).

Per-mechanism reporting metrics: For every (mechanism, activity level, seed) condition, the following are reported separately rather than collapsed into a single activity scalar, since two mechanisms can share an active-neuron percentage yet differ on all of these:

- Active-neuron percentage (the calibration target metric; observed, not target)
- Total spike count and mean spike rate
- Firing-rate entropy across neurons (how evenly activity is distributed vs concentrated in a few units)
- Lifetime sparsity (per-neuron fraction of inputs on which the neuron is active)
- Per-neuron participation / dead-neuron fraction (how many units are effectively unused)

Validation: Each mechanism is tuned independently to reach the target activity levels. Results are reported both by activity level and by sparsity mechanism, so the analysis never treats mechanistically different interventions as identical.

### 4.3 Continual learning protocol

#### Benchmarks

- Split-MNIST: Five binary classification tasks, digits 0-1, 2-3, 4-5, 6-7, and 8-9
- Permuted-MNIST: Ten tasks, each using a different random pixel permutation of MNIST
- Optional extension: CIFAR-10 or CIFAR-100 after the MNIST-scale experiments are complete

These benchmarks are standard in continual learning and allow comparison with prior work.

#### Task setting

Continual-learning difficulty depends strongly on the setting. Task-incremental learning, where the task label is provided at inference time, is the easiest setting and maximally suppresses forgetting because a separate output head is selected per task. Reporting only task-incremental results risks overstating how well sparsity controls forgetting. The study therefore evaluates two settings of increasing difficulty.

- Primary setting (task-incremental): Task labels available at inference time, per-task output heads. Used for mechanism analysis and as the baseline for comparison with prior work.
- Harder setting (class-incremental): No task labels at inference time; a single classifier head grows to cover all classes seen so far, and the model must discriminate across tasks without knowing which task a sample belongs to. This is the setting where forgetting is most severe, and it provides the stronger test of whether moderate sparsity reduces forgetting.
- Rationale: If the sparsity effect holds in the class-incremental setting, the result is far more convincing than a task-incremental-only finding. Where the two settings disagree, that difference is itself an informative result about the limits of the mechanism.
- Task order: Fixed order for reproducibility (a seed-level order sensitivity check is included in the statistical analysis).
- Task boundaries: Explicit boundaries between tasks.

Note: Split-MNIST is run in both settings. Class-incremental Split-MNIST uses a single 10-way head; task-incremental Split-MNIST uses per-task binary heads.

#### Training details

- Optimizer: Adam with learning rate 0.001
- Batch size: 128 for MNIST and 64 for CIFAR experiments
- Epochs: 10 per MNIST task and 20 per CIFAR task
- Replication: Confirmatory conditions (the primary confirmatory family below) are run with 8-10 random seeds; secondary and exploratory conditions use at least 5 seeds. Seeds vary network initialization, data shuffling, and task order (for the order-sensitivity check). All reported numbers are mean +/- standard deviation across seeds; single-run point estimates are not reported. (The pilot uses 3 seeds for feasibility only and its p-values are never reported as confirmatory evidence.)
- Grid: The sparsity sweep includes a 30% activity point (grid: 1%, 5%, 10%, 20%, 30%, 40%, 60%, 80%, 95%) to give the inverted-U fit adequate resolution near the hypothesized 20-40% optimum.

#### Statistical analysis

- Sample size and dispersion: N >= 5 seeds per condition; report mean +/- std for every metric.
- Effect sizes: Report Cohen's d alongside every pairwise comparison, so the magnitude of an effect is reported, not only its significance. Forgetting-reduction claims (for example "30% reduction") are accompanied by both a p-value and an effect size with a confidence interval.
- Significance tests: Paired t-tests (or Wilcoxon signed-rank where normality is violated) for comparisons across sparsity levels, with seeds as the unit of replication.
- Confirmatory test hierarchy (predefined): To avoid an uncontrolled multiple-comparison surface, tests are partitioned into three predefined families, declared before analysis:
  - Primary confirmatory family: The core hypotheses (H1, H2, H3, and the H4 mediation claim) evaluated on ONE predefined primary configuration - one dataset (Split-MNIST), one task setting (the primary/task-incremental setting), one continual-learning method (naive sequential), and one sparsity mechanism (spike threshold). This is the family that decides whether the central claims are supported. Corrected with Holm-Bonferroni.
  - Secondary mechanism family: The same hypotheses evaluated across the other sparsity mechanisms (winner-take-all, activity regularization) to test mechanism generality. Corrected within-family (Holm-Bonferroni).
  - Exploratory family: Everything else - additional datasets (Permuted-MNIST, CIFAR), additional CL methods (replay, EWC, SI, LwF, PackNet), the ANN comparison (RQ4), and alternative overlap metrics. Corrected with Benjamini-Hochberg FDR and reported as exploratory, not confirmatory.
- Multiple-comparison correction: Within each family above, all p-values are corrected (Holm-Bonferroni for the primary/secondary families, Benjamini-Hochberg FDR for the exploratory family). Corrected p-values are reported and the correction family is stated explicitly for every table.
- Observed vs target activity: All statistical models and plots use the OBSERVED active-neuron percentage per condition, never the calibration target.
- Effect sizes: Report Cohen's d with a confidence interval alongside every pairwise comparison, in every family.
- Non-linearity (H3): Quadratic regression with an F-test for the quadratic term (see Section 8.3 for the interior-peak requirement), estimated per mechanism before any pooled fit (Section 4.2).

### 4.4 Continual learning methods

| Method | Type | Reason for inclusion |
|---|---|---|
| Naive sequential learning | Baseline | Reference condition for maximum forgetting |
| Replay buffer | Rehearsal | Strong practical baseline, buffer size = 200 |
| Elastic Weight Consolidation | Regularization | Classic continual learning method using Fisher-based parameter importance |
| Synaptic Intelligence | Regularization | Online parameter-importance method with biological motivation |
| Learning without Forgetting | Distillation | Standard distillation-based continual learning baseline |
| PackNet | Parameter isolation | Optional architecture-based comparison |

Gradient projection and parameter isolation methods can be included in the extended evaluation if the core experiments show a clear sparsity effect.

## 5. Evaluation metrics

### 5.1 Forgetting metrics

1. Backward transfer (BWT): Average effect of learning later tasks on earlier task performance.
2. Forgetting score (F): max_accuracy - current_accuracy for each task.
3. Forward transfer (FWT): Effect of learning current tasks on later task performance.

### 5.2 Accuracy metrics

1. Average task accuracy: Mean accuracy across tasks after sequential training.
2. Final average accuracy: Mean accuracy across all tasks at the end of training.

### 5.3 Sparsity and efficiency metrics

1. Spike rate: Average spikes per neuron per time step.
2. Sparsity index: Percentage of inactive neurons.
3. Energy proxy: spike_count x synaptic_operations, following common neuromorphic efficiency estimates.

### 5.4 Mechanistic metrics

1. Cosine similarity between task-specific hidden representations.
2. PCA overlap between task representation subspaces.
3. Synaptic overlap, measured as correlation in weight updates between tasks.
4. Centered Kernel Alignment (CKA) between task-A representations before and after task-B training, separating representation preservation from readout change.
5. Linear-probe accuracy on frozen features per task, measured before and after task-B training, to quantify representation-space forgetting independent of the output head.

## 6. Related work foundation

### Key references

- Kirkpatrick et al. (2017): Elastic Weight Consolidation for catastrophic forgetting
- Lopez-Paz and Ranzato (2017): Gradient episodic memory for continual learning
- Zenke et al. (2017): Synaptic Intelligence
- Pfeiffer and Pfeil (2018): Review of deep learning with spiking neurons
- Kornblith et al. (2019): Similarity of neural network representations revisited (CKA), used here to measure representation-space forgetting
- Mascoli et al. (2022): Recent work on SNN continual learning, citation to be verified

### Difference from ANN sparsity work

This project focuses on spike sparsity rather than weight sparsity. The distinction matters for four reasons:

1. Spike sparsity is temporal activation sparsity, not structural sparsity in the parameter matrix.
2. SNNs add time-dependent membrane and spike dynamics that are absent from ordinary ANN activation sparsity.
3. Spike counts map more directly to energy use on neuromorphic hardware.
4. Spike sparsity has a biological analogue, although the experiments here do not claim full biological fidelity.

## 7. Biological interpretation

The biological motivation needs a narrow interpretation. Sparse activity is common in biological neural systems, but the mechanisms that produce it are complex. They include inhibitory circuits, homeostatic plasticity, and metabolic constraints. Threshold tuning, winner-take-all rules, and activity penalties are simplified computational controls, not direct models of those mechanisms.

The paper should therefore treat biological sparsity as motivation for the hypothesis, not as evidence that the proposed SNN model is biologically realistic. Relevant neuroscience references include Olshausen and Field (1996) on sparse coding and Buzsaki (2006) on neural rhythms.

## 8. Expected results and required evidence

### 8.1 H1: Moderate sparsity

Prediction: Activity levels of 20-40% reduce forgetting by at least 30% compared with dense baselines.

Evidence needed:

- Forgetting score at 20-40% activity is less than 0.7 x the forgetting score at 100% activity, with p < 0.05
- The forgetting reduction persists after matching task-A peak accuracy across sparsity levels (Section 3.3.1), ruling out the capacity confound
- The sparse network forgets less than a count-matched dense-frozen control (Section 3.3.3), ruling out the plasticity confound
- Representation-space evidence: higher pre/post-B CKA and smaller linear-probe accuracy drops at 20-40% activity than at dense baselines (Section 3.3.2)
- Cosine similarity between task representations decreases as sparsity increases

### 8.2 H2: Extreme sparsity

Prediction: Activity below 5% reduces accuracy by at least 50%.

Evidence needed:

- Accuracy at 1% and 5% activity is less than 0.5 x baseline accuracy, with p < 0.05

### 8.3 H3: Non-linear tradeoff

Prediction: Performance follows an inverted-U curve, with the best range near 20-40% activity.

Evidence needed:

- Quadratic regression shows a significant negative quadratic term, with p < 0.05 after multiple-comparison correction
- Interior-peak requirement: The estimated peak (vertex of the fitted curve) lies strictly in the interior of the tested range, inside (5%, 95%) activity, and ideally within the predicted 20-40% band. A significant quadratic term whose vertex falls at or outside the range boundary does not count as support for an inverted-U, because a curve that merely flattens or bends at an extreme can also yield a significant quadratic coefficient.
- The peak location is reported with a confidence interval (for example via bootstrap over seeds), not as a point estimate.

## 9. Limitations and future work

### 9.1 Current limitations

1. Simplified neuron model: LIF lacks adaptive thresholds and refractory-period dynamics. Later work should test adaptive LIF or Izhikevich models.
2. Benchmark scope: Split-MNIST and Permuted-MNIST are useful first benchmarks but relatively simple. Later work should test CIFAR-10, CIFAR-100, and more realistic datasets. (Difficulty of the continual-learning *setting* is addressed within this study by evaluating both task-incremental and the harder class-incremental setting, per Section 4.3; the remaining limitation is dataset complexity, not setting difficulty.)
3. Sparsity mechanisms: Threshold tuning, winner-take-all selection, and activity regularization may produce different representational effects even at matched activity levels.
4. Biological simplification: The sparsity controls are computational approximations and do not model the full biological basis of sparse coding.

### 9.2 Future directions

1. Adaptive sparsity: Allow sparsity to change during training based on task difficulty.
2. Recurrent architectures: Extend the experiments to recurrent SNNs for temporal sequence tasks.
3. Neuromorphic hardware validation: Test energy-efficiency claims on hardware such as Intel Loihi or IBM TrueNorth.
4. Biological experiments: Collaborate with neuroscience labs to test whether biological sparsity correlates with reduced interference in relevant learning settings.

## 10. References

1. McCloskey and Cohen (1989), Catastrophic interference in connectionist networks
2. Kirkpatrick et al. (2017), Overcoming catastrophic forgetting in neural networks
3. Lopez-Paz and Ranzato (2017), Gradient episodic memory for continual learning
4. Zenke et al. (2017), Continual learning through synaptic intelligence
5. Pfeiffer and Pfeil (2018), Deep learning with spiking neurons
6. Kornblith, Norouzi, Lee, and Hinton (2019), Similarity of neural network representations revisited, ICML
7. Mascoli et al. (2022), SNN continual learning, citation to be verified
8. Olshausen and Field (1996), Emergence of simple-cell receptive field properties by learning a sparse code
9. Buzsaki (2006), Rhythms of the brain
10. arXiv:2507.18139, Spike sparsity in SNNs
11. arXiv:2602.12236, SNN continual learning
