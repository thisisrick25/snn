# Investigating the Relationship Between Spike Sparsity and Catastrophic Forgetting in Continual Learning Spiking Neural Networks

## Version 2.0 — Addressing Peer Review Feedback

---

## Research Statement

**Catastrophic forgetting remains a fundamental barrier to deploying neural networks in real-world continual learning scenarios.** Artificial neural networks (ANNs) rapidly lose previously acquired knowledge upon training on new tasks, while biological brains exhibit remarkable lifelong learning alongside inherently sparse neural activity. Spiking neural networks (SNNs), which naturally manifest sparse, event-driven communication, present an intriguing hypothesis: that the very sparsity that makes SNNs energy-efficient may also confer resilience against catastrophic forgetting by reducing interference between task representations.

**This research investigates whether increasing spike sparsity reduces catastrophic forgetting in SNNs, identifies the optimal sparsity region, and establishes the causal mechanism linking sparsity to reduced interference.**

---

## 1. Research Questions (Refined)

### 1.1 Primary Question
**RQ1:** Does increasing spike sparsity reduce catastrophic forgetting in spiking neural networks?

### 12. Secondary Questions (Operationalized)
**RQ2:** How does spike sparsity affect retention of previously learned tasks across different continual learning protocols?

**RQ3:** What sparsity level provides the optimal trade-off between:
- Classification accuracy (≥ 90% of baseline performance)
- Forgetting rate (≤ 20% degradation from peak performance)
- Energy efficiency (spike count × synaptic operations)

**RQ4:** Do sparse SNNs (at optimal sparsity) outperform equally parameterized ANN baselines in continual learning benchmarks?

**RQ5:** What is the causal mechanism linking spike sparsity to reduced catastrophic forgetting?

---

## 2. Hypotheses (Testable & Falsifiable)

### H1: Moderately Sparse SNNs Exhibit Reduced Forgetting
**Prediction:** SNNs with 20-40% activity levels will show ≥ 30% reduction in forgetting compared to dense baselines.

**Mechanism:** Sparse activity reduces representational overlap between tasks. When fewer neurons are active, the probability of weight updates interfering with previously learned representations decreases.

**Test:** Measure forgetting score (F = best_accuracy - current_accuracy) across sparsity levels from 1% to 95%.

### H2: Extremely Sparse SNNs Underperform Due to Capacity Constraints
**Prediction:** SNNs with < 5% activity levels will show ≥ 50% accuracy degradation compared to dense baselines.

**Mechanism:** Insufficient active neurons cannot represent task-specific features, leading to underfitting.

**Test:** Measure classification accuracy at 1%, 5%, and 10% activity levels.

### H3: Optimal Sparsity Region Exists
**Prediction:** An inverted-U relationship exists between sparsity and continual learning performance, with peak performance at 20-40% activity levels.

**Test:** Fit a quadratic regression to accuracy vs. sparsity and test for non-linearity (F-test, p < 0.05).

---

## 3. Addressing the Causal Mechanism (New Section)

**Challenge from Reviewers:** Correlation ≠ Causation. Sparsity may correlate with reduced forgetting without causing it.

**Solution:** We will directly measure the proposed causal mechanism:

### 3.1 Representational Overlap Measurement
- **Cosine Similarity:** Measure cosine similarity between hidden state activations for task A and task B across different sparsity levels.
- **PCA Overlap:** Project representations onto principal components and measure overlap in subspaces.
- **Predicted Pattern:** Higher sparsity → lower cosine similarity → reduced forgetting

### 3.2 Ablation Study
- **Post-hoc Sparsity Manipulation:** After training on task A, artificially increase/decrease sparsity before task B and measure forgetting.
- **Control:** Compare to networks where sparsity is not manipulated.
- **Expected:** Sparsity manipulation post-training should still affect forgetting, establishing causality.

### 3.3 Learning Rate Control
- **Confound Check:** Ensure equal learning rates across sparsity levels.
- **Sensitivity Analysis:** Test if findings hold across learning rates (0.001, 0.0001, 0.00001).

---

## 4. Methodology (Addressing Confounds)

### 4.1 Models

#### ANN Baseline → Multi-Layer Perceptron (MLP)
- **Architecture:** Input → Hidden(256) → Hidden(256) → Output
- **Activation:** ReLU
- **Parameters:** ~260K (matched to SNN for fair comparison)
- **Justification:** While CNNs are standard for image tasks, MLPs allow direct comparison of sparsity effects without convolutions confounding the analysis. Results will be validated on a simplified ConvNet.

#### SNN Baseline → Leaky Integrate-and-Fire (LIF)
- **Library:** snntorch
- **Neuron Model:** LIF with parameters:
  - τ_mem = 20ms (membrane time constant)
  - V_thresh = 1.0 (firing threshold)
  - V_rest = 0.0 (resting potential)
  - V_reset = 0.0 (reset potential after spike)
- **Justification for LIF:** LIF is the foundational SNN model. While adaptive LIF or Izhikevich neurons offer richer dynamics, LIF captures the core sparsity mechanism. Future work will extend to adaptive models.

### 4.2 Sparsity Manipulation (Addressing Equivalence)

**Challenge from Reviewers:** Three different sparsity mechanisms create different topologies (neuron-level, population-level, weight-level).

**Solution:** Map all mechanisms to a common metric: **percentage of active neurons**.

| Mechanism | How It Controls Sparsity | Target Activity Levels |
|---|---|---|
| **Spike Threshold** | Higher θ → fewer spikes | 1%, 5%, 10%, 20%, 40%, 60%, 80%, 95% |
| **Winner-Take-All** | Only top-k neurons fire | Matched to target activity levels above |
| **Activity Regularization** | Penalty on high activity | λ tuned to achieve target activity levels |

**Validation:** Each mechanism will be calibrated independently to achieve the same target activity levels. A consistency check will verify that all three mechanisms produce comparable results at equivalent activity levels.

### 4.3 Continual Learning Protocol

#### Benchmark: Split-MNIST & Permuted-MNIST
- **Split-MNIST:** 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9)
- **Permuted-MNIST:** 10 tasks, each a random pixel permutation of MNIST
- **Justification:** Standard CL benchmarks enabling comparison with prior work.

#### Task Protocol
- **Task-Incremental:** Task labels provided during inference (standard CL setup)
- **Task Ordering:** Fixed (not random) to ensure reproducibility
- **Task Boundaries:** Explicit (known when switching tasks)

#### Training Details
- **Optimizer:** Adam (lr = 0.001)
- **Batch Size:** 128 (MNIST), 64 (CIFAR-10)
- **Epochs:** 10 per task (MNIST), 20 (CIFAR)
- **Statistical Testing:** Paired t-test (p < 0.05) for significance testing across sparsity levels

### 4.4 Continual Learning Methods (Expanded Based on Review)

| Method | Type | Reason for Inclusion |
|---|---|---|
| **Naive Sequential Learning** | Baseline | Reference for maximum forgetting |
| **Replay Buffer** | Rehearsal | Strongest practical baseline (buffer size = 200) |
| **Elastic Weight Consolidation (EWC)** | Regularization | Classic CL algorithm; Fisher matrix estimation |
| **Synaptic Intelligence (SI)** | Regularization | Online parameter importance; biologically inspired |
| **Learning without Forgetting (LwF)** | Distillation | Standard baseline; knowledge distillation across tasks |
| **PackNet (Optional)** | Parameter Isolation | Represents architecture-based methods |

**Missing Methods Acknowledged:** Parameter isolation (PackNet, Progressive Networks) and gradient projection methods will be included in extended evaluation.

---

## 5. Evaluation Metrics (Refined)

### 5.1 Forgetting Metrics
1. **Backward Transfer (BWT):** Average accuracy on previous tasks after learning all tasks
2. **Forgetting Score (F):** F = max_accuracy - current_accuracy (per task)
3. **Forward Transfer (FWT):** Average accuracy improvement on future tasks after learning current task

### 5.2 Accuracy Metrics
1. **Average Task Accuracy:** Mean accuracy across all tasks after sequential training
2. **Final Average Accuracy:** Accuracy averaged over all tasks at the end of training

### 5.3 Sparsity & Efficiency Metrics
1. **Spike Rate:** Average spikes per neuron per time step
2. **Sparsity Index:** Percentage of inactive neurons
3. **Energy Proxy:** E = spike_count × synaptic_operations (following neuromorphic literature conventions)

### 5.4 Mechanistic Metrics
1. **Cosine Similarity:** Between task-specific hidden representations
2. **PCA Overlap:** Shared variance in principal component subspaces
3. **Synaptic Overlap:** Correlation in weight updates between tasks

---

## 6. Related Work Foundation (Addressing Missing Citations)

### Key References (Required for Context)
- **Kirkpatrick et al. (2017):** Overcoming catastrophic forgetting in neural networks (EWC)
- **Lopez-Paz & Ranzato (2017):** Gradient episodic memory for continual learning
- **Zenke et al. (2017):** Continual learning through synaptic intelligence
- **Pfeiffer & Pfeil (2018):** Deep learning with spiking neurons (SNN review)
- **Mascoli et al. (2022):** Recent SNN continual learning work

### Explicit Differentiation from ANN Sparsity
**Challenge from Reviewers:** How is this different from ANN sparsity work (lottery tickets, sparse networks)?

**Response:**
1. **Spike Sparsity ≠ Weight Sparsity:** Spike sparsity refers to temporal sparsity in neural activations, not structural sparsity in weights.
2. **Temporal Dynamics:** SNN spike sparsity introduces a temporal dimension not present in ANN activation sparsity.
3. **Energy Efficiency:** SNN spike sparsity directly maps to energy consumption on neuromorphic hardware.
4. **Biological Grounding:** Spike sparsity is a biologically observed phenomenon with direct neural correlates.

---

## 7. Biological Plausibility (Addressed)

**Challenge from Reviewers:** Biological claims are superficial.

**Resolution:** Two possible approaches:

### Option A: Deep Engagement (Preferred)
- Reference specific biological mechanisms: inhibitory circuits (OLM cells, SOM interneurons), homeostatic plasticity, and metabolic constraints as sources of biological sparsity.
- Connect to experimental neuroscience: sparse coding in olfactory bulb, hippocampal place cells.
- Cite: Olshausen & Field (1996) sparse coding; Buzsáki (2006) neural syntax.

### Option B: Explicit Qualification
- Acknowledge that biological sparsity mechanisms are complex and not fully captured by our simplified sparsity controls.
- Frame biological inspiration as motivation, not claim of biological fidelity.
- **Recommended for this draft:** Option B (to avoid overreach)

---

## 8. Expected Results & Predictions

### 8.1 Primary Hypothesis (H1)
**Expected:** Moderate sparsity (20-40% activity) shows ≥ 30% reduction in forgetting compared to dense baselines.

**Evidence Needed:**
- Forgetting score at 20-40% sparsity < 0.7 × forgetting score at 100% sparsity (p < 0.05)
- Cosine similarity between task representations decreases with increasing sparsity

### 8.2 Secondary Hypothesis (H2)
**Expected:** Sparsity < 5% shows ≥ 50% accuracy degradation.

**Evidence Needed:**
- Accuracy at 1% and 5% sparsity < 0.5 × baseline accuracy (p < 0.05)

### 8.3 Tertiary Hypothesis (H3)
**Expected:** Inverted-U relationship with peak at 20-40% sparsity.

**Evidence Needed:**
- Quadratic regression shows significant non-linearity (F-test, p < 0.05)
- Peak accuracy within 20-40% sparsity range

---

## 9. Limitations & Future Work

### 9.1 Current Limitations
1. **Simplified Neuron Model:** LIF lacks adaptive thresholds and refractory period dynamics. Future work will use adaptive LIF or Izhikevich models.
2. **Benchmark Scope:** Split-MNIST and Permuted-MNIST are relatively simple. Future work will extend to CIFAR-10/100 and real-world datasets.
3. **Sparsity Mechanisms:** Three different sparsity mechanisms may create different topological effects. Future work will analyze each mechanism separately.
4. **Biological Simplification:** Our sparsity controls are simplified compared to biological sparse coding mechanisms.

### 9.2 Future Directions
1. **Adaptive Sparsity:** Allow sparsity to adapt during training based on task difficulty.
2. **Recurrent Architectures:** Extend to recurrent SNNs (RSNNs) for temporal sequence tasks.
3. **Neuromorphic Hardware Validation:** Validate energy efficiency claims on Intel Loihi or IBM TrueNorth.
4. **Biological Experiments:** Collaborate with neuroscience labs to test if biological sparsity correlates with reduced interference.

---

## 10. References

1. McCloskey & Cohen (1989) — Catastrophic interference in connectionist networks
2. Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in neural networks (EWC)
3. Lopez-Paz & Ranzato (2017) — Gradient episodic memory for continual learning
4. Zenke et al. (2017) — Continual learning through synaptic intelligence
5. Pfeiffer & Pfeil (2018) — Deep learning with spiking neurons
6. Mascoli et al. (2022) — SNN continual learning (TBD)
7. Olshausen & Field (1996) — Emergence of simple-cell receptive field properties by learning a sparse code
8. Buzsáki (2006) — Rhythms of the brain
9. [arXiv:2507.18139] — Spike sparsity in SNNs
10. [arXiv:2602.12236] — SNN continual learning

---

## Appendix: Reviewer Concern → Our Response Mapping

| Reviewer Concern | Our Response in This Draft |
|---|---|
| **Causal mechanism unproven** | Added Section 3: Direct measurement of representational overlap + ablation studies |
| **Sparsity methods confounded** | Section 4.2: All mechanisms mapped to common "percentage active neurons" metric |
| **Benchmark not specified** | Section 4.3: Split-MNIST & Permuted-MNIST explicitly named |
| **Missing LwF baseline** | Section 4.4: LwF added to evaluation |
| **LIF too simple** | Section 4.1: Justified as foundational; adaptive LIF in future work |
| **Biological plausibility superficial** | Section 7: Explicit qualification of biological claims |
| **Novelty vs. ANN sparsity** | Section 6: Explicit differentiation from ANN sparsity literature |
| **Hypotheses vague** | Section 2: Hypotheses operationalized with specific numerical predictions |
| **Task sequence unclear** | Section 4.3: Task-incremental, fixed ordering, explicit boundaries |
| **Statistical testing missing** | Section 4.3 & 5: Paired t-test, F-test for non-linearity specified |

---

**Document Status:** Ready for experimental validation.

**Next Step:** Implement experiments according to methodology, collect results, and iterate based on empirical findings.