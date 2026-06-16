# Investigating the relationship between spike sparsity and catastrophic forgetting in continual learning spiking neural networks

## Research statement

Catastrophic forgetting is a persistent problem in continual learning. Artificial neural networks often lose performance on earlier tasks after training on new ones. Biological brains, in contrast, learn over long periods while using sparse neural activity. Spiking neural networks (SNNs) are a useful setting for testing whether sparse, event-driven activity can reduce interference between task representations.

This project asks whether spike sparsity reduces catastrophic forgetting in SNNs, where the useful sparsity range lies, and whether the effect is explained by lower representational overlap between tasks.

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

Prediction: SNNs with less than 5% activity will show at least a 50% accuracy drop compared with dense baselines.

Mechanism: If too few neurons are active, the network may not have enough capacity to represent task-specific features.

Test: Measure classification accuracy at 1%, 5%, and 10% activity.

### H3: The sparsity-performance relationship is non-linear

Prediction: Continual learning performance will follow an inverted-U pattern, with the best performance around 20-40% activity.

Test: Fit a quadratic regression to accuracy as a function of sparsity and test for non-linearity with an F-test at p < 0.05.

## 3. Testing the proposed mechanism

A lower forgetting score would not, by itself, show that sparsity caused lower interference. The study therefore measures the proposed mechanism directly instead of treating the accuracy curve as enough evidence.

### 3.1 Representational overlap

- Cosine similarity: Measure cosine similarity between hidden-state activations for task A and task B across sparsity levels.
- PCA overlap: Project hidden representations onto principal components and measure overlap between task subspaces.
- Expected pattern: Higher sparsity should reduce cosine similarity and PCA overlap. Those reductions should line up with lower forgetting.

### 3.2 Ablation study

- Post-hoc sparsity manipulation: After training on task A, increase or decrease sparsity before training on task B and then measure forgetting.
- Control condition: Compare against networks where sparsity is not changed after task A.
- Expected pattern: If sparsity has a causal role, changing sparsity after task A should change forgetting on task A after learning task B.

### 3.3 Learning-rate control

- Confound check: Keep the main learning rate fixed across sparsity levels.
- Sensitivity analysis: Repeat key experiments with learning rates of 0.001, 0.0001, and 0.00001.

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
| Spike threshold | Higher theta produces fewer spikes | 1%, 5%, 10%, 20%, 40%, 60%, 80%, 95% |
| Winner-take-all | Only the top-k neurons fire | Matched to the same activity levels |
| Activity regularization | A penalty discourages high activity | Regularization strength tuned to the same activity levels |

Validation: Each mechanism will be tuned independently to reach the target activity levels. Results will be reported both by activity level and by sparsity mechanism, so the analysis does not treat mechanistically different interventions as identical.

### 4.3 Continual learning protocol

#### Benchmarks

- Split-MNIST: Five binary classification tasks, digits 0-1, 2-3, 4-5, 6-7, and 8-9
- Permuted-MNIST: Ten tasks, each using a different random pixel permutation of MNIST
- Optional extension: CIFAR-10 or CIFAR-100 after the MNIST-scale experiments are complete

These benchmarks are standard in continual learning and allow comparison with prior work.

#### Task setting

- Setting: Task-incremental learning, with task labels available at inference time
- Task order: Fixed order for reproducibility
- Task boundaries: Explicit boundaries between tasks

#### Training details

- Optimizer: Adam with learning rate 0.001
- Batch size: 128 for MNIST and 64 for CIFAR experiments
- Epochs: 10 per MNIST task and 20 per CIFAR task
- Statistical testing: Paired t-tests at p < 0.05 for comparisons across sparsity levels, plus F-tests for non-linearity in the sparsity-performance curve

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

## 6. Related work foundation

### Key references

- Kirkpatrick et al. (2017): Elastic Weight Consolidation for catastrophic forgetting
- Lopez-Paz and Ranzato (2017): Gradient episodic memory for continual learning
- Zenke et al. (2017): Synaptic Intelligence
- Pfeiffer and Pfeil (2018): Review of deep learning with spiking neurons
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
- Cosine similarity between task representations decreases as sparsity increases

### 8.2 H2: Extreme sparsity

Prediction: Activity below 5% reduces accuracy by at least 50%.

Evidence needed:

- Accuracy at 1% and 5% activity is less than 0.5 x baseline accuracy, with p < 0.05

### 8.3 H3: Non-linear tradeoff

Prediction: Performance follows an inverted-U curve, with the best range near 20-40% activity.

Evidence needed:

- Quadratic regression shows significant non-linearity, with p < 0.05
- The estimated peak lies within the 20-40% activity range

## 9. Limitations and future work

### 9.1 Current limitations

1. Simplified neuron model: LIF lacks adaptive thresholds and refractory-period dynamics. Later work should test adaptive LIF or Izhikevich models.
2. Benchmark scope: Split-MNIST and Permuted-MNIST are useful first benchmarks but relatively simple. Later work should test CIFAR-10, CIFAR-100, and more realistic datasets.
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
6. Mascoli et al. (2022), SNN continual learning, citation to be verified
7. Olshausen and Field (1996), Emergence of simple-cell receptive field properties by learning a sparse code
8. Buzsaki (2006), Rhythms of the brain
9. arXiv:2507.18139, Spike sparsity in SNNs
10. arXiv:2602.12236, SNN continual learning
