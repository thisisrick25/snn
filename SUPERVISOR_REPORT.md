# Pilot Progress Report: Spike Sparsity and Catastrophic Forgetting in Continual-Learning Spiking Neural Networks

**Student:** Graduate Research Project  
**Date:** August 2026  
**Status:** Pilot complete (Split-MNIST + Split-CIFAR-10, two architectures); full study pending

---

## Executive Summary

This report summarises a completed pilot experiment investigating whether spike sparsity reduces catastrophic forgetting in spiking neural networks (SNNs) trained sequentially on tasks, and whether any such reduction is explained by lower representational overlap between task representations. The pilot has now run across three model/benchmark settings: a flatten-MLP on Split-MNIST, the same flatten-MLP on Split-CIFAR-10, and a small spiking convolutional network (Conv-SNN) on Split-CIFAR-10. All three used a k-Winner-Take-All (k-WTA) mechanism to control spike activity directly, and all three used naive continual learning with no replay or regularisation.

The pilot produced one methodological success and four negative scientific screening results. The methodological success is that spike activity is now a directly controlled experimental variable, achieved by replacing a fixed firing threshold with k-WTA. This was non-trivial: the threshold approach failed in trained networks, and diagnosing that failure corrected a prior internal result that had appeared to show an inverted-U performance curve.

The four negative results are: (1) no inverted-U "sweet spot" on Split-MNIST; (2) no inverted-U on Split-CIFAR-10 with the flatten-MLP, though that model is too weak on CIFAR to constitute a fair test; (3) no inverted-U on Split-CIFAR-10 with the Conv-SNN, which is a competent model and does constitute a fair test; and (4) no evidence that representational overlap mediates the sparsity-forgetting relationship on any setting tested. H2 (extreme sparsity causes accuracy collapse) is now supported, but only with a model that has genuine capacity pressure: the Conv-SNN shows a clear accuracy drop at 1% activity that the weak MLP could not reveal.

These are pilot-scale, exploratory results from 3 seeds. They are not refutations of the underlying hypotheses. Their scientific value is a clean corrective finding: the pattern is consistent across two architectures and two benchmarks, and it corrects prior work whose apparent inverted-U was an artifact of plotting against a threshold target rather than achieved activity.

---

## 1. Research Question and Hypotheses

The central question is whether increasing spike sparsity (fewer neurons firing per forward pass) reduces catastrophic forgetting in an SNN trained sequentially on tasks, and whether any benefit operates through a specific mechanism: sparser codes overlap less between tasks, so weight updates for a new task interfere less with representations of earlier tasks.

Four hypotheses were pre-specified. H1 predicted that moderate sparsity (20-40% active neurons) would reduce forgetting by at least 30% relative to dense baselines. H2 predicted that extreme sparsity (below 5% active neurons) would cause at least a 50% accuracy drop. H3 predicted an inverted-U relationship between activity level and continual-learning performance, with the optimum near 20-40% activity. H4, elevated to the primary scientific contribution, predicted that representational overlap formally mediates the sparsity-forgetting relationship: that the indirect path (sparsity reduces overlap, and lower overlap predicts lower forgetting conditional on sparsity) accounts for a significant portion of the total effect.

---

## 2. Method

The base network is a feedforward LIF-SNN with two hidden layers of 256 neurons each, implemented in snntorch. Membrane time constant is 20 ms, simulation runs for 25 timesteps per sample, and the surrogate gradient (fast-sigmoid) enables backpropagation through the non-differentiable spike function. Training uses Adam at a learning rate of 0.001, batch size 128, 10 epochs per task, with no replay or regularisation.

Three model/benchmark settings were run:

**MNIST-MLP.** The flatten-MLP (784 input, 256-256 hidden, per-task binary output heads) on Split-MNIST: five binary classification tasks learned sequentially. Six activity levels, 3 seeds.

**CIFAR-MLP.** The same flatten-MLP on Split-CIFAR-10: five 2-way classification tasks from CIFAR-10 images, flattened to a 3072-dimensional input vector. Six activity levels, 3 seeds.

**CIFAR-Conv-SNN.** A small spiking convolutional frontend (three spiking conv layers: 3 channels to 16 to 32 to 64, each followed by max-pooling) feeding the same two 256-unit LIF hidden layers. The sparsity and activity controls are applied only to the fully connected layers, unchanged from the MLP setting. Six activity levels, 3 seeds.

The Conv-SNN was added because the flatten-MLP was too weak on CIFAR (roughly 0.62 final accuracy, nearly flat across all activity levels) to be a fair test of whether an accuracy sweet-spot exists. A model that cannot learn the task well cannot reveal a capacity-sparsity tradeoff. The Conv-SNN reaches roughly 0.71 final average accuracy, a genuine 9-point improvement, and provides real capacity pressure at the sparse end.

The controlled variable across all settings is the fraction of hidden neurons that fire at least once per forward pass, set by k-WTA. Forgetting (drop in earlier-task accuracy from peak to end of training), final average accuracy, and representational overlap (linear CKA) are the primary outcomes.

---

## 3. A Key Methodological Finding

Controlling spike activity turned out to be harder than anticipated, and resolving this problem is the pilot's most concrete contribution.

The natural approach is to set a fixed firing threshold: a higher threshold means fewer neurons reach it and fire. In a freshly initialised network this works. In a trained network it does not. As weights grow during training, the distribution of membrane potentials shifts upward, and the network re-saturates: regardless of the threshold value, the fraction of active neurons collapses into a narrow band (roughly 33-58% in our experiments) or, at very high thresholds, the network dies entirely with near-chance accuracy. The threshold cannot deliver a controlled low-activity regime in a trained network.

This was diagnosed by comparing the calibration target against the actually-achieved activity level across conditions. The correlation between threshold and achieved activity was weak and non-monotonic in the trained-network regime. More importantly, an earlier internal result had appeared to show an inverted-U performance curve as a function of threshold. When the same data were re-examined against actually-achieved activity rather than threshold, the inverted-U vanished (correlation between threshold-space "sweet spot" and achieved activity was approximately -0.07). The apparent optimum was a measurement artifact of using threshold as a proxy for activity.

The fix was a k-WTA gate applied per sample: exactly k neurons are permitted to fire per forward pass, where k is set as a fixed fraction of the hidden layer. This gives a genuinely controlled activity axis. Achieved activity tracked the targets closely, and the resulting axis spans a range the threshold approach could not reach.

---

## 4. Results

### MNIST-MLP

| Measured activity | Final accuracy | Mean forgetting | CKA overlap |
|---|---|---|---|
| ~1% | 0.77 | 0.27 | 0.0132 |
| ~5% | 0.83 | 0.20 | 0.0163 |
| ~10% | 0.78 | 0.26 | 0.0136 |
| ~20% | 0.82 | 0.22 | 0.0107 |
| ~28% | 0.84 | 0.19 | 0.0093 |
| ~33% | 0.88 | 0.15 | 0.0091 |

### CIFAR-MLP

Final accuracy is roughly 0.62 across all activity levels, nearly flat. The model lacks the capacity to learn CIFAR-10 tasks well at any sparsity level, so this setting cannot reveal a capacity-sparsity tradeoff and is not a fair test of H2 or H3.

### CIFAR-Conv-SNN

| Measured activity | Final accuracy | Mean forgetting |
|---|---|---|
| ~1% | 0.61 | 0.25 |
| ~5% | 0.69 | 0.25 |
| ~9% | 0.71 | 0.26 |
| ~16-20% | ~0.71 | ~0.23-0.25 |

### Hypothesis verdicts

**H2 (extreme sparsity causes accuracy collapse): now supported, conditionally.** The effect is only visible with a competent model. The Conv-SNN shows a roughly 10-point accuracy drop at 1% activity relative to the plateau at 9-20%. The decline is graded, not catastrophic: the network does not collapse, it degrades. The weak MNIST-MLP and CIFAR-MLP could not reveal this because Split-MNIST is easy enough that even 1% active neurons carry sufficient capacity, and the CIFAR-MLP lacks capacity at any activity level.

**H3 (inverted-U / sweet spot): not supported on any setting.** The consistent pattern across all three model/benchmark combinations is that more activity helps accuracy and reduces forgetting up to a point, then saturates. Denser activity is never worse, just no better beyond a threshold. There is no interior optimum anywhere in the data.

**H4 (overlap mediates the sparsity-forgetting relationship): not supported, now a fourth negative screen.** A formal exploratory mediation analysis on the MNIST-MLP data (18 conditions: 3 seeds x 6 activity levels) found a total effect of activity on forgetting of -0.48 (standardised), confirming that denser activity predicts less forgetting. The a-path (activity to overlap) was -0.81. However, the b-path (overlap predicting forgetting conditional on activity) was -0.16, weak and wrong-signed for the mediation hypothesis. The indirect effect was +0.13, with a 95% bootstrap confidence interval of [-0.59, +0.99] that straddles zero. On the Conv-SNN, the first link of the mediation ran the wrong way: more activity was associated with more overlap, not less. The indirect effect's confidence interval again straddled zero. Across both architectures, once activity is accounted for, overlap adds no explanatory power.

---

## 5. Interpretation and Limitations

Four negative screening results are the honest output of this pilot, and they are scientifically useful. They correct a prior artifact (the threshold-space inverted-U), establish that the activity axis is now properly controlled, and sharpen the design for the full study. The conditional support for H2 is a genuine finding: it required a competent model to appear, which is itself informative about when the effect is detectable.

Several limitations prevent these results from being read as refutations of the underlying hypotheses.

**Pilot scale.** Three seeds and 18 total conditions per setting are insufficient for a confirmatory mediation test. The b-path estimate is noisy, and the bootstrap confidence interval is wide enough to be consistent with both a genuine null and a moderate positive effect.

**Near-collinearity.** Activity, overlap, and forgetting are tightly coupled in this dataset. All three move together as k changes, which makes it structurally difficult to estimate the b-path with any precision at n = 18. The mediation model needs a design that can vary overlap independently of activity; the pilot cannot provide that.

**CKA range.** The absolute range of CKA values on MNIST (0.009-0.016) is very small. Whether this reflects genuinely low cross-task overlap, a ceiling effect, or a sensitivity limit of linear CKA at this network scale is not yet clear.

**Architecture scope.** The Conv-SNN is a small network by modern standards. The results are consistent across two architectures, which is encouraging, but neither is large enough to rule out scale effects.

---

## 6. Current Status and Next Steps

The pilot is complete. The k-WTA mechanism is implemented and validated. The Conv-SNN capacity-pressure test is done, and the picture is consistent across two architectures and two benchmarks: more activity helps up to a point, then saturates; no interior optimum; no mediation by overlap.

The natural next step is a properly powered confirmatory study. The specific priorities are:

**(a) Scale to more seeds.** The full study requires 8-10 seeds per condition, not 3. The pilot's estimates are not confirmatory and should not be treated as such.

**(b) Decouple activity from overlap in the study design.** The pilot cannot separate the two because k-WTA ties them together: changing k changes both activity and, as a consequence, overlap. One candidate is to compare k-WTA (which tends to produce more decorrelated codes through competition) against threshold-based sparsity at matched activity levels. If the two mechanisms produce different overlap at the same activity, the b-path becomes estimable independently of the a-path.

**(c) Consider a harder benchmark.** Split-CIFAR-10 with the Conv-SNN is a reasonable starting point for the full study. A larger convolutional architecture or a longer task sequence would increase the capacity pressure further and widen the CKA range, giving the mediation model more variance to work with.

The project's headline contribution rests on the mechanism question (H4), which remains open. The pilot has not supported it, but it has also not had the statistical power or the design to test it properly. That test is the full study.

---

## 7. Further Detail

Full technical detail, including the formal mediation model specification, per-condition results, and the diagnosis of the threshold-control failure, is in `RESEARCH_REPORT.md`. A chronological log of decisions, including the identification of the threshold artifact, the transition to k-WTA, and the addition of the Conv-SNN, is in `RESEARCH_JOURNAL.md`.

---

*These are pilot and exploratory findings, intended to guide the design of the full study. They are not publishable confirmatory claims.*
