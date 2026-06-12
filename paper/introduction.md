# Introduction

## Part 1: Task and Motivation

Continual learning—the ability to acquire new knowledge while preserving previously learned skills—is a cornerstone of intelligent systems. Unlike traditional machine learning, where models are trained on static datasets, continual learning requires adapting to non-stationary data distributions without succumbing to catastrophic forgetting [1, 2]. This phenomenon, first systematically characterized by McCloskey and Cohen [3], describes the rapid degradation of previously acquired knowledge when neural networks are trained on new tasks. For instance, a network trained to classify Task A may achieve near-perfect accuracy, but subsequent training on Task B results in complete loss of Task A performance, with further task introductions compounding this degradation [3, 4].

Biological brains, in stark contrast, exhibit remarkable continual learning capabilities. Humans learn continuously throughout their lifetimes, integrating new experiences without erasing prior knowledge [5, 6]. A key biological mechanism underlying this robustness is sparse neural activity: only a small fraction of neurons fire in response to any given stimulus, minimizing interference between task representations [7, 8]. This sparsity is not merely an emergent property but appears to be functionally important for memory consolidation and retrieval [9, 10].

## Part 2: SNNs as a Promising Substrate

Spiking neural networks (SNNs) have emerged as a compelling computational paradigm that more closely mirrors biological neural dynamics. Unlike ANNs, which transmit continuous-valued activations, SNNs communicate via discrete spikes, resulting in inherently sparse and event-driven processing [11, 12]. This biological plausibility extends beyond mere emulation: SNNs offer several functional advantages, including temporal information processing, asynchronous computation, and significantly lower energy consumption [13, 14].

The natural sparsity of SNNs raises an intriguing hypothesis: could the very mechanism that makes SNNs energy-efficient also confer resilience against catastrophic forgetting? If sparse activity patterns reduce interference between task representations, SNNs might inherently mitigate forgetting without requiring explicit regularization or memory replay mechanisms. This hypothesis is biologically motivated and technically plausible, yet remains empirically unverified.

## Part 3: Research Gap and Challenge

Despite growing interest in SNNs for continual learning, the relationship between spike sparsity and forgetting remains poorly understood. Existing work on SNN continual learning has focused primarily on architectural innovations [15, 16], synaptic plasticity rules [17, 18], or hybrid approaches combining SNNs with traditional continual learning techniques [19, 20]. While some studies have noted incidental sparsity effects [21, 22], no prior work has systematically isolated and manipulated sparsity as an independent variable to study its causal impact on forgetting.

This gap is significant for several reasons. First, if sparsity indeed reduces forgetting, it provides a principled, biologically grounded design criterion for neuromorphic continual learning systems. Second, understanding this relationship could inform the development of adaptive sparsity mechanisms that optimize the trade-off between learning new tasks and preserving old ones. Third, from a practical standpoint, sparser networks are more energy-efficient, making sparsity a desirable property even independent of its potential memory benefits [23].

## Part 4: Contributions

In this work, we systematically investigate the relationship between spike sparsity and catastrophic forgetting in SNNs. Our contributions are threefold:

1. **Controlled Sparsity Manipulation**: We implement and compare three distinct sparsity control mechanisms—spike threshold adjustment, winner-take-all inhibition, and activity regularization—enabling precise, independent manipulation of spiking activity across a wide range of sparsity levels.

2. **Empirical Characterization**: We conduct extensive experiments on standard continual learning benchmarks, measuring not only accuracy and forgetting but also spike rates and energy proxies. Our results reveal a non-monotonic relationship between sparsity and forgetting, with an identifiable optimal region.

3. **Mechanistic Insight**: Through analysis of synaptic overlap and representation drift, we provide evidence that sparsity reduces forgetting by minimizing interference between task-specific weight updates, offering a principled explanation for our empirical observations.

Our work demonstrates that sparsity is not merely a side effect of SNN dynamics but a tunable property that can be leveraged to improve continual learning performance. This insight bridges biological neuroscience and machine learning, suggesting that the very mechanisms that make brains efficient may also make them resilient.
