# Investigating the Relationship Between Spike Sparsity and Catastrophic Forgetting in Continual Learning Spiking Neural Networks 

## Continual Learning in Spiking Neural Networks

# Motivation

Artificial neural networks suffer from catastrophic forgetting.

In sequential training:

1. Training on Task A results in success.  
2. Subsequent training on Task B results in the complete loss of knowledge of Task A.  
3. Introducing Task C further erodes proficiency in both Task A and Task B.

Human brains:

* learn continuously  
* preserve previous knowledge  
* exhibit sparse neural activity

Neuromorphic systems naturally possess:

* sparse communication  
* event-driven processing  
* temporal dynamics

**The research gap:** *The relationship between spike sparsity and continual learning performance remains poorly understood.*

# Research Question 

**Primary question**: *Does increasing spike sparsity reduce catastrophic forgetting in spiking neural networks?*

**Secondary Questions:**

1. How does spike sparsity affect the retention of old training tasks?   
2. What sparsity level gives the best tradeoff between the following:  
* accuracy  
* forgetting  
* energy consumption  
3. Do sparse SNNs outperform equally sized ANNs in continual learning?

# Hypothesis 

1. Moderately sparse SNNs will exhibit less forgetting. Reason: Less interference between task representations.  
2. Extremely sparse SNNs will underperform. Reason: Insufficient representational capacity.  
3. There exists an optimal sparsity region. Expected: Too Dense → Optimal → Too Sparse

# Novelty

Most existing papers compare  
ANN vs SNN  
Replay vs EWC 

Our idea to investigate  
Spike Sparsity → Representation Drift → Catastrophic Forgetting 

# Models

ANN Baseline → MLP  
SNN Baseline → Leaky Integrate-and-Fire (LIF) Network   
Learn

1. snntorch  
2. Leaky Integrate-and-Fire neurons  
3. SNN training  
4. Continuous learning basics

# Continual Learning Methods

1. **Naive Sequential Learning**: Purpose: Reference baseline, measure maximum forgetting.  
2. **Replay Buffer**: Store examples from old tasks. Very strong baseline.  
3. **Elastic Weight Consolidation**: Protect important parameters. Classic continual-learning algorithm.  
4. **Synaptic Intelligence**: Biological-inspired parameter protection.

# Sparsity Manipulation 

Control spike activity through:

1. **Spike Threshold**: Higher threshold → Fewer spikes  
2. **Winner-Take-All**: Only a subset of neurons is active.  
3. **Activity Regularization**:

   Penalty: Loss \= TaskLoss \+ λ Activity

   Generate 10%, 20%, 40%, 60%, and 80% activity levels.

# Evaluation 

1. Average task Accuracy   
2. Forgetting score (best accuracy \- current accuracy)  
3. Spike rate (average spikes per neuron)  
4. Sparsity Index (percentage inactive neurons)  
5. Energy Proxy (spike count × synaptic operations)

# Reference

[https://arxiv.org/abs/2507.18139](https://arxiv.org/abs/2507.18139)  
[https://arxiv.org/abs/2602.12236](https://arxiv.org/abs/2602.12236)

~~Idea A~~  
~~Study Sparsity vs Forgetting~~  
~~Question: Does a sparser network forget less?~~

~~Idea B  
Adaptive Spike Thresholds  
Instead of a fixed threshold:  
Neurons adapt during learning.  
Question: Does adaptive excitability improve continual learning?~~

~~Idea C~~  
~~Replay Using Spikes Instead of Images~~  
~~Store:~~

* ~~spike trains~~

~~rather than~~

* ~~raw images~~

~~RQ1 Do SNNs forget less than ANNs? (Catastrophic Forgetting)~~

~~RQ2 Which continual learning method works best in SNNs?~~  
~~Examples:~~

* ~~Replay~~  
* ~~Elastic Weight Consolidation~~   
* ~~Synaptic Intelligence~~  
* ~~Spike-Timing-Dependent Plasticity STDP variants~~

~~RQ3 What is the relationship between:~~

* ~~spike sparsity~~  
* ~~energy efficiency~~  
* ~~Forgetting~~

