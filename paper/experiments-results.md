# Experiments and Results Outline

## 1. Experimental Setup

### 1.1 Datasets
We evaluate our models on standard continual learning benchmarks to ensure comparability with existing literature. We primarily use:
- **Split-MNIST**: A common benchmark where the original MNIST dataset is split into 5 binary classification tasks.
- **Permuted-MNIST**: A more challenging benchmark where each task is a different random permutation of the MNIST pixels.
- **CIFAR-10/100 (Task-Incremental)**: For evaluating performance on more complex image data, we split CIFAR-10 or CIFAR-100 into a sequence of tasks.

### 1.2 Training Details
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Batch Size**: 128 for MNIST, 64 for CIFAR.
- **Epochs**: 10 epochs per task for MNIST, 20 for CIFAR.
- **Architecture**: For SNNs, we use Leaky Integrate-and-Fire (LIF) neurons with a membrane time constant of $\tau_m = 20$ms. For ANNs, we use an MLP with an equivalent number of parameters.

## 2. Sparsity Manipulation Results

### 2.1 Sparsity Levels
We systematically vary sparsity across a wide range of levels: 1%, 5%, 10%, 20%, 40%, 60%, 80%, and 95% activity (where 100% is the dense baseline).

### 2.2 Sparsity vs. Forgetting
Our primary experiment measures the forgetting score as a function of sparsity level. We expect a non-monotonic relationship, with moderate sparsity showing the best performance.

### 2.3 Sparsity vs. Accuracy
We measure the average task accuracy across all tasks for each sparsity level. This will reveal if there is an accuracy-sparsity trade-off.

## 3. Comparison with ANNs

### 3.1 Baseline Comparison
We compare the performance of our SNNs with an MLP baseline of equivalent size. For the ANN, we also induce sparsity via dropout or L1 regularization to create a fair comparison.

### 3.2 Energy Efficiency
We compare our energy proxy (spike count $\times$ synaptic operations) between the SNN and ANN baselines. We expect the SNN to be significantly more energy-efficient, especially at higher sparsity levels.

## 4. Continual Learning Methods Comparison

We evaluate the following continual learning methods under different sparsity levels:
- Naive Sequential Learning
- Replay Buffer (buffer size = 200)
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)

### 4.1 Method Interaction with Sparsity
We investigate if the optimal sparsity level is the same across different continual learning methods or if sparsity and CL methods have synergistic or antagonistic effects.

## 5. Mechanistic Analysis

### 5.1 Synaptic Overlap
To understand *why* sparsity helps, we measure the overlap in weight updates between different tasks. We hypothesize that sparser networks will have less overlap, reducing interference.

### 5.2 Representation Drift
We measure the drift in the network's internal representations (activations) over time as new tasks are learned. We expect sparse networks to have more stable representations.

## 6. Results Summary

### 6.1 Main Finding
We present a summary table and figure showing the relationship between sparsity, accuracy, and forgetting.

| Sparsity Level | Accuracy | Forgetting Score | Energy Proxy |
|---|---|---|---|
| 1% | ... | ... | ... |
| ... | ... | ... | ... |
| 95% | ... | ... | ... |

### 6.2 Optimal Sparsity
We identify the optimal sparsity region where the trade-off between accuracy and forgetting is most favorable.

### 6.3 Visualization
- **Accuracy vs. Sparsity Curve**: A plot showing the average task accuracy as a function of sparsity.
- **Forgetting vs. Sparsity Curve**: A plot showing the forgetting score as a function of sparsity.
- **Energy vs. Sparsity Curve**: A plot showing the energy proxy as a function of sparsity.
