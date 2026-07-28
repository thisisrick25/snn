# Companion Learning Guide -- Part 1: Foundations of Deep Learning and Continual Learning

> **About this guide.** This is Part 1 of a multi-part companion learning guide written to accompany a research proposal on spike sparsity and catastrophic forgetting in spiking neural networks (SNNs). The guide is designed for graduate students who are new to the subject but aim for research-level understanding. Each part builds on the previous one. Part 1 covers the classical foundations: supervised deep learning, image classification benchmarks, the continual learning problem setting, and the catastrophic forgetting phenomenon. Part 2 will introduce spiking neural networks and the leaky integrate-and-fire neuron model. Part 3 will cover methods for mitigating catastrophic forgetting. Throughout the guide, the running experimental example is Split-MNIST -- a standard continual learning benchmark derived from the MNIST handwritten digit dataset -- trained with a naive sequential strategy (Adam optimizer, learning rate 0.001, batch size 128, 10 epochs per task, cross-entropy loss). This concrete anchor will help you connect abstract theory to the specific project you will read about in the proposal.

---

# Chapter 1: Neural Networks and Supervised Learning

## Overview

A neural network is a parameterized function that maps an input vector to an output vector. In supervised learning, we have a dataset of input-output pairs and we want to find parameter values that make the function approximate the true mapping as closely as possible. The multilayer perceptron (MLP) is the simplest fully-connected neural network architecture and serves as the baseline model throughout this guide. Understanding how MLPs are trained -- through backpropagation and gradient descent -- is the prerequisite for everything that follows, including why catastrophic forgetting happens and how spiking neural networks differ from their classical counterparts.

---

## Fundamental Theory

Supervised learning rests on three pillars: a hypothesis class (the set of functions the model can represent), a loss function (a scalar measure of how wrong the current parameters are), and an optimization algorithm (a procedure for reducing the loss).

**The statistical learning framework.** We assume data is drawn i.i.d. from an unknown joint distribution $p(x, y)$ over inputs $x \in \mathbb{R}^d$ and labels $y$. The goal is to find a function $f_\theta$ parameterized by $\theta$ that minimizes the expected risk:

$$R(\theta) = \mathbb{E}_{(x,y) \sim p} \left[ \ell(f_\theta(x), y) \right]$$

where $\ell$ is a loss function. Because $p$ is unknown, we minimize the empirical risk over a training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$:

$$\hat{R}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

The gap between $R(\theta)$ and $\hat{R}(\theta)$ is the generalization gap, and controlling it is the central concern of statistical learning theory.

---

## Technical Explanation

### The Multilayer Perceptron (MLP)

An MLP with $L$ layers computes a sequence of transformations. Let $h^{(0)} = x$ be the input. For each layer $l = 1, \ldots, L$:

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$
$$h^{(l)} = \sigma(z^{(l)})$$

where $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix, $b^{(l)} \in \mathbb{R}^{n_l}$ is the bias vector, and $\sigma$ is a nonlinear activation function applied elementwise. The final layer output $h^{(L)}$ is the network's prediction (possibly passed through a softmax for classification).

The total parameter count is $\sum_{l=1}^L (n_l \cdot n_{l-1} + n_l)$.

### The ReLU Activation Function

The Rectified Linear Unit (ReLU) is defined as:

$$\text{ReLU}(z) = \max(0, z)$$

Its derivative is:

$$\frac{d}{dz} \text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

(undefined at $z = 0$, but set to 0 or 1 by convention in practice).

ReLU is piecewise linear, which makes gradient computation cheap and avoids the vanishing gradient problem that plagued earlier sigmoid and tanh activations in deep networks.

### Cross-Entropy Loss

For $C$-class classification, the network outputs a vector of logits $z \in \mathbb{R}^C$. The softmax converts logits to a probability distribution:

$$p_c = \frac{e^{z_c}}{\sum_{c'=1}^C e^{z_{c'}}}$$

The cross-entropy loss for a single example with true class $y$ is:

$$\ell(z, y) = -\log p_y = -z_y + \log \sum_{c=1}^C e^{z_c}$$

For a batch of $N$ examples:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log p_{y_i}$$

Cross-entropy is the negative log-likelihood under the categorical distribution. Minimizing it is equivalent to maximizing the likelihood of the correct labels.

For binary classification (two classes), the loss simplifies to binary cross-entropy:

$$\ell(z, y) = -y \log \sigma(z) - (1-y) \log(1 - \sigma(z))$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid function. In the Split-MNIST running example, each task is a binary classification problem (e.g., digit 0 vs digit 1), so binary cross-entropy is the natural loss.

### Backpropagation

Backpropagation is an efficient algorithm for computing the gradient of the loss with respect to all parameters. It applies the chain rule of calculus layer by layer, propagating error signals from the output back to the input.

Define the loss gradient with respect to the pre-activation of layer $l$ as $\delta^{(l)} = \partial \mathcal{L} / \partial z^{(l)}$. The backprop recurrence is:

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}}$$

$$\delta^{(l)} = \left( W^{(l+1)\top} \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

where $\odot$ denotes elementwise multiplication and $\sigma'$ is the derivative of the activation. The parameter gradients are:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} h^{(l-1)\top}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

Backprop has time complexity $O(P)$ where $P$ is the total number of parameters -- the same order as a single forward pass.

### Gradient Descent and Adam

**Stochastic gradient descent (SGD)** updates parameters using the gradient computed on a mini-batch $\mathcal{B}$:

$$\theta \leftarrow \theta - \eta \nabla_\theta \hat{\mathcal{L}}_\mathcal{B}(\theta)$$

where $\eta$ is the learning rate. Mini-batch SGD introduces noise that can help escape local minima and is computationally efficient.

**Adam** (Adaptive Moment Estimation) maintains running estimates of the first moment (mean) $m_t$ and second moment (uncentered variance) $v_t$ of the gradient:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. In the Split-MNIST project, Adam is used with $\eta = 0.001$.

Adam adapts the learning rate per parameter, making it robust to sparse gradients and reducing the need for manual learning rate tuning.

---

## Core Concepts

**Hypothesis class.** The set of all functions representable by a given architecture and parameter space.

**Loss function.** A scalar-valued function $\ell(f_\theta(x), y)$ measuring prediction error for a single example.

**Empirical risk.** The average loss over the training set.

**Gradient.** The vector of partial derivatives of the loss with respect to all parameters; points in the direction of steepest ascent.

**Mini-batch.** A small random subset of the training data used to estimate the gradient at each step.

**Epoch.** One complete pass through the entire training dataset.

**Activation function.** A nonlinear function applied elementwise after each linear transformation; without it, the network collapses to a linear model regardless of depth.

**Softmax.** A function that converts a vector of real-valued logits into a probability distribution over classes.

**Logit.** The raw, unnormalized output of the final linear layer before softmax.

---

## What / Why / When / How

**What** is an MLP? A directed acyclic graph of linear transformations interleaved with nonlinearities, capable of approximating any continuous function on a compact domain (universal approximation theorem).

**Why** use cross-entropy loss? It is the natural probabilistic loss for classification: minimizing it is equivalent to minimizing the KL divergence between the predicted distribution and the true label distribution. It also has well-behaved gradients near the correct answer.

**When** does gradient descent converge? For convex losses, gradient descent with a sufficiently small learning rate converges to the global minimum. For non-convex losses (as in deep networks), convergence to a local minimum or saddle point is typical, but in practice deep networks often find good solutions.

**How** does backprop work efficiently? By caching intermediate activations during the forward pass and reusing them during the backward pass, backprop avoids redundant computation. The total cost is $O(P)$ per example, the same as a forward pass.

---

## Advantages and Disadvantages

**Advantages of MLPs:**
- Universal approximation: a single hidden layer with enough neurons can approximate any continuous function.
- Differentiable end-to-end: backprop applies cleanly.
- Simple to implement and reason about.
- Flexible: can be applied to any fixed-size input vector.

**Disadvantages of MLPs:**
- No inductive bias for spatial or sequential structure (unlike CNNs or RNNs).
- Parameter count scales quadratically with layer width.
- Prone to overfitting on small datasets without regularization.
- Shared weights across all tasks: this is the root cause of catastrophic forgetting, as we will see in Chapter 4.

**Advantages of Adam over SGD:**
- Adaptive per-parameter learning rates reduce sensitivity to hyperparameter choice.
- Handles sparse gradients well.
- Faster convergence in practice on many tasks.

**Disadvantages of Adam:**
- Can generalize slightly worse than well-tuned SGD in some settings (Wilson et al., 2017).
- Maintains additional state (moment estimates), increasing memory by a factor of ~3x over SGD.

---

## Historical Context

The perceptron was introduced by Rosenblatt (1958) as a single-layer linear classifier. Minsky and Papert (1969) showed its limitations, triggering the first "AI winter." The backpropagation algorithm was popularized by Rumelhart, Hinton, and Williams (1986), enabling training of multi-layer networks. LeCun et al. (1989) applied backprop to convolutional networks for digit recognition. The modern deep learning era began with Krizhevsky, Sutskever, and Hinton (2012), who demonstrated that deep ReLU networks trained with SGD on GPUs dramatically outperformed prior methods on ImageNet. Adam was introduced by Kingma and Ba (2015) and quickly became the default optimizer for most deep learning research.

---

## Comparison

| Optimizer | Adaptive LR | Memory overhead | Convergence speed | Generalization |
|-----------|-------------|-----------------|-------------------|----------------|
| SGD | No | Low | Slow (needs tuning) | Often best with tuning |
| SGD + Momentum | No | Low | Faster | Good |
| RMSProp | Yes | Medium | Fast | Good |
| Adam | Yes | High (~3x) | Fast | Slightly worse than SGD in some cases |

| Activation | Vanishing gradient | Sparse activation | Computation |
|------------|-------------------|-------------------|-------------|
| Sigmoid | Severe | No | Moderate |
| Tanh | Moderate | No | Moderate |
| ReLU | None (positive region) | Yes (dead neurons) | Very cheap |
| Leaky ReLU | None | Less | Very cheap |

---

## Visual Intuition

```
Input layer     Hidden layer 1    Hidden layer 2    Output layer
  (784)            (256)              (128)             (2)

  x_1 ---\
  x_2 ---+---> [h1_1] ---\
  x_3 ---+---> [h1_2] ---+---> [h2_1] ---\
  ...    |    [h1_3] ---+---> [h2_2] ---+--> [out_1]
  x_784 -/    ...       /    ...       /    [out_2]
              [h1_256] -/    [h2_128] -/

Each arrow represents a learned weight.
Each node applies: output = ReLU(W * input + b)
Final layer: output = softmax(W * h2 + b)
```

Think of each neuron as a "feature detector." Early layers detect simple patterns (edges, pixel intensities). Later layers combine these into more abstract representations. The network learns which combinations of features predict the correct label.

**Gradient descent intuition.** Imagine the loss surface as a hilly landscape. The gradient tells you which direction is uphill. Gradient descent takes a small step downhill at each iteration. Adam adds momentum (so you keep moving in a consistent direction) and adapts step size (so you take smaller steps in steep directions and larger steps in flat ones).

---

## Mathematical Foundations

**Universal approximation theorem (Cybenko, 1989; Hornik, 1991).** For any continuous function $f: [0,1]^d \to \mathbb{R}$ and any $\epsilon > 0$, there exists a single-hidden-layer network with a sigmoidal activation and a finite number of neurons $N$ such that:

$$\sup_{x \in [0,1]^d} |f_\theta(x) - f(x)| < \epsilon$$

This guarantees expressiveness but says nothing about how to find the right parameters or how many neurons are needed.

**Bias-variance decomposition.** For squared loss, the expected test error decomposes as:

$$\mathbb{E}[(f_\theta(x) - y)^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

- Bias: error from wrong assumptions in the model class.
- Variance: sensitivity to fluctuations in the training set.
- Noise: irreducible error from the data distribution.

Larger models reduce bias but increase variance. Regularization (dropout, weight decay) reduces variance at the cost of some bias.

**Softmax gradient.** For the cross-entropy loss with softmax, the gradient with respect to the logit $z_c$ is:

$$\frac{\partial \mathcal{L}}{\partial z_c} = p_c - \mathbf{1}[c = y]$$

This is the difference between the predicted probability and the one-hot target -- a clean, interpretable signal.

**Chain rule (formal statement).** If $\mathcal{L} = g(f(x))$, then:

$$\frac{d\mathcal{L}}{dx} = \frac{dg}{df} \cdot \frac{df}{dx}$$

Backprop applies this recursively across all layers, accumulating products of Jacobians from output to input.

---

## Implementation Notes

- **Weight initialization matters.** Random initialization with variance $2/n_{in}$ (He initialization) is standard for ReLU networks. Zero initialization causes all neurons to learn the same features (symmetry breaking failure).
- **Batch normalization** (Ioffe and Szegedy, 2015) normalizes layer inputs to zero mean and unit variance, stabilizing training and allowing higher learning rates.
- **Dropout** (Srivastava et al., 2014) randomly zeroes activations during training, acting as an ensemble regularizer.
- **Gradient clipping** prevents exploding gradients in deep or recurrent networks.
- **Numerical stability.** Compute softmax as $\text{softmax}(z - \max(z))$ to avoid overflow.
- **Computational complexity.** Forward pass: $O(\sum_l n_l \cdot n_{l-1})$. Backward pass: same order. Memory: $O(P + N \cdot \max_l n_l)$ for storing activations during backprop.

---

## Examples

**Simple example.** Consider a 2-layer MLP for XOR: input $x \in \{0,1\}^2$, output $y \in \{0,1\}$. A single hidden layer with 2 ReLU neurons and appropriate weights can solve XOR, which a linear model cannot. This illustrates why nonlinearity is essential.

**Running example: Split-MNIST Task 1 (0 vs 1).** The network receives a flattened 784-dimensional input (a 28x28 grayscale image of a handwritten digit). The MLP has two hidden layers (e.g., 256 and 128 neurons, ReLU activations) and a 2-neuron output head for binary classification. Adam with $\eta = 0.001$ updates parameters over 10 epochs on the training split of digits 0 and 1. Cross-entropy loss drives the output probabilities toward the correct class. After training, the network achieves near-perfect accuracy on Task 1.

---

## Current Research

- **Neural scaling laws** (Kaplan et al., 2020) show that loss decreases predictably as a power law with model size, dataset size, and compute.
- **Implicit regularization** of gradient descent: even without explicit regularization, SGD tends to find flat minima that generalize well (Keskar et al., 2017).
- **Loss landscape geometry**: Garipov et al. (2018) showed that good minima are connected by low-loss paths, challenging the view that local minima are isolated.
- **Mechanistic interpretability**: recent work attempts to reverse-engineer what individual neurons and circuits compute (Elhage et al., 2021).

---

## References

- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press. (Chapters 6, 7, 8.)
- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.
- Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.
- Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS 2012*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15, 1929-1958.
- Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*.

---

# Chapter 2: Image Classification and the MNIST Dataset Family

## Overview

Image classification is the task of assigning a semantic label to an image. It is the canonical supervised learning problem for computer vision and has driven most of the major advances in deep learning over the past two decades. The MNIST dataset of handwritten digits is the most widely used benchmark in the field: simple enough to train quickly, rich enough to reveal meaningful differences between methods, and standardized enough to allow fair comparison across decades of research. Understanding MNIST -- its structure, its statistics, and its limitations -- is essential for interpreting results in continual learning research, where it appears in the form of Split-MNIST, Permuted-MNIST, and related variants.

---

## Fundamental Theory

Image classification is a function approximation problem where the input is a tensor of pixel intensities and the output is a discrete label. The key theoretical challenge is that the mapping from pixels to labels is highly nonlinear, high-dimensional, and invariant to many transformations (translation, rotation, illumination change). Deep networks learn hierarchical representations that factor out these invariances.

**Formal definition.** Given an image $x \in \mathbb{R}^{H \times W \times C}$ (height $H$, width $W$, channels $C$) and a label set $\mathcal{Y} = \{1, \ldots, K\}$, the goal is to learn $f_\theta: \mathbb{R}^{H \times W \times C} \to \mathcal{Y}$ that minimizes classification error on the data distribution.

**The curse of dimensionality.** A 28x28 grayscale image has 784 dimensions. A 224x224 RGB image has 150,528 dimensions. Naive nearest-neighbor classifiers fail in high dimensions because all points become equidistant. Deep networks overcome this by learning low-dimensional manifold structure.

---

## Technical Explanation

### The MNIST Dataset

MNIST (Modified National Institute of Standards and Technology) contains 70,000 grayscale images of handwritten digits (0-9):
- 60,000 training images
- 10,000 test images
- Image size: 28x28 pixels, single channel (grayscale)
- Pixel values: integers in $[0, 255]$, typically normalized to $[0, 1]$ or $[-1, 1]$
- 10 classes (one per digit)
- Roughly balanced: ~6,000 training examples per class

Each image is stored as a 28x28 matrix. For an MLP, this matrix is **flattened** into a 784-dimensional vector:

$$x \in \mathbb{R}^{28 \times 28} \xrightarrow{\text{flatten}} x \in \mathbb{R}^{784}$$

This flattening discards spatial structure (which pixel is adjacent to which), which is why MLPs are less efficient than CNNs for images. However, for MNIST, the digit shapes are distinctive enough that spatial structure is not strictly necessary for high accuracy.

### Preprocessing

Standard preprocessing for MNIST:
1. **Normalization:** divide pixel values by 255 to map to $[0, 1]$.
2. **Standardization (optional):** subtract the dataset mean (0.1307) and divide by the standard deviation (0.3081).
3. **Flattening:** reshape from $(28, 28)$ to $(784,)$ for MLP input.

### The MNIST Family

Several variants of MNIST are used in continual learning research:

**Split-MNIST.** The 10-digit dataset is split into 5 binary classification tasks:
- Task 1: digit 0 vs digit 1
- Task 2: digit 2 vs digit 3
- Task 3: digit 4 vs digit 5
- Task 4: digit 6 vs digit 7
- Task 5: digit 8 vs digit 9

Each task uses only the subset of training and test images belonging to its two classes. This is the running example throughout this guide.

**Permuted-MNIST.** Each task applies a fixed random permutation to the pixel indices of all images. The label set remains the same (0-9) across tasks, but the input distribution changes. This tests domain-incremental learning.

**Rotated-MNIST.** Each task rotates all images by a fixed angle. Similar to Permuted-MNIST in spirit.

**Fashion-MNIST.** A drop-in replacement for MNIST with 10 classes of clothing items (Xiao et al., 2017). Same size and format, but harder.

**EMNIST.** Extended MNIST including letters and digits (Cohen et al., 2017).

---

## Core Concepts

**Benchmark.** A standardized dataset and evaluation protocol that allows fair comparison between methods.

**Flattening.** Reshaping a multi-dimensional array into a 1D vector. For MNIST: $(28, 28) \to (784,)$.

**Normalization.** Scaling pixel values to a standard range, typically $[0, 1]$ or $[-1, 1]$.

**Train/test split.** The partition of data into a training set (used to fit parameters) and a test set (used to evaluate generalization). The test set must never be used during training.

**Generalization.** The ability of a trained model to perform well on unseen data from the same distribution. Measured by test accuracy.

**Overfitting.** When a model performs well on training data but poorly on test data, indicating it has memorized training examples rather than learned general patterns.

**Underfitting.** When a model performs poorly on both training and test data, indicating insufficient capacity or training.

---

## What / Why / When / How

**What** is MNIST? A dataset of 70,000 28x28 grayscale images of handwritten digits, split 60,000/10,000 for training and testing, with 10 balanced classes.

**Why** is MNIST a standard benchmark? It is large enough to train meaningful models, small enough to iterate quickly, well-curated (no label noise), and has been used since 1998, providing a long history of results for comparison. Its simplicity makes it ideal for studying algorithmic properties (like catastrophic forgetting) in isolation from the confounds of harder datasets.

**When** is MNIST appropriate? For proof-of-concept experiments, algorithm development, and continual learning benchmarks. It is not appropriate for evaluating state-of-the-art vision systems, where CIFAR-10, ImageNet, or domain-specific datasets are preferred.

**How** is MNIST used in Split-MNIST? The 60,000 training images are partitioned by label into 5 groups of ~12,000 images each (two digits per group). A model is trained sequentially on these groups, one task at a time, without revisiting earlier data. The test set is similarly partitioned, and accuracy on all 5 test partitions is measured after each task.

---

## Advantages and Disadvantages

**Advantages of MNIST:**
- Small and fast: a full training run takes seconds to minutes on a CPU.
- Well-understood: decades of results provide context for new findings.
- Clean: minimal label noise, consistent image quality.
- Balanced: roughly equal class frequencies.
- Freely available: no licensing restrictions.

**Disadvantages of MNIST:**
- Too easy for modern architectures: a simple MLP achieves >98% accuracy; a CNN achieves >99.7%.
- Low resolution and grayscale: does not test color or texture understanding.
- Limited diversity: all images are centered, roughly the same size, on a black background.
- Results on MNIST do not always transfer to harder datasets.

**Advantages of Split-MNIST for continual learning research:**
- Controlled: task boundaries are sharp and known.
- Reproducible: standard splits allow comparison across papers.
- Fast: enables rapid iteration on algorithm design.

**Disadvantages of Split-MNIST:**
- Artificial: real continual learning involves gradual distribution shifts, not sharp task boundaries.
- Small per-task dataset: ~12,000 images per task, which may not stress-test memory or capacity.
- Binary tasks: real problems have many more classes.

---

## Historical Context

MNIST was created by LeCun, Cortes, and Burges (1998) by combining samples from NIST's Special Database 1 and Special Database 3, re-centering and normalizing the images. It was introduced alongside LeNet-5, a convolutional neural network that achieved 0.7% test error. For over a decade, MNIST was the primary benchmark for handwritten character recognition. As deep learning matured, MNIST became "solved" and researchers moved to harder benchmarks (CIFAR-10, ImageNet). However, MNIST retained its role as a development and debugging tool, and its use in continual learning benchmarks (Split-MNIST, Permuted-MNIST) has kept it relevant into the 2020s.

---

## Comparison

| Dataset | Classes | Images | Size | Difficulty | Common use |
|---------|---------|--------|------|------------|------------|
| MNIST | 10 | 70,000 | 28x28x1 | Easy | Benchmarking, CL research |
| Fashion-MNIST | 10 | 70,000 | 28x28x1 | Medium | Drop-in MNIST replacement |
| CIFAR-10 | 10 | 60,000 | 32x32x3 | Medium | Vision benchmarking |
| CIFAR-100 | 100 | 60,000 | 32x32x3 | Hard | Fine-grained classification |
| ImageNet | 1,000 | 1.2M | 224x224x3 | Very hard | Large-scale vision |

| CL Benchmark | Base dataset | Tasks | Task type | Difficulty |
|--------------|-------------|-------|-----------|------------|
| Split-MNIST | MNIST | 5 | Binary classification | Low |
| Permuted-MNIST | MNIST | 10+ | 10-class classification | Medium |
| Split-CIFAR-10 | CIFAR-10 | 5 | Binary classification | Medium |
| Split-CIFAR-100 | CIFAR-100 | 20 | 5-class classification | High |

---

## Visual Intuition

```
MNIST image (digit "3"):

  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . # # # # # # # # . . . . . . . . . .
  . . . . . . . . # # # # # # # # # # # . . . . . . . . .
  . . . . . . . # # # # . . . . . # # # # . . . . . . . .
  . . . . . . . . . . . . . . . . . # # # . . . . . . . .
  . . . . . . . . . . . . . . . . # # # . . . . . . . . .
  . . . . . . . . . . . . . # # # # # . . . . . . . . . .
  . . . . . . . . . . . . . . . . # # # . . . . . . . . .
  . . . . . . . . . . . . . . . . . # # # . . . . . . . .
  . . . . . . . # # # . . . . . . . # # # . . . . . . . .
  . . . . . . . # # # # # # # # # # # # . . . . . . . . .
  . . . . . . . . . # # # # # # # # # . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Flattening: read pixels left-to-right, top-to-bottom
  [0, 0, 0, ..., 0, 128, 255, 255, 128, 0, ..., 0]
   ^                                              ^
   pixel (0,0)                            pixel (27,27)
   
Result: a vector of length 784
```

**Split-MNIST task structure:**

```
Full MNIST (10 classes: 0-9)
         |
         v
  Split into 5 tasks:

  Task 1: [0, 1]  --> binary: is this a 0 or a 1?
  Task 2: [2, 3]  --> binary: is this a 2 or a 3?
  Task 3: [4, 5]  --> binary: is this a 4 or a 5?
  Task 4: [6, 7]  --> binary: is this a 6 or a 7?
  Task 5: [8, 9]  --> binary: is this an 8 or a 9?

Training order: Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5
(no revisiting earlier tasks)
```

---

## Mathematical Foundations

**Pixel representation.** Each pixel $p_{ij}$ takes a value in $\{0, 1, \ldots, 255\}$. After normalization:

$$\tilde{p}_{ij} = \frac{p_{ij}}{255} \in [0, 1]$$

**Flattening operator.** For a 2D image $P \in \mathbb{R}^{H \times W}$, the flattening operator $\text{vec}: \mathbb{R}^{H \times W} \to \mathbb{R}^{HW}$ is defined by:

$$[\text{vec}(P)]_{i \cdot W + j} = P_{ij}, \quad i \in \{0, \ldots, H-1\}, \; j \in \{0, \ldots, W-1\}$$

For MNIST: $H = W = 28$, $HW = 784$.

**Class balance.** Let $N_c$ be the number of training examples in class $c$. For MNIST, $N_c \approx 6000$ for all $c \in \{0, \ldots, 9\}$. The class imbalance ratio $\max_c N_c / \min_c N_c \approx 1.1$, which is negligible.

**Accuracy.** The standard evaluation metric:

$$\text{Acc} = \frac{1}{|\mathcal{D}_\text{test}|} \sum_{(x,y) \in \mathcal{D}_\text{test}} \mathbf{1}[f_\theta(x) = y]$$

For Split-MNIST, accuracy is reported per task and as an average across all tasks.

---

## Implementation Notes

- **Data loading.** Most deep learning frameworks (PyTorch, TensorFlow) provide built-in MNIST loaders. In PyTorch: `torchvision.datasets.MNIST`.
- **Normalization.** Apply `transforms.Normalize((0.1307,), (0.3081,))` for MNIST-specific standardization.
- **Flattening.** Use `transforms.Lambda(lambda x: x.view(-1))` or `nn.Flatten()` in the model.
- **Split-MNIST construction.** Filter the dataset by label: `dataset.targets.isin([0, 1])` for Task 1. Remap labels to $\{0, 1\}$ within each task.
- **Reproducibility.** Fix random seeds for data shuffling and model initialization. Use `torch.manual_seed(42)` and `numpy.random.seed(42)`.
- **Batch size.** The project uses batch size 128. Larger batches give more stable gradient estimates but may generalize slightly worse.

---

## Examples

**Simple example.** Load MNIST in PyTorch, flatten, and check dimensions:

```python
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
x, y = train_data[0]
print(x.shape)  # torch.Size([784])
print(y)        # e.g., 5
```

**Running example: constructing Task 1 of Split-MNIST.**

```python
# Filter for digits 0 and 1
mask = (train_data.targets == 0) | (train_data.targets == 1)
task1_data = torch.utils.data.Subset(train_data, mask.nonzero().squeeze())
# Remap labels: 0 -> 0, 1 -> 1 (already correct for Task 1)
# For Task 2: remap 2 -> 0, 3 -> 1
```

---

## Current Research

- **Beyond MNIST.** The community has largely moved to harder benchmarks: Split-CIFAR-100, Tiny-ImageNet, and domain-specific datasets for continual learning evaluation.
- **Dataset difficulty calibration.** Researchers study what makes a benchmark informative: class overlap, intra-class variance, inter-task similarity (Pfuelb and Gepperth, 2019).
- **Synthetic benchmarks.** Procedurally generated datasets allow precise control over task similarity and forgetting difficulty.
- **Real-world continual learning.** Datasets from robotics, medical imaging, and autonomous driving are increasingly used to test methods under realistic distribution shifts.

---

## References

- LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
- Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
- Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. (2017). EMNIST: Extending MNIST to handwritten letters. *IJCNN 2017*.
- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 12 on applications.)
- van de Ven, G. M. and Tolias, A. S. (2019). Three scenarios for continual learning. *NeurIPS 2019 Workshop on Continual Learning*.

---

# Chapter 3: Continual Learning and Its Settings

## Overview

Continual learning (CL), also called lifelong learning or sequential learning, is the study of systems that learn from a non-stationary stream of tasks or data distributions over time. Unlike standard supervised learning, where a fixed dataset is available throughout training, a continual learner receives data from one task at a time and is generally not permitted to store or freely revisit data from previous tasks. The central challenge is to acquire new knowledge without destroying knowledge already encoded in the model's parameters. This chapter formalizes the continual learning problem, introduces the three canonical evaluation scenarios defined by van de Ven and Tolias (2019), and connects each scenario to the Split-MNIST project that serves as the running example throughout this guide.

---

## Fundamental Theory

The continual learning problem departs from the standard i.i.d. assumption of statistical learning theory in a fundamental way. In standard supervised learning, all training data is drawn from a single fixed distribution $p(x, y)$ and is available simultaneously. In continual learning, the data distribution changes over time, and the learner must adapt to each new distribution without access to data from previous distributions.

**The sequential task setting.** Let there be $T$ tasks, indexed $t = 1, \ldots, T$. Each task $t$ is associated with a dataset $\mathcal{D}_t = \{(x_i^{(t)}, y_i^{(t)})\}_{i=1}^{N_t}$ drawn i.i.d. from a task-specific distribution $p_t(x, y)$. The distributions $p_1, p_2, \ldots, p_T$ are generally different from one another. The learner processes tasks sequentially: it trains on $\mathcal{D}_1$, then on $\mathcal{D}_2$, and so on. At the time of training on task $t$, the data $\mathcal{D}_1, \ldots, \mathcal{D}_{t-1}$ from previous tasks is not freely available (this is the defining constraint of the CL setting).

**The ideal objective.** After training on all $T$ tasks, the learner should perform well on all of them simultaneously. The ideal loss is:

$$\mathcal{L}_{\text{CL}}(\theta) = \frac{1}{T} \sum_{t=1}^T \mathcal{L}_t(\theta)$$

where $\mathcal{L}_t(\theta)$ is the loss on task $t$. Naive sequential training minimizes only $\mathcal{L}_T(\theta)$ at the final step, ignoring all previous tasks. This is the root cause of catastrophic forgetting, which we examine in Chapter 4.

**Why data is not i.i.d.** In the sequential setting, the effective training distribution at step $t$ is $p_t(x, y)$, not the mixture $\frac{1}{T}\sum_{t'} p_{t'}(x, y)$. Gradient descent on $\mathcal{L}_t$ moves parameters in directions that reduce loss on task $t$, which may increase loss on tasks $1, \ldots, t-1$. The non-i.i.d. nature of the data stream is what makes continual learning fundamentally harder than standard learning.

---

## Technical Explanation

### The Three Continual Learning Scenarios

Van de Ven and Tolias (2019) identified three distinct evaluation scenarios for continual learning, differing in what information is available at test time and how the output space is structured. These scenarios are not merely implementation choices; they represent fundamentally different problem formulations with different levels of difficulty.

**Scenario 1: Task-Incremental Learning (Task-IL)**

In task-incremental learning, the task identity $t$ is provided to the model at both training and test time. The model knows which task it is currently being evaluated on. This allows the use of separate output heads: one per task, each with its own set of output neurons. At test time, the model routes the input to the correct head using the provided task label.

Architecture: the shared body (feature extractor) is common to all tasks, but each task has its own dedicated output layer. For Split-MNIST with 5 binary tasks, there are 5 separate 2-neuron output heads.

This is the easiest scenario because the task label eliminates ambiguity about which output head to use. The model only needs to learn good features for each task; it does not need to distinguish between tasks at inference time. Forgetting still occurs in the shared body, but the per-task heads prevent cross-task output interference.

**Scenario 2: Domain-Incremental Learning (Domain-IL)**

In domain-incremental learning, the task identity is NOT provided at test time, but the output space (label set) is the same across all tasks. The input distribution changes across tasks (different "domains"), but the model must always produce an answer from the same fixed set of classes. A single output head is used throughout.

Example: Permuted-MNIST, where each task applies a different pixel permutation but the label set is always digits 0-9. The model must classify correctly without knowing which permutation was applied.

This scenario is harder than task-incremental because the model must handle multiple input distributions with a single output head, but easier than class-incremental because the output space does not grow and there is no need to distinguish which task an input belongs to.

**Scenario 3: Class-Incremental Learning (Class-IL)**

In class-incremental learning, the task identity is NOT provided at test time, AND new classes are added with each task. The model must maintain a single classifier over all classes seen so far, and at test time it must both identify which task the input belongs to and classify it correctly within that task -- all without any task label.

For Split-MNIST in the class-incremental setting: after training on all 5 tasks, the model must classify any digit image into one of 10 classes (or at minimum, into the correct binary pair) without being told which task it belongs to. The output head grows from 2 neurons after Task 1 to 10 neurons after Task 5.

This is the hardest scenario for two reasons. First, the model must solve an implicit task-identification problem at inference time. Second, the growing output head means that the model must not only remember old features but also correctly assign old inputs to old output neurons in competition with new ones. Naive sequential training causes severe forgetting here because the model's output layer is retrained on new classes, overwriting the decision boundaries for old classes.

### Connection to the Split-MNIST Project

The research project accompanying this guide uses Split-MNIST in the **task-incremental** setting as its primary benchmark. The five tasks are:

- Task 1: digit 0 vs digit 1 (labels remapped: 0 -> 0, 1 -> 1)
- Task 2: digit 2 vs digit 3 (labels remapped: 2 -> 0, 3 -> 1)
- Task 3: digit 4 vs digit 5 (labels remapped: 4 -> 0, 5 -> 1)
- Task 4: digit 6 vs digit 7 (labels remapped: 6 -> 0, 7 -> 1)
- Task 5: digit 8 vs digit 9 (labels remapped: 8 -> 0, 9 -> 1)

Each task has its own 2-neuron output head. At test time, the task label is provided, so the correct head is selected. The shared body (hidden layers) is trained sequentially on all five tasks.

The refined proposal also introduces a **class-incremental variant**: a single growing output head with no task label at inference. This variant is expected to show substantially more forgetting, providing a harder test of whether spike sparsity provides any protection against catastrophic interference.

---

## Core Concepts

**Task.** A self-contained learning problem with its own data distribution $p_t(x, y)$ and, in task-incremental learning, its own output head.

**Task identity / task label.** An integer $t \in \{1, \ldots, T\}$ indicating which task a given input belongs to. Provided at test time in task-incremental learning; withheld in domain-incremental and class-incremental learning.

**Output head.** The final linear layer (or set of output neurons) used to produce predictions for a given task. In task-incremental learning, each task has its own head. In class-incremental learning, a single head grows as new classes are added.

**Shared body.** The feature-extraction layers (all layers except the output head) shared across tasks. Forgetting in the shared body is the primary mechanism of catastrophic interference.

**Sequential training.** Training on tasks one at a time, in order, without revisiting earlier task data. The naive baseline for continual learning.

**Naive sequential training.** Sequential training with no special mechanism to prevent forgetting. The model simply minimizes the current task's loss, ignoring all previous tasks. This is the baseline used in the Split-MNIST project.

**Backward transfer.** The effect of learning a new task on performance on previously learned tasks. Negative backward transfer is catastrophic forgetting.

**Forward transfer.** The effect of previously learned tasks on the speed or quality of learning a new task. Positive forward transfer means prior knowledge helps.

**Average accuracy.** The mean test accuracy across all tasks after training on all tasks. The primary evaluation metric in continual learning.

---

## What / Why / When / How

**What** is continual learning? A learning paradigm in which a model is trained sequentially on a series of tasks, with the goal of performing well on all tasks simultaneously after training is complete.

**Why** is continual learning hard? Because gradient descent on the current task's loss moves parameters away from the solution for previous tasks. With shared weights and no mechanism to protect old knowledge, the model overwrites what it has learned. This is catastrophic forgetting, covered in depth in Chapter 4.

**When** does the scenario matter? The choice of scenario (task-IL, domain-IL, class-IL) determines what information is available at test time and therefore what the model must learn. Task-incremental is appropriate when task labels are naturally available at deployment (e.g., a robot that knows which environment it is in). Class-incremental is appropriate when no such label is available and the model must generalize across all tasks simultaneously.

**How** are the three scenarios implemented differently? The key difference is in the output layer and what information is passed at test time. Task-IL: route to the task-specific head using the provided task label. Domain-IL: use a single fixed-size head, no task label. Class-IL: use a single growing head, no task label. The shared body and training procedure are otherwise identical.

---

## Advantages and Disadvantages

**Task-incremental learning:**
- Advantages: easiest to solve; per-task heads prevent output-layer interference; task label provides strong signal.
- Disadvantages: requires task label at test time, which may not be available in practice; does not test the model's ability to distinguish tasks.

**Domain-incremental learning:**
- Advantages: more realistic than task-IL when the label space is fixed; tests robustness to distribution shift.
- Disadvantages: harder than task-IL; the model must handle multiple input distributions with a single head.

**Class-incremental learning:**
- Advantages: most realistic; tests the full continual learning challenge including implicit task identification.
- Disadvantages: hardest; naive methods fail catastrophically; requires sophisticated mechanisms to maintain old class boundaries.

**Naive sequential training (the project baseline):**
- Advantages: simple, fast, no hyperparameters beyond the standard optimizer; provides a clear lower bound on performance.
- Disadvantages: severe forgetting; not a practical solution for any real continual learning problem.

---

## Historical Context

The continual learning problem was recognized in the connectionist literature in the late 1980s, primarily through the work of McCloskey and Cohen (1989) on catastrophic interference (see Chapter 4). The field gained momentum in the 2010s as deep learning demonstrated strong performance on individual tasks, raising the question of whether the same networks could learn multiple tasks sequentially. Early work focused on regularization-based methods (Kirkpatrick et al., 2017; Zenke et al., 2017) and replay-based methods (Lopez-Paz and Ranzato, 2017). The three-scenario taxonomy of van de Ven and Tolias (2019) provided a unifying framework that clarified why different methods succeed in different settings and why results across papers were often incomparable. This taxonomy is now the standard framing for continual learning research.

---

## Comparison

| Scenario | Task label at test? | Output head | Difficulty | Typical forgetting |
|----------|--------------------|-----------|-----------|--------------------|
| Task-incremental | Yes | Per-task | Easiest | Moderate (body only) |
| Domain-incremental | No | Single fixed | Medium | Moderate to severe |
| Class-incremental | No | Single growing | Hardest | Severe |

**Why task-incremental is easiest:** The task label eliminates the need for the model to distinguish tasks at inference. The per-task output heads mean that output-layer weights for old tasks are never overwritten by new task training. Only the shared body is subject to forgetting, and even there, the gradient signal is more focused.

**Why class-incremental is hardest:** Without a task label, the model must implicitly identify which task an input belongs to and route it to the correct output neurons -- all within a single forward pass through a shared network. As new classes are added, the output layer is retrained, directly overwriting the decision boundaries for old classes. Even with a perfect shared body, the output layer alone causes severe forgetting.

---

## Visual Intuition

```
THE THREE CONTINUAL LEARNING SCENARIOS
(Split-MNIST example: 5 binary tasks)

TASK-INCREMENTAL (Task-IL)
--------------------------
Training:  Task 1 data -> [Shared Body] -> [Head 1: 0 vs 1]
           Task 2 data -> [Shared Body] -> [Head 2: 2 vs 3]
           ...
Test time: Input x + task label t=2 -> [Shared Body] -> [Head 2] -> prediction

  [Shared Body]
       |
  +----+----+----+----+----+
  |    |    |    |    |    |
[H1] [H2] [H3] [H4] [H5]   <- 5 separate 2-neuron heads
 0v1  2v3  4v5  6v7  8v9

Task label tells us which head to use. Easy.

DOMAIN-INCREMENTAL (Domain-IL)
-------------------------------
Training:  Task 1 data -> [Shared Body] -> [Single Head: 10 classes]
           Task 2 data -> [Shared Body] -> [Single Head: 10 classes]
           ...
Test time: Input x (no task label) -> [Shared Body] -> [Single Head] -> prediction

  [Shared Body]
       |
  [Single Head: 10 outputs]

No task label. Must classify correctly across all domains. Medium difficulty.

CLASS-INCREMENTAL (Class-IL)
-----------------------------
After Task 1: [Shared Body] -> [Head: 2 outputs (0, 1)]
After Task 2: [Shared Body] -> [Head: 4 outputs (0, 1, 2, 3)]
After Task 5: [Shared Body] -> [Head: 10 outputs (0..9)]

Test time: Input x (no task label) -> [Shared Body] -> [Head: 10 outputs] -> prediction

Must identify BOTH which task and which class. Hardest.
Old class boundaries overwritten as head grows.
```

**Analogy.** Imagine a student who must learn five foreign languages sequentially, with no access to earlier textbooks once a new language begins.

- Task-IL: the exam tells the student which language each question is in. Easy -- just recall the right vocabulary.
- Domain-IL: the exam uses the same vocabulary across all languages (e.g., numbers), but the student must answer without being told the language. Medium -- the vocabulary overlaps but the rules differ.
- Class-IL: the exam mixes all languages with no labels, and the student must identify the language and answer correctly. Hard -- requires both language identification and recall.

---

## Mathematical Foundations

**Formal task sequence.** Let $\mathcal{T} = (T_1, T_2, \ldots, T_K)$ be a sequence of tasks. Each task $T_t$ is defined by:
- A data distribution $p_t(x, y)$
- A dataset $\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^{N_t}$ sampled i.i.d. from $p_t$
- A label set $\mathcal{Y}_t$ (which may overlap with or be disjoint from $\mathcal{Y}_{t'}$ for $t' \neq t$)

**The continual learning objective.** After training on all $K$ tasks, the goal is to minimize:

$$\mathcal{L}_{\text{CL}}(\theta) = \frac{1}{K} \sum_{t=1}^K \mathbb{E}_{(x,y) \sim p_t} [\ell(f_\theta(x, t), y)]$$

where $f_\theta(x, t)$ denotes the model's prediction for input $x$ given task identity $t$ (in task-IL) or $f_\theta(x)$ without task identity (in domain-IL and class-IL).

**Naive sequential training.** At step $t$, the model minimizes only:

$$\hat{\mathcal{L}}_t(\theta) = \frac{1}{N_t} \sum_{i=1}^{N_t} \ell(f_\theta(x_i^{(t)}, t), y_i^{(t)})$$

This is equivalent to minimizing $\mathcal{L}_{\text{CL}}$ with all previous task losses set to zero. The gradient update:

$$\theta \leftarrow \theta - \eta \nabla_\theta \hat{\mathcal{L}}_t(\theta)$$

moves parameters in a direction that reduces loss on task $t$ but may increase loss on tasks $1, \ldots, t-1$. The expected increase in loss on task $t'$ after training on task $t$ is:

$$\Delta \mathcal{L}_{t'} = \mathcal{L}_{t'}(\theta_t) - \mathcal{L}_{t'}(\theta_{t-1})$$

When $\Delta \mathcal{L}_{t'} > 0$ for $t' < t$, we say task $t'$ has been forgotten. The total forgetting after training on all $K$ tasks is:

$$\mathcal{F} = \frac{1}{K-1} \sum_{t=1}^{K-1} \left[ \max_{t' \leq t} \text{Acc}_{t'}(\theta_{t'}) - \text{Acc}_{t'}(\theta_K) \right]$$

where $\text{Acc}_{t'}(\theta)$ is the accuracy on task $t'$ with parameters $\theta$, and $\theta_K$ are the final parameters after training on all $K$ tasks.

**Why task-IL forgetting is bounded.** In task-IL with per-task heads, the output layer weights $W_{\text{head},t}$ for task $t$ are never updated after task $t$ is complete (assuming the head is frozen or separate). Only the shared body weights $\theta_{\text{body}}$ are updated. The forgetting is therefore limited to the degradation of the shared representation, which is typically less severe than full output-layer overwriting.

**Why class-IL forgetting is unbounded.** In class-IL, the output layer is a single matrix $W_{\text{head}} \in \mathbb{R}^{C_t \times d}$ where $C_t$ grows with $t$. Training on task $t$ updates all rows of $W_{\text{head}}$ (or at least the rows corresponding to new classes), directly overwriting the decision boundaries for old classes. This is a structural source of forgetting that exists even if the shared body is perfectly preserved.

---

## Implementation Notes

**Task-incremental implementation in PyTorch.** Use a `ModuleList` of output heads, one per task. At training and test time, index into the list using the task label.

```python
class TaskILModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_tasks, n_classes_per_task):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, n_classes_per_task)
            for _ in range(n_tasks)
        ])

    def forward(self, x, task_id):
        features = self.body(x)
        return self.heads[task_id](features)
```

**Class-incremental implementation.** Use a single output head that grows as new classes are added. When adding task $t$, expand the output layer by $|\mathcal{Y}_t|$ neurons, initializing new weights randomly and preserving old weights.

**Label remapping.** In Split-MNIST, each task's labels must be remapped to $\{0, 1\}$ for task-IL (so the 2-neuron head can be used), but kept as global class indices $\{0, \ldots, 9\}$ for class-IL.

**Evaluation protocol.** After training on each task $t$, evaluate accuracy on the test sets of all tasks $1, \ldots, t$. Record the full accuracy matrix $A \in \mathbb{R}^{K \times K}$ where $A_{t,t'}$ is the accuracy on task $t'$ after training on task $t$. Average accuracy is $\frac{1}{K} \sum_{t'} A_{K,t'}$. Forgetting is computed from the diagonal and final row of $A$.

---

## Examples

**Simple example: two-task continual learning.** A model is trained first on Task 1 (classify cats vs dogs) and then on Task 2 (classify cars vs trucks). In task-IL, the model has two 2-neuron heads and uses the task label to select the correct one. In class-IL, the model has a single 4-neuron head and must classify any image into one of four classes without being told whether it is an animal or vehicle image.

**Running example: Split-MNIST task-incremental.** The MLP has a shared body (784 -> 256 -> 128, ReLU) and five 2-neuron heads. Training proceeds:
1. Train on Task 1 (digits 0, 1) for 10 epochs with Adam lr=0.001, batch 128.
2. Train on Task 2 (digits 2, 3) for 10 epochs. Evaluate on Tasks 1 and 2.
3. Continue through Task 5.

After Task 5, the average accuracy across all 5 tasks is the primary metric. With naive sequential training, accuracy on Task 1 typically drops significantly by the time Task 5 is complete, because the shared body has been updated to serve Tasks 2-5 at the expense of Task 1.

**Running example: Split-MNIST class-incremental (proposal addition).** The same MLP body, but with a single output head that grows from 2 to 10 neurons across the 5 tasks. At test time after Task 5, the model must classify any digit image into one of 10 classes without a task label. Naive sequential training is expected to produce near-chance performance on early tasks, as the output layer is completely retrained for each new pair of classes.

---

## Current Research

- **Benchmark standardization.** The field has moved toward standardized evaluation protocols following van de Ven and Tolias (2019), but inconsistencies remain. Recent surveys (De Lange et al., 2022) catalog methods and their performance across scenarios.
- **Realistic CL settings.** Researchers are moving beyond fixed task boundaries to online, blurry, and continual pre-training settings where task boundaries are gradual or absent.
- **Class-IL as the primary challenge.** Most recent CL methods target class-incremental learning, as it is the most practically relevant and hardest scenario. Methods include DER++ (Buzzega et al., 2020), FOSTER (Wang et al., 2022), and MEMO (Zhou et al., 2022).
- **Evaluation metrics.** Beyond average accuracy and forgetting, researchers study forward transfer, backward transfer, and computational cost as a function of the number of tasks.

---

## References

- van de Ven, G. M. and Tolias, A. S. (2019). Three scenarios for continual learning. *NeurIPS 2019 Workshop on Continual Learning*.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., and Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.
- Lopez-Paz, D. and Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS 2017*.
- Zenke, F., Poole, B., and Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML 2017*.
- McCloskey, M. and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.
- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.

---

# Chapter 4: Catastrophic Forgetting and the Stability-Plasticity Dilemma

## Overview

Catastrophic forgetting -- also called catastrophic interference -- is the tendency of a neural network to abruptly and severely lose performance on previously learned tasks when trained on new ones. It is the central obstacle in continual learning and the primary motivation for the research project this guide accompanies. The phenomenon arises from a fundamental tension in any learning system: to learn new information, the system must change its internal representations (plasticity), but changing those representations risks destroying previously encoded knowledge (stability). This tension is called the stability-plasticity dilemma. Understanding catastrophic forgetting at a mechanistic level -- why it happens, when it is severe, and what factors modulate it -- is essential for evaluating whether spike sparsity in spiking neural networks can serve as a natural mitigation mechanism.

---

## Fundamental Theory

Catastrophic forgetting is not a bug in any particular algorithm; it is a structural consequence of using shared, continuously updated parameters to represent multiple tasks. The same weights that encode knowledge about Task A are modified when learning Task B, and if the gradient of Task B's loss points in a direction that is harmful for Task A, the knowledge of Task A is overwritten.

**The stability-plasticity dilemma.** Any learning system must balance two competing demands:

- **Plasticity:** the ability to rapidly acquire new information by modifying internal representations.
- **Stability:** the ability to retain previously acquired information against interference from new learning.

A system that is maximally plastic (e.g., a network with a very high learning rate that fully adapts to each new task) will forget everything it has learned before. A system that is maximally stable (e.g., a network with frozen weights) cannot learn anything new. The dilemma is that increasing one property tends to decrease the other, and there is no free lunch: any mechanism that protects old knowledge must constrain the freedom to learn new knowledge.

This dilemma was first articulated in the neuroscience literature by Grossberg (1987) and has since become a central organizing concept in both computational neuroscience and machine learning.

---

## Technical Explanation

### Why Catastrophic Forgetting Happens: The Shared Weights Mechanism

Consider a network with parameters $\theta$ trained sequentially on tasks $T_1$ and $T_2$. After training on $T_1$, the parameters $\theta_1$ encode a solution to $T_1$: the weights are configured such that the network correctly classifies inputs from $T_1$. When training begins on $T_2$, the optimizer computes gradients of $\mathcal{L}_{T_2}$ with respect to $\theta$ and updates:

$$\theta_2 = \theta_1 - \eta \nabla_\theta \mathcal{L}_{T_2}(\theta_1)$$

The gradient $\nabla_\theta \mathcal{L}_{T_2}$ is computed entirely with respect to $T_2$'s loss. It has no information about $T_1$. If the gradient direction that reduces $\mathcal{L}_{T_2}$ also increases $\mathcal{L}_{T_1}$, then the update harms $T_1$'s performance. After many such updates over 10 epochs of $T_2$ training, the parameters $\theta_2$ may be far from the region of parameter space that solves $T_1$, and performance on $T_1$ collapses.

The key insight is that the gradient of $\mathcal{L}_{T_2}$ is not constrained to be orthogonal to the gradient of $\mathcal{L}_{T_1}$. In general, for two tasks with different data distributions, the loss landscapes are different, and the directions of steepest descent are not aligned. The more different the tasks, the more likely the gradients are to conflict.

### The Role of Representational Overlap

Forgetting is more severe when the two tasks use the same neurons and weights to represent their solutions. If Task 1 uses neurons $\{n_1, n_2, n_3\}$ and Task 2 uses neurons $\{n_4, n_5, n_6\}$ (disjoint sets), then training on Task 2 does not affect the weights connected to $\{n_1, n_2, n_3\}$, and Task 1 is preserved. But if both tasks use the same neurons, the weights are shared, and updating them for Task 2 directly interferes with Task 1.

This is the key insight behind the project's central hypothesis: if spiking neural networks with high spike sparsity activate different subsets of neurons for different tasks, the representational overlap is reduced, and catastrophic forgetting may be mitigated. Sparse representations naturally tend toward disjoint activation patterns, which is the biological analogue of the "sparse coding" hypothesis in neuroscience.

### A Brief Taxonomy of Continual Learning Defences

The research project uses naive sequential training (no defence) as its baseline, specifically to isolate the effect of spike sparsity on forgetting. However, it is important to understand the landscape of existing defences to contextualize the project's contribution.

**Regularization-based methods** add a penalty term to the loss that discourages large changes to parameters that are important for previous tasks.

- *Elastic Weight Consolidation (EWC)* (Kirkpatrick et al., 2017): adds a quadratic penalty $\frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{1,i})^2$ where $F_i$ is the Fisher information of parameter $i$ with respect to Task 1's data. Parameters with high Fisher information are important for Task 1 and are penalized more for changing.
- *Synaptic Intelligence (SI)* (Zenke et al., 2017): accumulates an online estimate of each parameter's importance based on how much it contributed to reducing the loss during training, and uses this to construct a similar quadratic penalty.

**Replay-based methods** maintain a buffer of examples from previous tasks and interleave them with new task data during training.

- *Gradient Episodic Memory (GEM)* (Lopez-Paz and Ranzato, 2017): stores a small episodic memory of previous task examples and constrains the gradient update to not increase the loss on any stored example.
- *Experience Replay (ER)*: simply replays stored examples alongside new task data.

**Parameter isolation methods** allocate different subsets of parameters to different tasks, preventing any overlap.

- *PackNet* (Mallya and Lazebnik, 2018): after training on each task, prunes the network to identify the most important weights, freezes them, and uses the remaining free weights for the next task.
- *Progressive Neural Networks* (Rusu et al., 2016): add a new column of neurons for each task, with lateral connections to previous columns. Old columns are frozen.

**Knowledge distillation methods** use the predictions of the old model as soft targets to prevent the new model from drifting too far.

- *Learning without Forgetting (LwF)* (Li and Hoiem, 2017): when training on a new task, adds a distillation loss that encourages the new model's predictions on new task data to match the old model's predictions on the same data.

**The project's approach.** The pilot study uses naive sequential training with no defence. The hypothesis is that the intrinsic properties of spiking neural networks -- specifically, the sparse activation patterns induced by the leaky integrate-and-fire neuron model -- may provide a form of implicit parameter isolation, reducing forgetting without any explicit mechanism. This is tested by comparing forgetting rates between standard MLPs and SNNs under identical naive training conditions.

---

## Core Concepts

**Catastrophic forgetting (catastrophic interference).** The abrupt, severe loss of performance on previously learned tasks when a neural network is trained on new tasks. First documented in connectionist networks by McCloskey and Cohen (1989).

**Stability-plasticity dilemma.** The fundamental tension between the need to retain old knowledge (stability) and the need to acquire new knowledge (plasticity). Any learning system must navigate this trade-off.

**Representational overlap.** The degree to which two tasks activate the same neurons and use the same weights. High overlap leads to more interference; low overlap (sparse, disjoint representations) leads to less.

**Fisher information.** A measure of how much information the data provides about a parameter. In the context of EWC, parameters with high Fisher information are important for the current task and should be protected from large changes.

**Quadratic penalty.** A regularization term of the form $\frac{\lambda}{2} \sum_i \Omega_i (\theta_i - \theta_i^*)^2$ that penalizes deviation from a reference parameter vector $\theta^*$. Used in EWC and SI to protect important parameters.

**Episodic memory.** A small buffer of stored examples from previous tasks, used in replay-based methods to prevent forgetting.

**Backward transfer.** The change in performance on a previous task caused by learning a new task. Negative backward transfer is forgetting; positive backward transfer is rare but possible (learning Task 2 improves Task 1).

**Plasticity loss.** The reduction in a model's ability to learn new tasks caused by mechanisms designed to prevent forgetting. A model that is too stable cannot adapt to new tasks.

---

## What / Why / When / How

**What** is catastrophic forgetting? The phenomenon where training a neural network on a new task causes it to rapidly lose the ability to perform previously learned tasks, because the shared weights are overwritten by the new task's gradient updates.

**Why** does it happen? Because gradient descent on the new task's loss moves parameters in directions that reduce the new task's error, without any constraint to preserve the old task's solution. The shared weight structure means that any parameter update affects all tasks simultaneously.

**When** is forgetting most severe? Forgetting is most severe when: (1) the tasks are dissimilar (their loss landscapes conflict strongly); (2) the network has high representational overlap between tasks (same neurons used for both); (3) the learning rate is high (large parameter updates); (4) training on the new task is long (many gradient steps); (5) the network capacity is small relative to the number of tasks.

**How** can forgetting be measured? The standard measure is the drop in accuracy on Task $t$ between the time immediately after training on Task $t$ and the time after training on all subsequent tasks. Formally: $\text{Forgetting}_t = \text{Acc}_t(\theta_t) - \text{Acc}_t(\theta_K)$ where $\theta_t$ are the parameters after Task $t$ and $\theta_K$ are the final parameters.

---

## Advantages and Disadvantages

**Naive sequential training (the project baseline):**
- Advantages: maximally plastic; learns each new task as well as possible; simple to implement; provides a clear lower bound on continual learning performance.
- Disadvantages: maximally forgetful; performance on early tasks collapses to near-chance after many tasks; not a practical solution.

**Regularization-based methods (EWC, SI):**
- Advantages: no additional memory for storing old data; computationally efficient; principled Bayesian interpretation (EWC approximates a Laplace approximation to the posterior).
- Disadvantages: the quadratic penalty is a local approximation that becomes inaccurate far from the reference point; performance degrades as the number of tasks grows; requires storing importance weights for all parameters.

**Replay-based methods (GEM, ER):**
- Advantages: directly addresses the root cause (lack of old data); strong empirical performance; flexible.
- Disadvantages: requires storing old data, which may violate privacy constraints or memory budgets; the buffer size is a critical hyperparameter; does not scale well to many tasks.

**Parameter isolation methods (PackNet):**
- Advantages: zero forgetting by construction (old parameters are frozen); scales to many tasks.
- Disadvantages: requires knowing task boundaries; network capacity is consumed over time; eventually runs out of free parameters.

---

## Historical Context

The phenomenon of catastrophic forgetting was first systematically documented by McCloskey and Cohen (1989) in a study of connectionist networks learning paired-associate lists. They showed that training a network on a second list of word pairs caused it to completely forget the first list, in stark contrast to human memory, which shows gradual, partial forgetting (interference) rather than catastrophic loss. The term "catastrophic interference" was coined to distinguish this from the more gradual forgetting seen in biological systems.

The problem was largely set aside during the 1990s and early 2000s as the field focused on single-task learning. It returned to prominence with the rise of deep learning, as researchers began to ask whether deep networks could serve as general-purpose learning systems. Goodfellow et al. (2013) provided an empirical analysis of catastrophic forgetting in deep networks, showing that it is a pervasive problem. Kirkpatrick et al. (2017) introduced EWC, drawing an explicit connection to Bayesian continual learning and the neuroscience of synaptic consolidation. This paper sparked a wave of research that continues to the present day.

The stability-plasticity dilemma has deep roots in neuroscience. Grossberg (1987) introduced the term in the context of adaptive resonance theory. The complementary learning systems theory (McClelland, McNaughton, and O'Reilly, 1995) proposed that the brain resolves the dilemma by using two complementary memory systems: the hippocampus (fast, plastic, episodic) and the neocortex (slow, stable, semantic). This biological insight has inspired replay-based continual learning methods.

---

## Comparison

| Method | Forgetting | Plasticity | Memory cost | Compute cost | Scalability |
|--------|-----------|-----------|-------------|-------------|-------------|
| Naive sequential | Severe | High | None | Low | High |
| EWC | Moderate | Moderate | O(P) importance weights | Low | Degrades with tasks |
| SI | Moderate | Moderate | O(P) importance weights | Low | Degrades with tasks |
| GEM | Low | Moderate | O(buffer * d) | Medium | Limited by buffer |
| PackNet | None | Decreasing | None extra | Medium | Limited by capacity |
| LwF | Moderate | Moderate | None | Medium | Moderate |

**EWC vs SI:** EWC computes importance weights using the Fisher information matrix (expensive, requires a pass over old data after each task). SI computes importance weights online during training (cheaper, no old data needed). SI is more practical but less theoretically grounded.

**Regularization vs replay:** Regularization methods are more memory-efficient but less effective, especially in class-incremental settings. Replay methods are more effective but require storing old data.

**The project's implicit comparison:** By using naive sequential training as the baseline and comparing SNN vs MLP forgetting rates, the project implicitly tests whether spike sparsity provides a form of implicit parameter isolation -- a property that PackNet achieves explicitly through pruning and freezing.

---

## Visual Intuition

```
CATASTROPHIC FORGETTING: PARAMETER SPACE VIEW

Imagine parameter space as a 2D landscape (in reality it is millions of dimensions).

After training on Task 1:
  theta_1 is in the "valley" for Task 1 (low loss on T1)

  Loss(T1)
    |
    |    *theta_1
    |   / \
    |  /   \
    | /     \
    +-----------> theta

After training on Task 2 (naive sequential):
  Gradient of Loss(T2) pushes theta away from theta_1
  theta_2 is in the valley for Task 2, but far from Task 1's valley

  Loss(T1)  Loss(T2)
    |           |
    |    *theta_1    *theta_2
    |   / \        / \
    |  /   \      /   \
    | /     \    /     \
    +----------------------------> theta

  theta_2 is in a high-loss region for Task 1 --> FORGETTING

REPRESENTATIONAL OVERLAP AND FORGETTING

High overlap (standard MLP):
  Task 1 uses neurons: [n1, n2, n3, n4, n5, n6, n7, n8]
  Task 2 uses neurons: [n1, n2, n3, n4, n5, n6, n7, n8]
  Overlap: 100% --> Training on T2 overwrites ALL T1 weights --> severe forgetting

Low overlap (sparse SNN hypothesis):
  Task 1 uses neurons: [n1, n2, n3]  (sparse: only 3 of 8 active)
  Task 2 uses neurons: [n6, n7, n8]  (sparse: different 3 of 8 active)
  Overlap: 0% --> Training on T2 does NOT touch T1 weights --> no forgetting

Reality (partial overlap):
  Task 1 uses neurons: [n1, n2, n3, n4]
  Task 2 uses neurons: [n3, n4, n5, n6]
  Overlap: 50% (n3, n4) --> partial forgetting proportional to overlap
```

**Analogy: the overwritten notebook.** Imagine a student who takes notes in a single notebook. After studying Topic A, the notebook contains Topic A's notes. When studying Topic B, the student erases the notebook and writes Topic B's notes. Topic A is completely forgotten. This is catastrophic forgetting with 100% representational overlap.

Now imagine the student uses different pages for different topics. Topic A uses pages 1-10, Topic B uses pages 11-20. Writing Topic B's notes does not affect Topic A's pages. This is parameter isolation with 0% overlap.

Sparse representations are like using different pages: each task activates a different sparse subset of neurons, so the weights for one task are not overwritten by another.

**The stability-plasticity dilemma as a dial:**

```
STABILITY <-----------------------------------------> PLASTICITY

Frozen weights     EWC/SI      Naive SGD     High LR SGD
(no forgetting,   (moderate   (severe        (instant
 no learning)      balance)    forgetting,    forgetting,
                               good           good
                               plasticity)    plasticity)
```

---

## Mathematical Foundations

**Formalizing forgetting.** Let $\theta_t$ denote the parameters after training on task $t$. Define the accuracy of the model on task $t'$ with parameters $\theta$ as $A_{t'}(\theta)$. The forgetting on task $t'$ after training through task $T$ is:

$$\mathcal{F}_{t'} = A_{t'}(\theta_{t'}) - A_{t'}(\theta_T), \quad t' < T$$

The average forgetting is:

$$\bar{\mathcal{F}} = \frac{1}{T-1} \sum_{t'=1}^{T-1} \mathcal{F}_{t'}$$

**Gradient interference.** The gradient of Task 2's loss with respect to parameters $\theta$ is $g_2 = \nabla_\theta \mathcal{L}_{T_2}(\theta)$. The gradient of Task 1's loss is $g_1 = \nabla_\theta \mathcal{L}_{T_1}(\theta)$. The update $\theta \leftarrow \theta - \eta g_2$ increases Task 1's loss by approximately:

$$\Delta \mathcal{L}_{T_1} \approx \eta \langle g_1, g_2 \rangle$$

where $\langle \cdot, \cdot \rangle$ is the inner product. If $\langle g_1, g_2 \rangle > 0$, the update helps both tasks. If $\langle g_1, g_2 \rangle < 0$, the update helps Task 2 but hurts Task 1. GEM (Lopez-Paz and Ranzato, 2017) explicitly constrains updates to satisfy $\langle g_1, g_2 \rangle \geq 0$.

**EWC: Fisher information weighting.** The Fisher information matrix $F$ for task $T_1$ is:

$$F_{ij} = \mathbb{E}_{x \sim p_{T_1}} \left[ \frac{\partial \log p_\theta(y|x)}{\partial \theta_i} \frac{\partial \log p_\theta(y|x)}{\partial \theta_j} \right]$$

EWC uses the diagonal of $F$ as importance weights. The regularized loss for task $T_2$ is:

$$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{T_2}(\theta) + \frac{\lambda}{2} \sum_i F_{ii} (\theta_i - \theta_{1,i})^2$$

The quadratic penalty anchors each parameter $\theta_i$ near its Task 1 value $\theta_{1,i}$, with strength proportional to $F_{ii}$. Parameters that were important for Task 1 (high $F_{ii}$) are penalized more for changing.

**Bayesian interpretation of EWC.** EWC can be derived from a Bayesian perspective. After training on Task 1, the posterior over parameters is $p(\theta | \mathcal{D}_1) \propto p(\mathcal{D}_1 | \theta) p(\theta)$. When training on Task 2, the ideal update uses this posterior as the prior:

$$p(\theta | \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 | \theta) p(\theta | \mathcal{D}_1)$$

EWC approximates $p(\theta | \mathcal{D}_1)$ as a Gaussian centered at $\theta_1$ with precision matrix $F$. This is a Laplace approximation to the posterior. The resulting regularized loss is exactly the EWC objective.

**Representational overlap and forgetting (formal).** Let $S_t \subseteq \{1, \ldots, N\}$ be the set of neurons active for task $t$ (i.e., neurons with non-zero activation on task $t$'s inputs). The representational overlap between tasks $t$ and $t'$ is:

$$\text{Overlap}(t, t') = \frac{|S_t \cap S_{t'}|}{|S_t \cup S_{t'}|}$$

(Jaccard similarity). When $\text{Overlap}(t, t') = 0$, the tasks use completely disjoint neurons, and training on task $t'$ does not affect the weights used by task $t$. When $\text{Overlap}(t, t') = 1$, all neurons are shared, and forgetting is maximal.

The project's central hypothesis is that SNNs with high spike sparsity have lower $\text{Overlap}(t, t')$ than standard MLPs, leading to less forgetting under naive sequential training.

---

## Implementation Notes

**Measuring forgetting in practice.** After training on each task $t$, evaluate the model on the test sets of all tasks $1, \ldots, t$. Store the accuracy matrix $A \in \mathbb{R}^{T \times T}$ where $A_{t, t'}$ is the accuracy on task $t'$ after training on task $t$. Forgetting on task $t'$ is $A_{t', t'} - A_{T, t'}$.

**Naive sequential training in PyTorch.** Simply iterate over tasks and call the standard training loop for each:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for task_id, task_loader in enumerate(task_loaders):
    for epoch in range(10):
        for x, y in task_loader:
            optimizer.zero_grad()
            logits = model(x, task_id)  # task-IL: pass task_id
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    # Evaluate on all tasks seen so far
    for t in range(task_id + 1):
        acc = evaluate(model, test_loaders[t], t)
        print(f"After task {task_id+1}, accuracy on task {t+1}: {acc:.3f}")
```

**Pitfalls:**
- Do not shuffle data across tasks during training. Each task must be presented as a contiguous block.
- Ensure the optimizer state (Adam's moment estimates) is reset between tasks if you want a clean baseline. In practice, carrying over the optimizer state is the standard naive baseline.
- Use the same random seed for all experiments to ensure reproducibility.
- Report both per-task accuracy and average accuracy. A single number hides the forgetting pattern.

**Computational complexity of EWC.** Computing the diagonal Fisher information requires one forward-backward pass over the Task 1 training set after training is complete. This is $O(N_1 \cdot P)$ where $N_1$ is the Task 1 dataset size and $P$ is the number of parameters. Storing the importance weights requires $O(P)$ additional memory.

---

## Examples

**Simple example: two-neuron network.** Consider a network with two neurons, $n_1$ and $n_2$. Task 1 requires $n_1$ to be active (weight $w_1 = 1$) and $n_2$ to be inactive ($w_2 = 0$). Task 2 requires the opposite: $n_1$ inactive ($w_1 = 0$) and $n_2$ active ($w_2 = 1$). Training on Task 2 sets $w_1 = 0$, which destroys the Task 1 solution. This is catastrophic forgetting in its simplest form. If instead Task 2 required $n_3$ (a third neuron not used by Task 1), there would be no forgetting.

**Running example: Split-MNIST naive sequential training.** The MLP (784 -> 256 -> 128 -> 2, per-task heads) is trained on Tasks 1-5 sequentially with Adam lr=0.001, batch 128, 10 epochs per task. Typical results under naive sequential training:

```
After Task 1: Acc(T1) = 99.5%
After Task 2: Acc(T1) = 95.2%, Acc(T2) = 99.3%
After Task 3: Acc(T1) = 88.1%, Acc(T2) = 91.4%, Acc(T3) = 99.1%
After Task 4: Acc(T1) = 72.3%, Acc(T2) = 80.2%, Acc(T3) = 90.5%, Acc(T4) = 99.0%
After Task 5: Acc(T1) = 55.8%, Acc(T2) = 68.4%, Acc(T3) = 78.2%, Acc(T4) = 88.1%, Acc(T5) = 98.9%
Average accuracy after Task 5: 77.9%
Forgetting on Task 1: 99.5% - 55.8% = 43.7%
```

(These numbers are illustrative; actual results depend on architecture and random seed.)

The project compares these forgetting rates between a standard MLP and an SNN with equivalent architecture, testing whether the SNN's sparse spike patterns reduce the forgetting on Tasks 1-4 when Task 5 is trained.

---

## Current Research

- **Theoretical understanding of forgetting.** Ramasesh et al. (2021) showed that forgetting in task-incremental learning is related to the geometry of the loss landscape and the angle between task gradients. Tasks with more similar gradients forget less.
- **Forgetting in large language models.** Catastrophic forgetting is a major challenge in fine-tuning large pre-trained models. Methods like LoRA (Hu et al., 2022) and adapter layers partially address this by limiting the number of parameters updated during fine-tuning.
- **Biological plausibility.** The complementary learning systems theory (McClelland et al., 1995) proposes that the brain avoids catastrophic forgetting through a two-memory system: fast hippocampal learning and slow neocortical consolidation. Replay-based CL methods are directly inspired by this theory.
- **Sparsity and forgetting.** Several recent works have explored the connection between sparse representations and reduced forgetting. French (1999) showed theoretically that sparse, non-overlapping representations minimize interference. Sparse autoencoders and sparse coding networks have been shown empirically to forget less than dense networks (Aljundi et al., 2019).
- **The project's contribution.** The project tests whether the biologically motivated sparse firing patterns of spiking neural networks provide a natural, implicit form of this sparsity-based protection, without any explicit regularization or replay mechanism.

---

## References

- McCloskey, M. and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., and Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.
- Zenke, F., Poole, B., and Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML 2017*.
- Lopez-Paz, D. and Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS 2017*.
- Li, Z. and Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12), 2935-2947.
- Mallya, A. and Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR 2018*.
- van de Ven, G. M. and Tolias, A. S. (2019). Three scenarios for continual learning. *NeurIPS 2019 Workshop on Continual Learning*.
- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.
- Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., and Bengio, Y. (2013). An empirical investigation of catastrophic forgetting in gradient-based neural networks. *ICLR 2014 Workshop*.

---

*End of Part 1. Part 2 will introduce spiking neural networks, the leaky integrate-and-fire neuron model, surrogate gradient training, and the relationship between spike sparsity and representational overlap.*



---

# Companion Learning Guide -- Part 2: Spiking Neural Networks

> **Note:** This is Part 2 of a multi-part companion guide accompanying a research proposal on spike sparsity and catastrophic forgetting in spiking neural networks. Part 1 covered the foundations of deep learning and continual learning, including gradient descent, backpropagation, the stability-plasticity dilemma, and canonical continual learning methods such as EWC and replay. This part assumes that material is understood and builds directly on it. Here we develop the core machinery of spiking neural networks from first principles: the biology that motivates them, the mathematical neuron models that make them tractable, the encoding schemes that connect them to standard datasets, the surrogate gradient methods that make them trainable, and the sparsity properties that make them energy-efficient. Every concept is grounded in the specific implementation choices made in the accompanying research project, so you will see concrete numbers and code-level details throughout.

---

## Table of Contents

1. [Chapter 1: Biological Neurons and the Motivation for SNNs](#chapter-1)
2. [Chapter 2: The Leaky Integrate-and-Fire Neuron Model](#chapter-2)
3. [Chapter 3: Encoding and Time in SNNs](#chapter-3)
4. [Chapter 4: Training SNNs with Surrogate Gradients](#chapter-4)
5. [Chapter 5: Spike Sparsity and Energy Efficiency](#chapter-5)

---

<a name="chapter-1"></a>
# Chapter 1: Biological Neurons and the Motivation for SNNs

## Overview

The brain computes using electrical pulses called action potentials or spikes. Each spike is a stereotyped, all-or-nothing event: a neuron either fires or it does not. This binary, event-driven communication is radically different from the continuous, always-on activations used in conventional artificial neural networks (ANNs). Spiking neural networks (SNNs) are a class of neural network models that adopt this spike-based communication, making them the closest computational analogue to biological neural circuits. Understanding why spikes exist in biology, and what computational advantages they might confer, is the essential first step before studying any SNN model.

## Fundamental Theory

A biological neuron is a cell with three functional regions. The **dendrites** receive incoming signals from other neurons. The **soma** (cell body) integrates those signals over time. The **axon** transmits the neuron's output to downstream neurons via synaptic connections.

The key quantity governing a neuron's behavior is its **membrane potential** $V_m(t)$, the electrical voltage difference across the cell membrane. At rest, this sits near $-70$ mV in a typical cortical neuron. When a presynaptic neuron fires, it releases neurotransmitters that cause ion channels in the postsynaptic membrane to open, allowing charged ions (Na$^+$, K$^+$, Ca$^{2+}$, Cl$^-$) to flow across the membrane. This flow changes $V_m$.

If enough excitatory input arrives in a short window, $V_m$ rises to a **threshold** near $-55$ mV. At that point, voltage-gated sodium channels open in a positive feedback loop: Na$^+$ rushes in, $V_m$ shoots up to roughly $+40$ mV, then K$^+$ channels open and $V_m$ rapidly falls back, overshooting to a hyperpolarized state before returning to rest. This entire event -- the action potential -- lasts about 1--2 ms and is essentially identical every time it occurs. The neuron then enters a brief **refractory period** during which it cannot fire again.

The information transmitted is therefore not in the amplitude of the spike (which is fixed) but in the **timing** and **rate** of spikes.

## Technical Explanation

The membrane potential dynamics can be described by a simple RC circuit analogy. The membrane has a capacitance $C_m$ and a leak conductance $g_L$. In the absence of input, the membrane potential decays exponentially toward the resting potential $E_L$:

$$C_m \frac{dV_m}{dt} = -g_L (V_m - E_L) + I(t)$$

where $I(t)$ is the total synaptic input current. This is the foundation of the leaky integrate-and-fire model developed in Chapter 2.

A **synapse** is the junction between two neurons. When a presynaptic spike arrives, it triggers a postsynaptic current (PSC) that decays with a synaptic time constant $\tau_{syn}$. The strength of the synapse is its **weight** $w$, which scales the amplitude of the PSC. Learning in biological networks is thought to occur primarily through changes in synaptic weights, governed by rules such as spike-timing-dependent plasticity (STDP).

## Core Concepts

- **Membrane potential $V_m$:** The electrical state variable of a neuron; integrates input over time.
- **Threshold $V_{th}$:** The voltage at which a spike is triggered.
- **Action potential / spike:** The stereotyped all-or-nothing output event.
- **Refractory period:** The brief post-spike interval during which the neuron cannot fire.
- **Synapse:** The connection between neurons; characterized by a weight $w$ and a time constant $\tau_{syn}$.
- **Leak:** The passive tendency of the membrane potential to return to rest; governed by $\tau_{mem} = C_m / g_L$.

## What / Why / When / How

**What** is a spiking neural network? An SNN is a network of neuron models that communicate via discrete spike events rather than continuous-valued activations. Each neuron maintains an internal state (membrane potential) that evolves over time, and emits a spike when that state crosses a threshold.

**Why** use spikes? Three reasons dominate the literature. First, biological plausibility: the brain uses spikes, so SNNs are a more faithful model of neural computation, which matters for neuroscience applications. Second, energy efficiency: on neuromorphic hardware, a spike triggers computation only when it arrives; neurons that do not spike consume no dynamic power. Third, temporal processing: spikes carry timing information that continuous activations discard, potentially enabling richer representations of time-varying inputs.

**When** are SNNs appropriate? SNNs are most natural for event-driven sensory data (e.g., from dynamic vision sensors / event cameras), for deployment on neuromorphic chips (Intel Loihi, IBM TrueNorth, BrainScaleS), and for research into biologically plausible learning. They are less mature than ANNs for general-purpose vision and language tasks.

**How** do SNNs differ from ANNs computationally? In an ANN, each layer computes $\mathbf{y} = f(W\mathbf{x} + \mathbf{b})$ once per forward pass. In an SNN, the network is simulated over $T$ discrete timesteps; at each step, each neuron updates its membrane potential and may or may not emit a spike. The output of a layer at timestep $t$ is a binary spike vector $\mathbf{s}^t \in \{0,1\}^n$.

## Advantages and Disadvantages

**Advantages:**
- Energy efficiency on neuromorphic hardware: sparse, event-driven computation.
- Temporal dynamics: natural representation of time-varying signals.
- Biological plausibility: closer to how the brain actually works.
- Potential for online, local learning rules (STDP, e-prop).

**Disadvantages:**
- Training difficulty: the spike function is non-differentiable, requiring workarounds (Chapter 4).
- Longer simulation time: $T$ timesteps per forward pass vs. one pass in an ANN.
- Hyperparameter sensitivity: membrane time constants, thresholds, and encoding schemes add complexity.
- Immature tooling: fewer production-ready libraries compared to PyTorch/TensorFlow ecosystems.
- Performance gap: on standard benchmarks, SNNs still lag behind equivalent ANNs in accuracy.

## Historical Context

The history of neural network generations is a useful organizing framework. **First-generation** networks (McCulloch-Pitts neurons, perceptrons) used binary threshold units. **Second-generation** networks (modern ANNs) use continuous-valued, differentiable activation functions and are trained with backpropagation. **Third-generation** networks, as named by Wolfgang Maass in his landmark 1997 paper, use spiking neurons and exploit the timing of spikes as an additional computational resource.

Maass (1997) proved that networks of spiking neurons are computationally more powerful than sigmoidal networks in the sense that they can compute certain functions with fewer neurons when spike timing is used. This theoretical result provided the foundational motivation for the SNN research program. However, the practical challenge of training SNNs efficiently remained unsolved for another two decades, until surrogate gradient methods (Chapter 4) made deep SNNs tractable.

Early SNN models were trained with biologically inspired local rules (Hebbian learning, STDP) rather than gradient descent. These rules are elegant but difficult to scale to deep networks and complex tasks. The modern era of SNN research, beginning roughly around 2018--2020, is characterized by adapting the tools of deep learning -- backpropagation, batch normalization, Adam optimizer -- to the spiking setting.

## Comparison

| Property | ANN (ReLU) | SNN (LIF) |
|---|---|---|
| Neuron output | Continuous real value | Binary spike $\{0,1\}$ |
| Temporal dynamics | None (stateless) | Membrane potential evolves over $T$ steps |
| Computation trigger | Every forward pass | Only on spike arrival (event-driven) |
| Differentiability | Yes (ReLU almost everywhere) | No (Heaviside step function) |
| Energy model | MACs (multiply-accumulate) | SynOps (synaptic operations, additions only) |
| Hardware target | GPU | Neuromorphic chip (Loihi, TrueNorth) |
| Training maturity | Very high | Moderate (surrogate gradients, ~2018+) |

## Visual Intuition

The following ASCII diagram shows the membrane potential of a single LIF neuron receiving a stream of input spikes. The potential rises with each input, leaks between inputs, and fires when it crosses the threshold.

```
V_m
 |
 |  threshold -------- - - - - - - - - - - - - - - - - - - -
 |                                    *
 |                          *        /|
 |               *         / \      / |
 |              /|        /   \    /  |  reset
 |             / |       /     \  /   |----
 |            /  |      /       \/    |
 |           /   |     /              |
 |          /    |----/               |
 | rest ----                          |
 |
 +-----|-----|-----|-----|-----|-----|---> time
       t1    t2    t3    t4    t5   spike

  Each upward step = incoming presynaptic spike
  Downward drift between steps = membrane leak
  * = membrane potential at that moment
  spike at t5: V_m >= threshold, reset to 0
```

The key intuition: the neuron is a **leaky integrator**. It accumulates evidence (input spikes) over time, but that evidence decays if it does not arrive fast enough. Only when enough evidence accumulates quickly enough does the neuron fire.

## Mathematical Foundations

The continuous-time membrane dynamics of a leaky integrate-and-fire neuron are:

$$\tau_{mem} \frac{dV_m}{dt} = -(V_m - V_{rest}) + R \cdot I(t)$$

where $\tau_{mem} = C_m / g_L$ is the membrane time constant, $V_{rest}$ is the resting potential, $R = 1/g_L$ is the membrane resistance, and $I(t)$ is the input current. When $V_m$ reaches threshold $V_{th}$, a spike is emitted and $V_m$ is reset to $V_{reset}$.

Setting $V_{rest} = V_{reset} = 0$ and absorbing $R$ into the input (so the input is already in voltage units), this simplifies to:

$$\tau_{mem} \frac{dV_m}{dt} = -V_m + I(t)$$

This is the form used in most computational SNN work and in the snntorch library. The full derivation of the discrete-time update from this equation is given in Chapter 2.

## Implementation Notes

- In practice, $V_{rest}$ and $V_{reset}$ are both set to 0 in most SNN frameworks, including snntorch. This is a simplification that makes the math cleaner without losing the essential dynamics.
- The refractory period is often omitted in simple LIF implementations; snntorch's `Leaky` neuron does not implement a refractory period by default.
- Synaptic dynamics (the time course of the PSC) are also often omitted; the input to the LIF neuron is treated as an instantaneous current injection at each timestep.
- The "third generation" framing from Maass (1997) is a theoretical classification, not a strict engineering standard. Many modern SNN papers use it loosely.

## Examples

**Simple example:** A single LIF neuron receives a constant input current $I = 1.2$ (in normalized units, threshold = 1.0). With $\tau_{mem} = 20$ ms and $dt = 1$ ms, the membrane potential rises each step and eventually crosses threshold, producing a spike. After reset, the process repeats. This is the simplest possible SNN computation.

**Real-world example:** In the accompanying research project, a two-hidden-layer SNN is trained on Split-MNIST (a continual learning benchmark). Each hidden layer contains LIF neurons with $\tau_{mem} = 20$ ms, $dt = 1$ ms, threshold = 1.0 (default), and reset-to-zero. The network receives 25 timesteps of input per image. The fraction of hidden neurons that spike at least once over those 25 timesteps is the "active neuron percentage," the primary sparsity metric.

## Current Research

Current research on biological motivation for SNNs includes: (1) using SNNs as models of specific brain areas (visual cortex, hippocampus) to test computational hypotheses; (2) studying how biological learning rules like STDP can be reconciled with gradient-based training; (3) investigating whether the temporal coding capacity of SNNs (beyond rate coding) can be exploited for practical tasks; (4) understanding the role of dendritic computation, which simple LIF models ignore entirely.

Open problems include: whether spike timing genuinely provides a computational advantage over rate coding for real-world tasks; how to bridge the gap between biologically plausible local learning rules and the global credit assignment of backpropagation; and how to model the diversity of biological neuron types (not just LIF) in a tractable way.

## References

- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659--1671.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press.
- Roy, K., Jaiswal, A., & Panda, P. (2019). Towards spike-based machine intelligence with neuromorphic computing. *Nature*, 575(7784), 607--617.

---
<a name="chapter-2"></a>
# Chapter 2: The Leaky Integrate-and-Fire Neuron Model

## Overview

The leaky integrate-and-fire (LIF) model is the workhorse of computational neuroscience and modern SNN research. It captures the two most essential features of a biological neuron -- integration of input over time and threshold-triggered spiking -- while discarding the biophysical complexity of Hodgkin-Huxley-style models. The result is a model simple enough to simulate millions of neurons in real time, yet rich enough to exhibit the temporal dynamics that make SNNs interesting. This chapter derives the LIF model from first principles, shows how to discretize it for simulation, and explains every parameter used in the accompanying research project.

## Fundamental Theory

The LIF model is derived from the RC circuit analogy introduced in Chapter 1. The membrane is modeled as a capacitor $C_m$ in parallel with a resistor $R_m = 1/g_L$. The capacitor stores charge (representing the membrane potential), and the resistor allows charge to leak away (representing the passive ion channels that pull the membrane back to rest).

Applying Kirchhoff's current law to this circuit:

$$C_m \frac{dV_m}{dt} = -\frac{V_m - V_{rest}}{R_m} + I(t)$$

Multiplying both sides by $R_m$ and defining $\tau_{mem} = R_m C_m$:

$$\tau_{mem} \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)$$

Setting $V_{rest} = 0$ and absorbing $R_m$ into the input (defining $u(t) = R_m I(t)$ as the effective input in voltage units):

$$\tau_{mem} \frac{dV_m}{dt} = -V_m + u(t)$$

This is the standard LIF ODE. The spike condition and reset rule complete the model:

- **Spike condition:** If $V_m(t) \geq V_{th}$, emit a spike $s(t) = 1$ and reset $V_m$.
- **Reset rule (reset-to-zero):** $V_m \leftarrow V_{reset} = 0$ immediately after spiking.
- **Reset rule (subtract-threshold):** $V_m \leftarrow V_m - V_{th}$ after spiking (alternative).

The accompanying project uses **reset-to-zero**, which is the default in snntorch's `Leaky` neuron.

## Technical Explanation

### Discretizing the LIF ODE

To simulate the LIF neuron on a computer, we discretize time into steps of size $dt$. The standard approach is the Euler method applied to the homogeneous part of the ODE, combined with exact integration of the input.

Starting from:

$$\tau_{mem} \frac{dV_m}{dt} = -V_m + u(t)$$

The homogeneous solution (no input) is $V_m(t) = V_m(0) \exp(-t/\tau_{mem})$. Over one timestep $dt$, the membrane potential decays by a factor:

$$\beta = \exp\left(-\frac{dt}{\tau_{mem}}\right)$$

This is the **decay factor** or **membrane decay constant**. It is the single most important parameter in the discrete LIF model.

Adding the input term (treated as constant over the interval $[t, t+dt]$):

$$V_m[t+1] = \beta \cdot V_m[t] + (1 - \beta) \cdot u[t]$$

However, in the snntorch convention (and in most SNN deep learning papers), the input is not scaled by $(1-\beta)$; instead, the weighted input $W\mathbf{x}$ is added directly. This is equivalent to absorbing $(1-\beta)$ into the weight matrix $W$. The snntorch `Leaky` neuron implements:

$$V_m[t+1] = \beta \cdot V_m[t] + W \mathbf{s}_{pre}[t]$$

where $W \mathbf{s}_{pre}[t]$ is the weighted sum of presynaptic spikes at timestep $t$ (or, for the first layer, the weighted input current from the encoder).

### The Spike and Reset

After computing the updated membrane potential, the spike is determined by a Heaviside step function:

$$s[t] = \Theta(V_m[t] - V_{th}) = \begin{cases} 1 & \text{if } V_m[t] \geq V_{th} \\ 0 & \text{otherwise} \end{cases}$$

If $s[t] = 1$, the membrane is reset:

$$V_m[t] \leftarrow V_m[t] \cdot (1 - s[t]) = 0 \quad \text{(reset-to-zero)}$$

The full discrete update for one neuron at one timestep is therefore:

$$V_m[t] = \beta \cdot V_m[t-1] \cdot (1 - s[t-1]) + I_{in}[t]$$
$$s[t] = \Theta(V_m[t] - V_{th})$$

where $I_{in}[t] = W \mathbf{s}_{pre}[t]$ is the synaptic input. The term $(1 - s[t-1])$ implements the reset: if the neuron spiked at the previous step, its membrane is zeroed before the new input is added.

## Core Concepts

- **Beta ($\beta$):** The membrane decay factor per timestep. $\beta \in (0,1)$. High $\beta$ means slow decay (long memory); low $\beta$ means fast decay (short memory).
- **Membrane time constant ($\tau_{mem}$):** The continuous-time decay constant. $\tau_{mem} = -dt / \ln(\beta)$.
- **Threshold ($V_{th}$):** The membrane potential at which a spike is emitted. Default 1.0 in snntorch.
- **Reset-to-zero:** After spiking, $V_m$ is set to 0. This is the snntorch default.
- **Subtract-threshold reset:** After spiking, $V_m \leftarrow V_m - V_{th}$. Preserves "excess" potential.
- **Refractory period:** Not implemented in snntorch `Leaky` by default.

## What / Why / When / How

**What** is the LIF model? A two-equation model: a linear ODE for membrane potential dynamics, and a threshold-and-reset rule for spike generation.

**Why** use LIF instead of more complex models? The LIF model is the simplest model that captures integration and threshold firing. More complex models (Izhikevich, AdEx, Hodgkin-Huxley) reproduce more biological phenomena (bursting, adaptation, resonance) but are harder to train and simulate at scale. For deep learning applications, LIF is the standard choice.

**When** does the LIF model fail? When the phenomenon of interest depends on subthreshold oscillations, spike-frequency adaptation, bursting, or dendritic computation. For most classification tasks, these phenomena are not critical.

**How** is $\beta$ chosen? Either by setting $\tau_{mem}$ and $dt$ and computing $\beta = \exp(-dt/\tau_{mem})$, or by treating $\beta$ as a learnable parameter (as in some recent works). In the accompanying project, $\tau_{mem} = 20$ ms and $dt = 1$ ms, giving $\beta = \exp(-1/20) \approx 0.9512$.

## Advantages and Disadvantages

**Advantages of LIF:**
- Analytically tractable: the ODE has a closed-form solution.
- Computationally cheap: one multiply and one add per neuron per timestep.
- Well-studied: decades of theoretical and experimental literature.
- Compatible with surrogate gradient training (Chapter 4).

**Disadvantages of LIF:**
- No spike-frequency adaptation: a real neuron's firing rate decreases with sustained input; LIF fires at a constant rate.
- No subthreshold resonance: LIF cannot exhibit preferred input frequencies.
- Reset-to-zero discards information: the "excess" potential above threshold is lost.
- Fixed threshold: biological neurons have dynamic thresholds; LIF uses a fixed $V_{th}$.

## Historical Context

The integrate-and-fire model was introduced by Louis Lapicque in 1907, making it one of the oldest mathematical models in neuroscience. Lapicque used it to describe the electrical excitability of nerve and muscle tissue. The "leaky" version (with the RC circuit interpretation) became standard in the 1960s and 1970s as computational neuroscience developed.

The LIF model gained renewed prominence in the SNN deep learning era because it is the simplest model for which surrogate gradient training works well. Eshraghian et al. (2023) built the snntorch library around the LIF model precisely because of this tractability.

## Comparison

| Model | Complexity | Biological realism | Training ease | Phenomena captured |
|---|---|---|---|---|
| LIF | Very low | Low | High | Integration, threshold, reset |
| Izhikevich | Low | Medium | Medium | Bursting, adaptation, resonance |
| AdEx | Medium | Medium-high | Medium | Adaptation, subthreshold dynamics |
| Hodgkin-Huxley | High | High | Low | Full spike shape, channel dynamics |

For deep SNN training, LIF is the clear choice. For neuroscience modeling, more complex models are often needed.

## Visual Intuition

The following diagram shows the discrete-time evolution of a LIF neuron's membrane potential over 10 timesteps, with $\beta = 0.95$, $V_{th} = 1.0$, and a constant input of $I_{in} = 0.2$ per step.

```
Step:  0    1    2    3    4    5    6    7    8    9   10
Input: 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

V_m computation (no spike yet):
t=0: V = 0.00 * 0.95 + 0.20 = 0.200
t=1: V = 0.20 * 0.95 + 0.20 = 0.390
t=2: V = 0.39 * 0.95 + 0.20 = 0.571
t=3: V = 0.57 * 0.95 + 0.20 = 0.741
t=4: V = 0.74 * 0.95 + 0.20 = 0.904
t=5: V = 0.90 * 0.95 + 0.20 = 1.058  --> SPIKE (>= 1.0), reset to 0
t=6: V = 0.00 * 0.95 + 0.20 = 0.200  (after reset)
t=7: V = 0.20 * 0.95 + 0.20 = 0.390
...

V_m
1.1 |                    *
1.0 |. . . . . . . . . ./|. . . threshold
0.9 |               *  / |
0.8 |             /  \/  |
0.7 |           *        |
0.6 |         /          |
0.5 |       *            |
0.4 |     *              *
0.3 |   *                  *
0.2 | *                      *
0.0 |                          (reset)
    +--+--+--+--+--+--+--+--+--+--+-> t
    0  1  2  3  4  5  6  7  8  9  10

Spike train: 0 0 0 0 0 1 0 0 0 0 0
```

The membrane potential rises geometrically (not linearly) because each step's starting value is slightly lower than the previous step's ending value due to the leak. The steady-state firing rate is determined by the balance between input and leak.

## Mathematical Foundations

### Derivation of beta

Starting from the continuous LIF ODE with no input:

$$\tau_{mem} \frac{dV_m}{dt} = -V_m$$

This is a first-order linear ODE with solution:

$$V_m(t) = V_m(0) \exp\left(-\frac{t}{\tau_{mem}}\right)$$

Over one discrete timestep of duration $dt$:

$$V_m(t + dt) = V_m(t) \exp\left(-\frac{dt}{\tau_{mem}}\right) = \beta \cdot V_m(t)$$

Therefore:

$$\boxed{\beta = \exp\left(-\frac{dt}{\tau_{mem}}\right)}$$

### Project-specific values

In the accompanying research project:
- $\tau_{mem} = 20$ ms
- $dt = 1$ ms
- $\beta = \exp(-1/20) = \exp(-0.05) \approx 0.9512$
- $V_{th} = 1.0$ (snntorch default)
- $V_{reset} = 0$ (reset-to-zero, snntorch default)

### Steady-state firing rate

For a constant input $I_{in}$ and no reset, the membrane potential converges to a steady state $V_\infty = I_{in} / (1 - \beta)$. With $\beta = 0.9512$ and $I_{in} = 0.2$:

$$V_\infty = \frac{0.2}{1 - 0.9512} = \frac{0.2}{0.0488} \approx 4.1$$

Since $V_\infty > V_{th} = 1.0$, the neuron will fire repeatedly. The inter-spike interval (ISI) can be computed by finding the number of steps $k$ such that the membrane potential, starting from 0 after reset, first reaches $V_{th}$:

$$V_{th} = I_{in} \sum_{j=0}^{k-1} \beta^j = I_{in} \frac{1 - \beta^k}{1 - \beta}$$

Solving for $k$:

$$k = \frac{\ln\left(1 - V_{th}(1-\beta)/I_{in}\right)}{\ln(\beta)}$$

With the project values and $I_{in} = 0.2$: $k = \ln(1 - 1.0 \times 0.0488 / 0.2) / \ln(0.9512) = \ln(0.756) / (-0.05) \approx 5.6$ steps, consistent with the diagram above (spike at step 5).

### The full vectorized update

For a layer of $n$ LIF neurons receiving input from a layer of $m$ neurons:

$$\mathbf{V}[t] = \beta \cdot \mathbf{V}[t-1] \odot (1 - \mathbf{s}[t-1]) + W \mathbf{s}_{pre}[t]$$
$$\mathbf{s}[t] = \Theta(\mathbf{V}[t] - V_{th})$$

where $\odot$ is elementwise multiplication, $W \in \mathbb{R}^{n \times m}$ is the weight matrix, and $\mathbf{s}_{pre}[t] \in \{0,1\}^m$ is the presynaptic spike vector. This is exactly what snntorch computes in its forward pass.

## Implementation Notes

- **snntorch API:** `snn.Leaky(beta=0.9512, threshold=1.0, reset_mechanism='zero')`. The `beta` parameter is the decay factor, not the time constant. Always compute $\beta$ from $\tau_{mem}$ and $dt$ explicitly.
- **State initialization:** The membrane potential must be initialized to zero at the start of each new input sample. In snntorch, call `mem = lif.init_leaky()` before the timestep loop.
- **Gradient flow:** The Heaviside function $\Theta$ has zero gradient almost everywhere. This is the core training challenge addressed in Chapter 4.
- **Numerical stability:** $\beta$ must be in $(0,1)$. Values close to 1 (slow decay) can cause gradient vanishing over long sequences; values close to 0 (fast decay) cause gradient explosion in the backward pass through time.
- **Batch processing:** In practice, the membrane potential tensor has shape `[batch_size, n_neurons]`. The timestep loop iterates $T$ times, updating this tensor at each step.
- **Learnable beta:** snntorch supports `learn_beta=True`, which treats $\beta$ as a learnable parameter. This is not used in the accompanying project (fixed $\beta$).

## Examples

**Simple Python example (snntorch):**

```python
import torch
import snntorch as snn

# LIF neuron with project parameters
beta = 0.9512  # exp(-1/20)
lif = snn.Leaky(beta=beta, threshold=1.0, reset_mechanism='zero')

# Initialize membrane potential
mem = lif.init_leaky()  # shape: [batch_size, n_neurons]

# Simulate 25 timesteps
spk_rec = []
mem_rec = []
for t in range(25):
    # cur: weighted input at this timestep, shape [batch_size, n_neurons]
    spk, mem = lif(cur[t], mem)
    spk_rec.append(spk)
    mem_rec.append(mem)

spk_rec = torch.stack(spk_rec)  # shape: [25, batch_size, n_neurons]
```

**Real-world example:** In the project's two-hidden-layer SNN, each hidden layer is a `snn.Leaky` neuron with $\beta = 0.9512$. The first layer receives the encoded input (a current proportional to the pixel value, repeated for 25 timesteps). The second layer receives the spike output of the first layer. The output layer is also a `snn.Leaky` neuron; the class prediction is made by summing the output spikes over all 25 timesteps and taking the argmax.

## Current Research

Active research directions in LIF model development include: (1) **learnable time constants**, where $\beta$ (or equivalently $\tau_{mem}$) is learned per neuron, allowing the network to adapt its temporal dynamics to the task; (2) **adaptive threshold mechanisms**, where $V_{th}$ increases after each spike and decays back, implementing spike-frequency adaptation; (3) **dendritic LIF models**, where the neuron has multiple compartments with different dynamics; (4) **stochastic LIF models**, where the threshold is replaced by a probabilistic firing rule, enabling gradient estimation without surrogate gradients.

The question of whether reset-to-zero or subtract-threshold reset is better for training is still debated. Subtract-threshold preserves information about the excess potential, which can improve accuracy, but reset-to-zero is simpler and more common in practice.

## References

- Lapicque, L. (1907). Recherches quantitatives sur l'excitation electrique des nerfs traitee comme une polarisation. *Journal de Physiologie et de Pathologie Generale*, 9, 620--635.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press. (Chapter 4 covers LIF in detail.)
- Eshraghian, J. K., Ward, M., Neftci, E., Wang, X., Liao, B., Shrestha, A., Linares-Barranco, B., & Perez-Nieves, N. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016--1054.

---
<a name="chapter-3"></a>
# Chapter 3: Encoding and Time in SNNs

## Overview

A spiking neural network operates over time. Its neurons communicate via discrete spike events, and the information content of those spikes depends critically on how the input is represented as a spike train. This is the **encoding problem**: given a static input (say, a 28x28 pixel image), how do you convert it into a temporal sequence of spikes that a network of LIF neurons can process? The answer is not unique, and the choice of encoding scheme has profound consequences for the network's behavior, its energy consumption, and its ability to learn. This chapter surveys the main encoding strategies, explains the role of the number of timesteps $T$, and describes in detail the rate coding scheme used in the accompanying research project.

## Fundamental Theory

In biology, sensory information is encoded in the activity of populations of neurons. Two broad coding schemes have been proposed and debated for decades:

**Rate coding:** The information is carried by the average firing rate of a neuron over some time window. A neuron that fires 100 spikes per second encodes a stronger stimulus than one that fires 10 spikes per second. Rate coding is robust to noise and is well-supported by experimental evidence in many sensory systems.

**Temporal coding:** The information is carried by the precise timing of individual spikes, not just their average rate. A neuron that fires at exactly 10 ms after stimulus onset encodes different information from one that fires at 20 ms. Temporal coding is more information-efficient (more bits per spike) but more sensitive to noise and harder to decode.

These two extremes define a spectrum. Most biological systems likely use a mixture of both, with rate coding dominating in some areas and temporal coding in others.

## Technical Explanation

### Rate Coding

In rate coding for SNNs, the input value $x \in [0,1]$ (e.g., a normalized pixel intensity) is converted to a spike train by treating $x$ as the probability of firing at each timestep. At each of the $T$ timesteps, a Bernoulli random variable with parameter $x$ determines whether a spike is emitted:

$$s[t] \sim \text{Bernoulli}(x) \quad \text{for } t = 1, \ldots, T$$

Over $T$ timesteps, the expected number of spikes is $T \cdot x$, so the average firing rate is $x$ spikes per timestep. A pixel with intensity 1.0 fires every timestep; a pixel with intensity 0.0 never fires; a pixel with intensity 0.5 fires on average half the time.

This is the **Poisson rate coding** or **stochastic rate coding** scheme. It is the most common encoding in SNN deep learning papers.

### Direct / Current Injection (the project's scheme)

The accompanying research project uses a simpler variant: **direct encoding** (also called current injection or rate coding without stochasticity). Instead of sampling a Bernoulli variable, the same input value $x$ is presented as a constant current to the first layer at every timestep:

$$I_{in}[t] = x \quad \text{for all } t = 1, \ldots, T$$

This is equivalent to saying: the static image is "shown" to the network at every one of the $T = 25$ timesteps. The first-layer LIF neurons receive this constant current, integrate it over time, and produce a spike train whose rate is determined by the input intensity and the LIF dynamics (threshold, beta).

This scheme is deterministic (no randomness in the encoding), simpler to implement, and avoids the variance introduced by stochastic sampling. It is the default in many snntorch tutorials and is used in the project because it is straightforward and reproducible.

### Latency / Temporal Coding

In latency coding, the input value $x$ determines the **time** at which a neuron fires, not the rate. A strong input (high $x$) causes an early spike; a weak input causes a late spike or no spike. The information is in the first spike time.

Formally, if the input current is proportional to $x$ and the LIF neuron starts from rest, the time to first spike $t_{spike}$ is:

$$t_{spike} = \frac{\tau_{mem}}{1} \ln\left(\frac{I_{in}}{I_{in} - V_{th}(1-\beta)}\right)$$

(in continuous time). Neurons with higher input fire earlier. This scheme is maximally sparse (at most one spike per neuron per input) and potentially very fast, but it requires precise timing and is harder to train.

### Population Coding

Population coding uses a bank of neurons with different tuning curves to represent a scalar input. Each neuron fires most strongly when the input is near its preferred value. This is common in neuroscience models of sensory cortex but less common in SNN deep learning.

## Core Concepts

- **Timesteps $T$:** The number of discrete time steps over which the SNN is simulated for each input. In the project, $T = 25$.
- **Spike train:** A binary sequence $\{s[1], s[2], \ldots, s[T]\} \in \{0,1\}^T$ representing a neuron's output over $T$ timesteps.
- **Rate coding:** Input value encoded as firing rate (spikes per timestep).
- **Direct encoding:** The same input value is presented at every timestep as a constant current.
- **Latency coding:** Input value encoded as time to first spike.
- **Temporal coding:** General term for schemes where spike timing carries information.

## What / Why / When / How

**What** is the role of $T$? The number of timesteps controls the temporal resolution of the network. More timesteps allow more spikes per neuron, enabling finer rate coding and more temporal dynamics. Fewer timesteps reduce computation but limit the information that can be encoded.

**Why** does $T$ matter for training? The SNN is trained with backpropagation through time (BPTT, Chapter 4), which unrolls the network over $T$ steps. Larger $T$ means longer unrolled graphs, more memory, and more computation. There is a tradeoff between temporal resolution and training efficiency.

**When** should you use rate coding vs. temporal coding? Rate coding is simpler, more robust, and easier to train. Temporal coding is more energy-efficient (fewer spikes) and potentially more powerful, but harder to train and more sensitive to noise. For classification tasks on static images, rate coding (or direct encoding) is the standard choice.

**How** does direct encoding work in snntorch? The input tensor has shape `[T, batch_size, n_input]`. For direct encoding, the same input is simply repeated $T$ times along the first dimension. In the project, this is done by `input.unsqueeze(0).repeat(T, 1, 1)` or equivalently by passing the same input to the first layer at each timestep in the loop.

## Advantages and Disadvantages

**Rate coding / direct encoding:**
- Advantages: Simple, deterministic, easy to implement, robust to noise, well-understood.
- Disadvantages: Requires many timesteps for accurate rate estimation; not energy-efficient (many spikes); discards temporal information.

**Latency / temporal coding:**
- Advantages: Very sparse (few spikes), fast (information in first spike), potentially more powerful.
- Disadvantages: Hard to train, sensitive to noise, requires precise timing, less mature tooling.

**Stochastic rate coding (Poisson):**
- Advantages: Biologically plausible, naturally handles uncertainty.
- Disadvantages: Introduces variance (different runs give different results), requires averaging over multiple trials for stable gradients.

## Historical Context

The debate between rate coding and temporal coding in neuroscience dates to the 1920s (Adrian's work on sensory nerve firing rates) and has never been fully resolved. In the SNN deep learning community, rate coding dominated early work because it is the easiest to combine with gradient-based training. Temporal coding has gained interest more recently as researchers seek to reduce the number of timesteps (and thus the energy cost) of SNN inference.

The use of $T = 25$ timesteps in the project is a common choice in the SNN literature for MNIST-scale tasks. It provides enough temporal resolution for rate coding to work well while keeping training computationally feasible.

## Comparison

| Encoding | Spikes per neuron | Information carrier | Training ease | Energy efficiency |
|---|---|---|---|---|
| Direct (project) | Up to $T$ | Rate (via LIF dynamics) | High | Moderate |
| Stochastic rate | Up to $T$ (random) | Rate | High | Moderate |
| Latency | At most 1 | Time to first spike | Medium | High |
| Phase coding | Variable | Spike phase | Low | High |
| Population | Variable | Population rate | Medium | Low |

## Visual Intuition

The following diagram shows how a single pixel with intensity 0.6 is encoded under three different schemes over $T = 10$ timesteps.

```
Pixel intensity: 0.6

Direct encoding (same value every step):
Input current: [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
(LIF neuron integrates this and fires when V_m >= threshold)

Stochastic rate encoding (Bernoulli(0.6)):
Spike train (one sample): [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
Average rate: 6/10 = 0.6 (matches input)

Latency encoding (fires at step proportional to 1/x):
Spike train: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
(fires at step 5 out of 10, since 0.6 is moderately strong)

Time axis:
t:  1  2  3  4  5  6  7  8  9  10
    |  |  |  |  |  |  |  |  |  |
Direct: the LIF neuron sees 0.6 at every step
Stoch:  1  0  1  1  0  1  0  1  1  0   (random)
Latency:0  0  0  0  1  0  0  0  0  0   (one spike, early = strong)
```

The key insight: direct encoding is the simplest -- just repeat the input. The LIF neuron's dynamics then determine the actual spike train. The threshold controls how many spikes are produced: a higher threshold means the neuron needs more accumulated input before firing, resulting in fewer spikes.

## Mathematical Foundations

### Direct encoding and LIF firing rate

With direct encoding (constant input $I_{in} = x$ at every step) and a LIF neuron with decay $\beta$ and threshold $V_{th}$, the steady-state firing rate $r$ (spikes per timestep) can be derived.

After a spike (reset to 0), the membrane potential at step $k$ is:

$$V_m[k] = x \sum_{j=0}^{k-1} \beta^j = x \cdot \frac{1 - \beta^k}{1 - \beta}$$

The neuron fires at the first step $k^*$ where $V_m[k^*] \geq V_{th}$:

$$k^* = \left\lceil \frac{\ln(1 - V_{th}(1-\beta)/x)}{\ln(\beta)} \right\rceil$$

The firing rate is $r = 1/k^*$ spikes per timestep. For the project parameters ($\beta = 0.9512$, $V_{th} = 1.0$):

- $x = 0.2$: $k^* \approx 6$, $r \approx 0.17$ spikes/step
- $x = 0.5$: $k^* \approx 3$, $r \approx 0.33$ spikes/step
- $x = 1.0$: $k^* \approx 1$, $r = 1.0$ spikes/step (fires every step)

This shows that the LIF neuron with direct encoding implements an approximately linear rate code for moderate input values.

### Information content

A spike train of length $T$ with firing rate $r$ carries at most $H = -r\log_2 r - (1-r)\log_2(1-r)$ bits per timestep (binary entropy). At $r = 0.5$, this is maximized at 1 bit/step. At $r = 0.1$ (sparse), it is about 0.47 bits/step. Sparsity reduces information per neuron but reduces energy cost.

### The role of T in rate estimation

For stochastic rate coding, the variance of the estimated rate from $T$ timesteps is $r(1-r)/T$. With $T = 25$ and $r = 0.5$, the standard deviation is $\sqrt{0.25/25} = 0.1$. This is the fundamental noise floor of rate coding with finite $T$.

## Implementation Notes

- **Direct encoding in snntorch:** Simply pass the same input tensor to the first layer at each timestep. No special encoding module is needed.
- **Stochastic encoding in snntorch:** Use `spikegen.rate(data, num_steps=T)` which samples Bernoulli spikes. This introduces randomness; fix the random seed for reproducibility.
- **Input normalization:** Inputs should be in $[0,1]$ for rate coding to make sense. For MNIST, pixel values are divided by 255.
- **T selection:** $T = 25$ is a common choice for MNIST. For more complex datasets (CIFAR-10), $T = 50$--$100$ is typical. Larger $T$ improves accuracy but increases training time quadratically (due to BPTT).
- **Memory layout:** In snntorch, the time dimension is the first dimension of the tensor: shape `[T, batch_size, n_neurons]`. This is different from PyTorch's default RNN convention `[seq_len, batch, features]` but is the same convention.

## Examples

**Simple example:** A single pixel with value 0.8 is encoded with direct encoding and $T = 5$. The input to the first LIF layer at each step is 0.8. With $\beta = 0.9512$ and $V_{th} = 1.0$:

```
t=1: V = 0.00 * 0.9512 + 0.8 = 0.800
t=2: V = 0.80 * 0.9512 + 0.8 = 1.561 --> SPIKE, reset to 0
t=3: V = 0.00 * 0.9512 + 0.8 = 0.800
t=4: V = 0.80 * 0.9512 + 0.8 = 1.561 --> SPIKE, reset to 0
t=5: V = 0.00 * 0.9512 + 0.8 = 0.800

Spike train: [0, 1, 0, 1, 0]  (fires every 2 steps)
```

**Real-world example:** In the project, a 28x28 MNIST image (784 pixels) is flattened to a vector of 784 values in $[0,1]$. This vector is passed as the input current to the first hidden layer (256 LIF neurons) at each of the 25 timesteps. The first hidden layer produces a spike train of shape `[25, batch_size, 256]`. The second hidden layer (256 LIF neurons) receives this spike train and produces another spike train of shape `[25, batch_size, 256]`. The output layer (10 LIF neurons) receives the second hidden layer's spikes and produces output spikes; the class prediction is `argmax(sum over T of output spikes)`.

## Current Research

Current research on SNN encoding includes: (1) **learned encoders**, where a small ANN or convolutional layer converts the input to a spike train, allowing the encoding to be optimized end-to-end; (2) **event-based sensors**, where the input is already a spike train from a dynamic vision sensor (DVS camera), making encoding trivial; (3) **few-timestep SNNs**, where $T$ is reduced to 1--4 steps by using more sophisticated encoding (e.g., direct current injection with a learned scaling factor); (4) **temporal coding training**, where the loss function is defined in terms of first spike times rather than spike counts.

The trend is toward fewer timesteps (for energy efficiency) and more sophisticated encoding (to compensate for the reduced temporal resolution).

## References

- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press. (Chapter 1 covers rate vs. temporal coding.)
- Eshraghian, J. K., Ward, M., Neftci, E., Wang, X., Liao, B., Shrestha, A., Linares-Barranco, B., & Perez-Nieves, N. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016--1054.
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659--1671.

---

<a name="chapter-4"></a>
# Chapter 4: Training SNNs with Surrogate Gradients

## Overview

Training a spiking neural network with gradient descent requires computing how the loss changes with respect to every weight in the network. In a standard ANN, this is accomplished by backpropagation: the chain rule is applied layer by layer, and the gradient flows smoothly because every activation function (ReLU, sigmoid, tanh) is differentiable almost everywhere. In an SNN, the activation function is the Heaviside step function: the neuron either fires (output 1) or does not (output 0), with an instantaneous transition at the threshold. The derivative of the Heaviside function is the Dirac delta -- zero everywhere except at the threshold, where it is infinite. This means that standard backpropagation produces zero gradients almost everywhere and is undefined at the threshold. The network cannot learn.

The surrogate gradient method, systematically developed and analyzed by Neftci, Mostafa, and Zenke (2019), resolves this impasse with a conceptually simple but powerful idea: use the true spike function in the forward pass (so the network actually spikes), but substitute a smooth, differentiable surrogate function for the spike function's derivative in the backward pass. The surrogate provides a useful gradient signal even though it does not correspond to the true derivative of the forward computation. This chapter explains why this works, what surrogates are used in practice, how backpropagation through time (BPTT) extends this to the temporal dimension of SNNs, and how the accompanying research project implements all of this using snntorch.

## Fundamental Theory

### The Non-Differentiability Problem

The spike function of a LIF neuron is the Heaviside step function applied to the membrane potential relative to the threshold:

$$s[t] = \Theta(V_m[t] - V_{th}) = \begin{cases} 1 & \text{if } V_m[t] \geq V_{th} \\ 0 & \text{otherwise} \end{cases}$$

To train the network with gradient descent, we need the gradient of the loss $\mathcal{L}$ with respect to the weights $W$. By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial s} \cdot \frac{\partial s}{\partial V_m} \cdot \frac{\partial V_m}{\partial W}$$

The problematic term is $\frac{\partial s}{\partial V_m} = \frac{d\Theta}{dV_m}$. The Heaviside function has derivative zero everywhere except at $V_m = V_{th}$, where it is the Dirac delta (infinite). In practice, this means:

- For neurons that do not spike: $\frac{\partial s}{\partial V_m} = 0$, so the gradient is zero. No learning signal propagates through non-spiking neurons.
- For neurons that spike: the gradient is technically infinite, which causes numerical instability.

This is the **dead neuron problem** for SNNs, analogous to the dying ReLU problem but more severe because it affects all neurons, not just those with negative pre-activations.

### The Surrogate Gradient Solution

The surrogate gradient method (Neftci, Mostafa & Zenke, 2019) decouples the forward and backward passes:

- **Forward pass:** Use the true Heaviside function. The network produces real binary spikes.
- **Backward pass:** Replace $\frac{d\Theta}{dV_m}$ with a smooth surrogate function $\sigma'(V_m - V_{th})$ that is nonzero in a neighborhood of the threshold.

This is implemented using a custom autograd function in PyTorch. The forward method computes the Heaviside spike; the backward method returns the surrogate derivative instead of the true (zero) derivative. The network "thinks" it is differentiating through a smooth function, even though the forward pass used a step function.

This approach is sometimes called the "straight-through estimator" (STE) when the surrogate is the identity function, a technique originally proposed for training binary neural networks. The surrogate gradient method generalizes this to smooth, threshold-centered surrogates that are better suited to the SNN setting.

## Technical Explanation

### Common Surrogate Functions

Several surrogate functions have been proposed and studied. The two most common in the SNN deep learning literature are:

**Fast-sigmoid (used in the accompanying project):**

$$\sigma(x) = \frac{x}{1 + k|x|}$$

where $x = V_m - V_{th}$ and $k > 0$ is a slope parameter (default $k = 25$ in snntorch). The derivative used as the surrogate gradient is:

$$\sigma'(x) = \frac{1}{(1 + k|x|)^2}$$

This function is peaked at $x = 0$ (i.e., at the threshold), falls off as $1/x^2$ away from the threshold, and is always positive. It is computationally cheap (no exponentials or trigonometric functions).

**Arctangent (atan):**

$$\sigma(x) = \frac{1}{\pi} \arctan(\pi x / 2) + \frac{1}{2}$$

with surrogate derivative:

$$\sigma'(x) = \frac{1}{1 + (\pi x / 2)^2}$$

The atan surrogate has heavier tails than fast-sigmoid, meaning it provides gradient signal further from the threshold. This can help in networks where the membrane potential rarely approaches the threshold.

Both surrogates are symmetric around the threshold, peak at the threshold, and decay to zero far from the threshold. The choice between them is largely empirical; fast-sigmoid is the default in snntorch and is used in the project.

### Backpropagation Through Time (BPTT) for SNNs

An SNN simulated over $T$ timesteps is mathematically equivalent to a recurrent neural network (RNN) unrolled over $T$ steps. The membrane potential at step $t$ depends on the membrane potential at step $t-1$ (through the decay term $\beta V_m[t-1]$) and the input at step $t$. This temporal dependency creates a computational graph that spans all $T$ timesteps.

BPTT for SNNs works as follows:

1. **Forward pass:** Run the SNN for all $T$ timesteps, recording the membrane potential and spike output at each step for each layer.
2. **Compute loss:** The loss is computed from the output spikes (e.g., cross-entropy on the summed output spike counts).
3. **Backward pass:** Backpropagate the loss gradient through the unrolled computational graph, using the surrogate gradient wherever a spike function is encountered.

The gradient of the loss with respect to the membrane potential at timestep $t$ in layer $l$ receives contributions from two sources: (a) the direct contribution through the spike at timestep $t$, and (b) the contribution propagated backward through time from timestep $t+1$ via the membrane decay term $\beta$.

This is identical in structure to BPTT for vanilla RNNs, with the surrogate gradient playing the role of the activation function's derivative. The same challenges apply: vanishing gradients (if $\beta^T$ is small) and exploding gradients (if $\beta^T$ is large). With $\beta = 0.9512$ and $T = 25$, $\beta^{25} \approx 0.28$, which is in a reasonable range -- gradients decay by a factor of about 4 over the full sequence, which is manageable.

## Core Concepts

- **Heaviside step function:** The true spike function; derivative is zero almost everywhere.
- **Surrogate gradient:** A smooth function substituted for the Heaviside derivative in the backward pass only.
- **Fast-sigmoid surrogate:** $\sigma'(x) = 1/(1+k|x|)^2$; the surrogate used in the project (snntorch default, $k=25$).
- **Arctangent surrogate:** $\sigma'(x) = 1/(1+(\pi x/2)^2)$; an alternative with heavier tails.
- **BPTT:** Backpropagation through time; unrolls the SNN over $T$ steps and backpropagates through the full temporal graph.
- **Straight-through estimator:** The special case where the surrogate derivative is 1 (identity); the simplest possible surrogate.
- **Dead neuron problem:** When a neuron's membrane potential is far from threshold, the surrogate gradient is near zero and the neuron receives no learning signal.

## What / Why / When / How

**What** is a surrogate gradient? It is a smooth function that approximates the derivative of the spike function in a neighborhood of the threshold. It is used only in the backward pass; the forward pass uses the true Heaviside function.

**Why** does this work? The surrogate gradient is a biased estimator of the true gradient (which is zero). However, the bias is useful: it provides a gradient signal that points in the direction of increasing spike probability, which is correlated with the direction of decreasing loss. Empirically, networks trained with surrogate gradients converge to good solutions on a wide range of tasks.

**When** should you use surrogate gradients? Whenever you want to train an SNN with gradient descent. There is currently no practical alternative for deep SNNs on standard tasks. Biologically inspired local rules (STDP, e-prop) are alternatives but are harder to scale and typically achieve lower accuracy.

**How** is the surrogate gradient implemented in snntorch? snntorch uses PyTorch's custom autograd mechanism. The spike function is wrapped in a class that overrides the backward method to return the surrogate derivative. The user selects the surrogate via `spike_grad = snn.surrogate.fast_sigmoid()` and passes it to the neuron constructor: `lif = snn.Leaky(beta=beta, spike_grad=spike_grad)`.

## Advantages and Disadvantages

**Advantages of surrogate gradients:**
- Enables end-to-end gradient-based training of deep SNNs.
- Compatible with standard optimizers (Adam, SGD) and loss functions (cross-entropy).
- Simple to implement in PyTorch via custom autograd.
- Empirically effective across a wide range of tasks and architectures.

**Disadvantages:**
- The gradient is biased: it does not correspond to the true derivative of the forward computation.
- Hyperparameter sensitivity: the slope parameter $k$ of the surrogate affects training dynamics significantly.
- BPTT memory cost: storing activations for all $T$ timesteps requires $T$ times more memory than a single forward pass.
- Temporal credit assignment: gradients must propagate through $T$ steps, which can be slow and numerically challenging for large $T$.

## Historical Context

The surrogate gradient idea has roots in the training of binary neural networks (Hinton's "straight-through estimator," circa 2012, in unpublished lecture notes) and in early SNN training work by Bohte, Kok, and La Poutre (2000) on SpikeProp. The systematic treatment of surrogate gradients for SNNs, including a theoretical analysis of why they work and a comparison of different surrogate functions, was provided by Neftci, Mostafa, and Zenke (2019) in IEEE Signal Processing Magazine. This paper is the canonical reference for the method and is the basis for the snntorch implementation.

Prior to surrogate gradients, SNNs were trained primarily with: (1) ANN-to-SNN conversion (train an ANN, then convert weights to an SNN by interpreting activations as firing rates); (2) spike-timing-dependent plasticity (STDP, a local Hebbian rule); (3) evolutionary algorithms. Surrogate gradients made it possible to train SNNs from scratch with the same tools used for ANNs, dramatically improving accuracy and scalability.

## Comparison

| Training method | Gradient | Accuracy | Scalability | Biological plausibility |
|---|---|---|---|---|
| Surrogate gradient (BPTT) | Biased smooth approx. | High | High | Low |
| ANN-to-SNN conversion | N/A (no SNN training) | High | High | Very low |
| STDP | Local, no backprop | Low | Low | High |
| SpikeProp | Exact (single spike) | Low | Low | Low |
| e-prop | Local approximation | Medium | Medium | Medium |

Surrogate gradient BPTT is the dominant method for deep SNN training as of 2024.

## Visual Intuition

The following diagram illustrates the forward/backward pass split at a single LIF neuron.

```
FORWARD PASS (true Heaviside):

V_m = 0.8  -->  spike function  -->  s = 0   (below threshold 1.0)
V_m = 1.2  -->  spike function  -->  s = 1   (above threshold 1.0)

The step is sharp: no gradient flows through this in the true backward pass.

BACKWARD PASS (surrogate gradient, fast-sigmoid, k=25):

x = V_m - V_th

x = -0.2  -->  sigma'(-0.2) = 1/(1+25*0.2)^2 = 1/36 = 0.028
x =  0.0  -->  sigma'( 0.0) = 1/(1+25*0.0)^2 = 1/1  = 1.000  (peak)
x = +0.2  -->  sigma'(+0.2) = 1/(1+25*0.2)^2 = 1/36 = 0.028
x = -1.0  -->  sigma'(-1.0) = 1/(1+25*1.0)^2 = 1/676 = 0.0015

Surrogate gradient profile:
sigma'
1.0 |         *
    |        / \
    |       /   \
0.5 |      /     \
    |     /       \
0.1 |    /         \
    |   /           \
0.0 |--/             \--
    +--+--+--+--+--+--+-> x = V_m - V_th
      -2 -1  0  1  2

The gradient is largest at the threshold and decays away from it.
Neurons far from threshold receive little gradient (the dead neuron problem).
```

The key intuition: the surrogate gradient acts as a "soft" version of the spike function's derivative. It tells the optimizer: "if you increase the membrane potential of this neuron, it becomes more likely to spike, and here is how much more likely." This is a useful signal even though it is not the exact derivative of the Heaviside function.

## Mathematical Foundations

### The surrogate gradient chain rule

Let $\mathcal{L}$ be the loss, $s^l[t]$ the spike of neuron $i$ in layer $l$ at timestep $t$, and $V^l[t]$ the corresponding membrane potential. The surrogate gradient method computes:

$$\frac{\partial \mathcal{L}}{\partial V^l[t]} = \frac{\partial \mathcal{L}}{\partial s^l[t]} \cdot \sigma'(V^l[t] - V_{th})$$

where $\sigma'$ is the surrogate derivative (e.g., fast-sigmoid derivative), replacing the true $\frac{d\Theta}{dV}$.

### BPTT gradient through membrane potential

The membrane potential at step $t$ depends on the membrane potential at step $t-1$ (after reset):

$$V^l[t] = \beta \cdot V^l[t-1] \cdot (1 - s^l[t-1]) + W^l s^{l-1}[t]$$

The gradient of the loss with respect to $V^l[t-1]$ receives a contribution from $V^l[t]$ via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial V^l[t-1]} \ni \frac{\partial \mathcal{L}}{\partial V^l[t]} \cdot \beta \cdot (1 - s^l[t-1]) - \frac{\partial \mathcal{L}}{\partial V^l[t]} \cdot \beta \cdot V^l[t-1] \cdot \sigma'(V^l[t-1] - V_{th})$$

The first term propagates the gradient backward through the decay; the second term accounts for the effect of the previous spike on the reset. Summing over all timesteps gives the full BPTT gradient.

### Fast-sigmoid surrogate: explicit formula

The fast-sigmoid surrogate function used in snntorch is:

$$\sigma(x; k) = \frac{x}{1 + k|x|}$$

Its derivative (the surrogate gradient) is:

$$\sigma'(x; k) = \frac{1}{(1 + k|x|)^2}$$

With $k = 25$ (snntorch default):
- At $x = 0$ (threshold): $\sigma'(0) = 1$
- At $x = \pm 0.1$: $\sigma'(\pm 0.1) = 1/(1+2.5)^2 = 1/12.25 \approx 0.082$
- At $x = \pm 0.5$: $\sigma'(\pm 0.5) = 1/(1+12.5)^2 = 1/182.25 \approx 0.0055$

The gradient falls off rapidly, concentrating the learning signal near the threshold.

### Gradient vanishing over time

The gradient of the loss with respect to the membrane potential at timestep $t=0$ (the first step) involves a product of $T$ terms, each involving $\beta$:

$$\left|\frac{\partial \mathcal{L}}{\partial V^l[0]}\right| \leq \beta^T \cdot \left|\frac{\partial \mathcal{L}}{\partial V^l[T]}\right| \cdot \prod_{t=1}^{T} |\sigma'(V^l[t] - V_{th})|$$

With $\beta = 0.9512$ and $T = 25$: $\beta^{25} = \exp(-25/20) = \exp(-1.25) \approx 0.287$. The gradient is attenuated by a factor of about 3.5 over the full sequence, which is acceptable. For larger $T$ or smaller $\beta$, gradient vanishing becomes more severe.

## Implementation Notes

- **snntorch API:** Pass `spike_grad = snn.surrogate.fast_sigmoid()` to the `Leaky` constructor. The slope parameter $k$ can be set via `snn.surrogate.fast_sigmoid(slope=25)`.
- **Loss function:** The standard choice is cross-entropy on the summed output spike counts: `loss = F.cross_entropy(spk_out.sum(0), targets)`. Summing over the time dimension converts the spike train to a rate-coded output.
- **Optimizer:** Adam with learning rate $10^{-3}$ is the project's choice and is standard for SNN training.
- **Memory:** BPTT requires storing all intermediate activations for $T$ steps. For $T=25$ and a batch of 256, this is 25x the memory of a single forward pass. Gradient checkpointing can reduce this at the cost of recomputation.
- **Truncated BPTT:** For very large $T$, gradients can be truncated to only propagate back $k < T$ steps. This reduces memory and computation but may hurt accuracy.
- **Detaching membrane state between samples:** The membrane potential must be detached from the computational graph between different input samples (i.e., `mem = mem.detach()` at the start of each new sample). Failing to do this causes gradients to flow across sample boundaries, which is incorrect.

## Examples

**Simple example -- custom autograd for surrogate gradient:**

The following pseudocode illustrates how snntorch implements the surrogate gradient using PyTorch's autograd:

    class SpikeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, membrane, threshold):
            ctx.save_for_backward(membrane - threshold)
            return (membrane >= threshold).float()

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            # fast-sigmoid surrogate derivative
            surrogate_grad = 1.0 / (1.0 + 25.0 * x.abs()) ** 2
            return grad_output * surrogate_grad, None

    spike = SpikeFunction.apply(membrane, threshold)

**Real-world example (the project):** The project's SNN is defined with:

    spike_grad = snn.surrogate.fast_sigmoid(slope=25)
    lif1 = snn.Leaky(beta=0.9512, spike_grad=spike_grad, reset_mechanism='zero')
    lif2 = snn.Leaky(beta=0.9512, spike_grad=spike_grad, reset_mechanism='zero')

During training, the forward pass runs for $T = 25$ timesteps, accumulating spikes. The loss is cross-entropy on the summed output spikes. Adam (lr=1e-3) updates the weights via BPTT. The surrogate gradient flows through both hidden LIF layers and the output LIF layer at every timestep.

## Current Research

Active research on surrogate gradients includes: (1) **adaptive surrogates**, where the slope parameter $k$ is adjusted during training (e.g., annealed from a small value to a large value, sharpening the surrogate as training progresses); (2) **learned surrogates**, where the surrogate function itself is parameterized and learned; (3) **online learning with surrogate gradients**, where gradients are computed and applied at each timestep rather than after the full sequence (reducing memory cost); (4) **theoretical analysis**, characterizing when and why surrogate gradients converge and what the bias-variance tradeoff looks like for different surrogate choices.

The relationship between surrogate gradient training and biologically plausible learning rules (e-prop, online BPTT) is an active area of research, with the goal of finding methods that are both effective and implementable in neuromorphic hardware without off-chip gradient computation.

## References

- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51--63.
- Eshraghian, J. K., Ward, M., Neftci, E., Wang, X., Liao, B., Shrestha, A., Linares-Barranco, B., & Perez-Nieves, N. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016--1054.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press.

---

<a name="chapter-5"></a>
# Chapter 5: Spike Sparsity and Energy Efficiency in SNNs

## Overview

One of the most frequently cited advantages of spiking neural networks is their potential for energy efficiency. The argument is intuitive: if most neurons are silent most of the time (i.e., the spike trains are sparse), then most synaptic connections carry no signal at most timesteps, and a neuromorphic hardware implementation need only perform computation when a spike actually arrives. This event-driven computation model stands in contrast to the dense, always-on matrix multiplications of conventional ANNs running on GPUs. The energy savings can, in principle, be substantial.

This chapter develops the concept of spike sparsity precisely, explains the energy model that connects sparsity to efficiency, introduces the specific sparsity metric and control mechanism used in the accompanying research project, and is careful to distinguish between computational estimates of energy and actual hardware measurements. It also connects sparsity to the research question at the heart of the project: whether higher spike sparsity reduces catastrophic forgetting by reducing the overlap between the neural representations used for different tasks.

A critical caveat runs through this entire chapter: the energy proxy used in the project is a computational estimate based on spike counts, not a measurement of actual power consumption on any hardware. This distinction matters enormously and will be stated explicitly wherever relevant.

## Fundamental Theory

### What is Spike Sparsity?

Spike sparsity refers to the fraction of neurons in a network that are active (i.e., that fire at least one spike) during the processing of a given input. In a fully dense network, every neuron fires at every timestep; the sparsity is zero. In a fully sparse network, no neuron fires; the sparsity is one (or the activity is zero). In practice, useful SNNs operate somewhere in between.

There are several ways to quantify sparsity:

1. **Spike rate per neuron:** The average number of spikes emitted by a single neuron over $T$ timesteps, divided by $T$. This is the firing rate in spikes per timestep.

2. **Active neuron fraction:** The fraction of neurons that emit at least one spike over the $T$ timesteps. A neuron is "active" if it fires at least once; it is "silent" if it fires zero times.

3. **Total spike count:** The total number of spikes emitted by all neurons in a layer (or the whole network) over all $T$ timesteps and all samples in a batch.

The accompanying project uses the **active neuron fraction** as its primary sparsity metric, defined precisely in the Mathematical Foundations section below.

### Why Does Sparsity Matter for Energy?

On conventional hardware (CPUs, GPUs), every neuron's activation is computed at every forward pass, regardless of whether it is zero or not. The dominant operation is the multiply-accumulate (MAC): for each output neuron, compute the dot product of the weight vector with the input vector. The number of MACs is fixed by the architecture and does not depend on the input.

On neuromorphic hardware (Intel Loihi, IBM TrueNorth, BrainScaleS), computation is event-driven: a synaptic operation (SynOp) occurs only when a presynaptic spike arrives. If a neuron does not spike, none of its downstream synapses are activated, and no energy is consumed for those connections. The energy cost is therefore proportional to the number of spikes, not the number of neurons or connections.

This is the fundamental energy advantage of SNNs: if the spike trains are sparse, the number of SynOps is much smaller than the number of MACs in an equivalent ANN, and the energy consumption is correspondingly lower.

Roy, Jaiswal, and Panda (2019) provide a detailed analysis of this energy model in the context of neuromorphic computing, showing that SNNs can achieve orders-of-magnitude energy reductions compared to ANNs on neuromorphic hardware for tasks where the spike trains are sufficiently sparse.

## Technical Explanation

### The SynOps Energy Model

On neuromorphic hardware, the energy cost of a single synaptic operation (SynOp) is approximately $E_{SynOp}$. A SynOp occurs when a presynaptic spike travels across a synapse and updates the postsynaptic membrane potential. The total energy consumed by a network processing one input is:

$$E_{SNN} \approx N_{SynOps} \times E_{SynOp}$$

where $N_{SynOps}$ is the total number of synaptic operations. If neuron $i$ in layer $l$ fires $n_i^l$ spikes over $T$ timesteps, and it has $f_i^l$ downstream synapses (fan-out), then:

$$N_{SynOps} = \sum_{l} \sum_{i} n_i^l \times f_i^l$$

For a fully connected layer with $n_{in}$ input neurons and $n_{out}$ output neurons, the fan-out of each input neuron is $n_{out}$. If the total spike count in that layer is $S^l = \sum_i n_i^l$, then:

$$N_{SynOps}^l = S^l \times n_{out}^l$$

### The ANN MAC Energy Model

In an ANN, the energy cost of a single multiply-accumulate operation is approximately $E_{MAC}$. For a fully connected layer with $n_{in}$ inputs and $n_{out}$ outputs, the number of MACs per forward pass is $n_{in} \times n_{out}$. The total energy is:

$$E_{ANN} = N_{MACs} \times E_{MAC} = n_{in} \times n_{out} \times E_{MAC}$$

This is independent of the input values. Every weight is multiplied by every input activation, regardless of whether the activation is zero or not (in practice, hardware may skip zero multiplications, but this is not guaranteed).

### The Energy Ratio

The ratio of SNN to ANN energy for a single layer is:

$$\frac{E_{SNN}}{E_{ANN}} \approx \frac{S^l \times n_{out}^l \times E_{SynOp}}{n_{in}^l \times n_{out}^l \times E_{MAC}} = \frac{S^l}{n_{in}^l} \times \frac{E_{SynOp}}{E_{MAC}}$$

The term $S^l / n_{in}^l$ is the average number of spikes per input neuron per forward pass (over $T$ timesteps). If the average firing rate is $r$ spikes per timestep, then $S^l / n_{in}^l = r \times T$.

On neuromorphic hardware, $E_{SynOp} / E_{MAC}$ is typically much less than 1 (SynOps are cheaper than MACs because they involve only addition, not multiplication). Combined with low firing rates ($r \ll 1$), the SNN can be substantially more energy-efficient.

### The Project's Energy Proxy

The accompanying project does not run on neuromorphic hardware. It runs on a GPU (or CPU) using PyTorch. The "energy proxy" used in the project is a computational estimate:

$$\text{energy\_proxy} = \text{total\_spike\_count} \times \text{synaptic\_ops\_per\_spike}$$

where `total_spike_count` is the sum of all spikes emitted by all hidden neurons over all $T$ timesteps and all samples in a batch, and `synaptic_ops_per_spike` is the fan-out of the spiking layer (i.e., the number of downstream neurons, which equals the width of the next layer).

**This is explicitly NOT a measurement of actual hardware energy consumption.** It is a proxy that captures the key factor that determines energy on neuromorphic hardware (the number of SynOps), but it does not account for: static power consumption, memory access costs, routing overhead on neuromorphic chips, or the actual $E_{SynOp}$ value of any specific hardware. The proxy is useful for comparing the relative energy cost of different sparsity levels within the same experiment, but it cannot be used to make absolute energy claims.

### Controlling Sparsity via the LIF Threshold

In the accompanying project, spike sparsity is controlled by adjusting the LIF firing threshold $V_{th}$. The mechanism is straightforward:

- **Higher threshold:** The membrane potential must accumulate more input before a spike is triggered. With the same input distribution, fewer neurons reach the threshold, so fewer neurons fire. The active neuron fraction decreases.
- **Lower threshold:** The membrane potential reaches the threshold more easily. More neurons fire. The active neuron fraction increases.

This is the primary experimental manipulation in the project: the threshold is raised from its default value of 1.0 to higher values (e.g., 1.5, 2.0, 3.0), and the effect on both sparsity and catastrophic forgetting is measured. The hypothesis is that higher sparsity (fewer active neurons) leads to less representational overlap between tasks, which in turn reduces catastrophic forgetting.

It is important to note that raising the threshold also affects accuracy: if the threshold is too high, too few neurons fire and the network loses representational capacity. There is a tradeoff between sparsity (and its potential benefits for continual learning) and task performance.

## Core Concepts

- **Spike sparsity:** The fraction of neurons that are silent (do not fire) over the simulation window. High sparsity = few active neurons.
- **Active neuron fraction:** The fraction of hidden neurons that fire at least one spike over $T$ timesteps. This is the project's primary sparsity metric.
- **SynOp (synaptic operation):** A single addition to a postsynaptic membrane potential, triggered by a presynaptic spike. The fundamental energy unit on neuromorphic hardware.
- **MAC (multiply-accumulate):** The fundamental operation in ANN inference: multiply an input by a weight and add to an accumulator. More expensive than a SynOp.
- **Energy proxy:** A computational estimate of energy based on spike counts and fan-out. Not a hardware measurement.
- **Threshold control:** The mechanism by which sparsity is adjusted in the project: raising $V_{th}$ reduces the active neuron fraction.
- **Representational overlap:** The degree to which the same neurons are active for different tasks. High overlap is associated with catastrophic forgetting; low overlap (high sparsity) may reduce it.

## What / Why / When / How

**What** is the active neuron percentage in the project? It is the fraction of hidden neurons (across both hidden layers) that emit at least one spike over the $T = 25$ timesteps, averaged over the batch. Formally, for a batch of $B$ samples and two hidden layers each with $N = 256$ neurons:

$$\text{active\_neuron\_pct} = \frac{1}{2BN} \sum_{b=1}^{B} \sum_{l \in \{1,2\}} \sum_{i=1}^{N} \mathbf{1}\left[\sum_{t=1}^{T} s_i^{l,b}[t] \geq 1\right]$$

where $s_i^{l,b}[t] \in \{0,1\}$ is the spike of neuron $i$ in layer $l$ for sample $b$ at timestep $t$.

**Why** use active neuron fraction rather than average firing rate? The active neuron fraction captures whether a neuron participates in the representation at all, which is more directly relevant to representational overlap than the average rate. A neuron that fires once is "active" and contributes to the representation; a neuron that fires zero times is "silent" and does not.

**When** does sparsity help with continual learning? The hypothesis (to be tested in the project) is that when different tasks activate largely non-overlapping subsets of neurons, learning a new task does not overwrite the weights important for old tasks. This is analogous to the "sparse coding" hypothesis in neuroscience, where sparse representations reduce interference between stored memories.

**How** is the energy proxy computed in the project? At each forward pass, the total number of spikes in the two hidden layers is counted. This is multiplied by the fan-out (the number of neurons in the next layer, which is 256 for the first hidden layer and the number of output classes for the second). The result is the energy proxy for that batch.

## Advantages and Disadvantages

**Advantages of high spike sparsity:**
- Reduced energy consumption on neuromorphic hardware (fewer SynOps).
- Potential reduction in catastrophic forgetting via reduced representational overlap.
- Implicit regularization: sparse representations may generalize better.
- Interpretability: sparse activations are easier to analyze.

**Disadvantages of high spike sparsity:**
- Reduced representational capacity: fewer active neurons means less information is encoded.
- Accuracy tradeoff: too few spikes can degrade task performance.
- Training instability: very high thresholds can cause the network to produce no spikes at all, making the loss uninformative and training to stall.
- The energy benefit is only realized on neuromorphic hardware; on GPUs, sparse spike trains do not reduce computation.

## Historical Context

The connection between neural sparsity and energy efficiency in biological brains was noted by Attwell and Laughlin (2001), who estimated that the brain's energy budget is dominated by synaptic transmission and that sparse firing is essential for keeping this budget manageable. The idea that sparse coding also reduces interference between memories was developed in the context of Hopfield networks and sparse distributed memory (Kanerva, 1988).

In the SNN deep learning literature, the energy efficiency argument was formalized by Roy, Jaiswal, and Panda (2019), who provided quantitative comparisons of SynOps vs. MACs for several benchmark tasks. The connection between SNN sparsity and continual learning is more recent; work on sparse activation in SNN continual learning has begun to appear in the literature, motivated by the observation that sparse representations naturally reduce the overlap between task-specific activation patterns.

The use of threshold adjustment as a sparsity control mechanism is straightforward and has been used in various forms in the SNN literature. More sophisticated sparsity control methods include activity regularization (adding a penalty on the total spike count to the loss function) and homeostatic plasticity rules (biologically inspired mechanisms that adjust thresholds to maintain a target firing rate).

## Comparison

| Sparsity control method | Mechanism | Pros | Cons |
|---|---|---|---|
| Threshold increase (project) | Raise $V_{th}$ | Simple, direct, interpretable | Affects accuracy; no per-neuron control |
| Activity regularization | Add spike-count penalty to loss | Differentiable, end-to-end | Adds hyperparameter; may conflict with task loss |
| Homeostatic plasticity | Adaptive per-neuron threshold | Biologically plausible | Complex; harder to train |
| Dropout on spikes | Randomly zero out spikes | Simple; regularization effect | Not true sparsity; random, not structured |
| Learned sparse coding | Sparse autoencoder pre-training | Rich representations | Two-stage training; complex |

## Visual Intuition

The following diagram shows how raising the threshold affects the spike trains of a single neuron receiving the same constant input.

```
Input current: 0.5 per timestep, beta = 0.9512, T = 10

Threshold = 1.0 (default):
V_m: 0.50, 0.98, 1.43* -> reset, 0.50, 0.98, 1.43* -> reset, 0.50, 0.98, 1.43* -> reset, 0.50
Spikes: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Active: YES (3 spikes)

Threshold = 2.0 (raised):
V_m: 0.50, 0.98, 1.43, 1.86, 2.27* -> reset, 0.50, 0.98, 1.43, 1.86, 2.27*
Spikes: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
Active: YES (2 spikes, but fewer)

Threshold = 5.0 (very high):
V_m: 0.50, 0.98, 1.43, 1.86, 2.27, 2.66, 3.03, 3.38, 3.72, 4.04
Spikes: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Active: NO (0 spikes -- neuron is silent for all T=10 steps)

Active neuron fraction across a layer of 256 neurons:
Threshold 1.0: ~80% active  (many neurons fire)
Threshold 2.0: ~50% active  (moderate sparsity)
Threshold 5.0: ~10% active  (high sparsity)

Higher threshold --> fewer active neurons --> sparser representation
```

The key intuition: the threshold acts as a gate. Only neurons that receive enough input to accumulate membrane potential above the threshold will fire. By raising the threshold, we require more evidence before a neuron fires, resulting in a sparser, more selective representation.

## Mathematical Foundations

### Formal definition of active neuron percentage

Let $B$ be the batch size, $L = 2$ the number of hidden layers, $N = 256$ the number of neurons per hidden layer, and $T = 25$ the number of timesteps. Let $s_i^{l,b}[t] \in \{0,1\}$ be the spike of neuron $i$ in layer $l$ for sample $b$ at timestep $t$.

Define the indicator of activity for neuron $i$ in layer $l$ for sample $b$:

$$a_i^{l,b} = \mathbf{1}\left[\sum_{t=1}^{T} s_i^{l,b}[t] \geq 1\right]$$

The active neuron percentage is:

$$\text{ANP} = \frac{1}{L \cdot B \cdot N} \sum_{l=1}^{L} \sum_{b=1}^{B} \sum_{i=1}^{N} a_i^{l,b} \times 100\%$$

This is a number between 0% (all neurons silent) and 100% (all neurons fire at least once).

### Energy proxy formula

Let $S^l = \sum_{b=1}^{B} \sum_{i=1}^{N} \sum_{t=1}^{T} s_i^{l,b}[t]$ be the total spike count in layer $l$ over the batch. Let $f^l$ be the fan-out of layer $l$ (the number of neurons in the next layer). The energy proxy for layer $l$ is:

$$\text{EP}^l = S^l \times f^l$$

For the project's architecture (784 -> 256 -> 256 -> 2):
- Layer 1 (hidden 1, 256 neurons): fan-out = 256 (connects to hidden layer 2)
- Layer 2 (hidden 2, 256 neurons): fan-out = 2 (connects to output, 2-way head per task)

The total energy proxy is:

$$\text{EP} = S^1 \times 256 + S^2 \times 2$$

**This is a computational estimate, not a hardware energy measurement.** It counts the number of synaptic operations that would be performed on ideal neuromorphic hardware, but does not account for static power, memory access, routing, or the actual energy per SynOp of any specific chip.

### Comparison with ANN MACs

For the equivalent ANN (same architecture, ReLU activations, no temporal dimension):
- Layer 1: $784 \times 256 = 200{,}704$ MACs per sample
- Layer 2: $256 \times 256 = 65{,}536$ MACs per sample
- Output: $256 \times 2 = 512$ MACs per sample
- Total: $266{,}752$ MACs per sample

For the SNN with 50% active neurons and $T = 25$ timesteps:
- Average spikes in layer 1: $0.5 \times 256 \times 25 = 3{,}200$ spikes per sample (if each active neuron fires once per timestep on average -- this is an overestimate; active means at least once over all $T$ steps)
- More precisely: if ANP = 50%, then on average 128 neurons fire at least once. If each fires on average $r \times T$ times total, the SynOps depend on $r$.

The key point is that the SNN energy proxy scales with the total spike count, which is controlled by the threshold. The ANN MAC count is fixed. Whether the SNN is more efficient depends on the actual firing rate and the hardware-specific ratio $E_{SynOp}/E_{MAC}$.

### Threshold and firing rate relationship

For a LIF neuron with constant input $I_{in}$, decay $\beta$, and threshold $V_{th}$, the inter-spike interval (ISI) is:

$$\text{ISI} = \left\lceil \frac{\ln(1 - V_{th}(1-\beta)/I_{in})}{\ln(\beta)} \right\rceil$$

The firing rate is $r = 1/\text{ISI}$ spikes per timestep. Doubling $V_{th}$ approximately doubles the ISI (for moderate input levels), halving the firing rate. This is the quantitative basis for using threshold adjustment as a sparsity control.

## Implementation Notes

- **Computing ANP in PyTorch:** Given `spk_rec` of shape `[T, B, N]`, the active neuron percentage is:

      active = (spk_rec.sum(dim=0) > 0).float()  # shape [B, N]
      anp = active.mean().item() * 100.0

- **Computing the energy proxy:** Given `spk_rec1` (layer 1 spikes, shape `[T, B, 256]`) and `spk_rec2` (layer 2 spikes, shape `[T, B, 256]`):

      total_spikes_l1 = spk_rec1.sum().item()
      total_spikes_l2 = spk_rec2.sum().item()
      energy_proxy = total_spikes_l1 * 256 + total_spikes_l2 * 2

- **Threshold as a hyperparameter:** In snntorch, the threshold is set at construction time: `snn.Leaky(beta=beta, threshold=V_th)`. To sweep over threshold values, create a new model for each value.
- **Accuracy-sparsity tradeoff:** Monitor both accuracy and ANP during threshold sweeps. If accuracy drops sharply at a given threshold, the network is too sparse to represent the task. A useful heuristic is to keep ANP above 10--20% to maintain representational capacity.
- **Batch size effects:** ANP is averaged over the batch. With small batches, ANP estimates are noisy. Use a batch size of at least 64 for stable estimates.
- **Layer-wise vs. network-wide ANP:** The project averages ANP over both hidden layers. It can also be informative to track ANP per layer separately, as different layers may have different sparsity levels.

## Examples

**Simple example:** A single hidden layer with 4 neurons, $T = 5$ timesteps, batch size 2.

    Spike records (T=5, B=2, N=4):
    Sample 1: [[1,0,0,1], [0,0,0,1], [1,0,0,0], [0,0,0,1], [1,0,0,0]]
    Sample 2: [[0,1,0,0], [0,0,0,0], [0,1,0,0], [0,0,0,0], [0,1,0,0]]

    Sum over T:
    Sample 1: [3, 0, 0, 3]  -> active: [1, 0, 0, 1]  -> 2/4 = 50% active
    Sample 2: [0, 3, 0, 0]  -> active: [0, 1, 0, 0]  -> 1/4 = 25% active

    ANP = (50% + 25%) / 2 = 37.5%

    Energy proxy (fan-out = 10 for next layer):
    Total spikes = (3+0+0+3) + (0+3+0+0) = 9
    EP = 9 * 10 = 90 SynOps

**Real-world example (the project):** With threshold = 1.0 (default), the project's SNN achieves approximately 70--80% ANP on MNIST (most hidden neurons fire at least once over 25 timesteps). With threshold = 2.0, ANP drops to approximately 40--50%. With threshold = 3.0, ANP drops to approximately 15--25%. The research question is whether the lower ANP at higher thresholds correlates with reduced catastrophic forgetting on Split-MNIST, and whether this effect is mediated by reduced representational overlap between tasks.

## Current Research

Research on spike sparsity in SNNs is active on several fronts. On the energy side, recent work has moved toward more precise energy modeling that accounts for memory access costs, routing overhead on neuromorphic chips, and the distinction between static and dynamic power. Simple SynOp-count proxies are increasingly recognized as insufficient for accurate energy comparison, and researchers are developing more detailed simulation tools for specific neuromorphic platforms.

On the continual learning side, recent work on sparse activation in SNN continual learning has explored whether sparsity-inducing mechanisms (threshold adjustment, activity regularization, homeostatic plasticity) can reduce catastrophic forgetting. The theoretical connection between sparse representations and reduced interference is well-established in the memory literature, but empirical results in deep SNNs are mixed and task-dependent.

Open problems include: (1) how to jointly optimize for accuracy, sparsity, and continual learning performance without sacrificing any of the three; (2) whether the sparsity-forgetting relationship holds across different architectures, datasets, and task sequences; (3) how to measure representational overlap precisely and connect it to both sparsity and forgetting; (4) whether the energy benefits of SNN sparsity can be realized in practice on current neuromorphic hardware for real-world tasks.

## References

- Roy, K., Jaiswal, A., & Panda, P. (2019). Towards spike-based machine intelligence with neuromorphic computing. *Nature*, 575(7784), 607--617.
- Eshraghian, J. K., Ward, M., Neftci, E., Wang, X., Liao, B., Shrestha, A., Linares-Barranco, B., & Perez-Nieves, N. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016--1054.
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659--1671.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press.
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51--63.

---

*End of Part 2. Part 3 covers representational overlap metrics, the experimental design, and the analysis of results.*


---

# Companion Learning Guide -- Part 3: Mechanism, Measurement, Methodology, and Statistics

> **Note:** This is the final part of a three-part companion guide. Part 1 covered deep learning foundations and continual learning; Part 2 covered spiking neural networks, the LIF neuron model, surrogate gradients, and spike sparsity. This part connects those foundations to the actual research mechanism, shows how to measure it, explains the experimental methodology and its honest complications, and provides the statistical toolkit needed to interpret results. By the end you should be able to understand, implement, reproduce, and extend the proposal.

---

## Table of Contents

1. [Chapter 1: The Central Hypothesis and Mechanism](#chapter-1)
2. [Chapter 2: Continual-Learning Measurement](#chapter-2)
3. [Chapter 3: Representational Overlap Metrics](#chapter-3)
4. [Chapter 4: Confound Controls](#chapter-4)
5. [Chapter 5: Experimental Methodology and Reproducibility](#chapter-5)
6. [Chapter 6: Statistics for This Study](#chapter-6)
7. [Capstone: Putting It All Together](#capstone)
8. [Glossary](#glossary)
9. [Further Reading](#further-reading)

---

<a name="chapter-1"></a>
# Chapter 1: The Central Hypothesis and Mechanism

## Overview

The proposal rests on a specific causal story: that raising the LIF firing threshold suppresses spiking activity, which in turn reduces the overlap between the hidden representations used for different tasks, which in turn reduces the interference that causes catastrophic forgetting. This chapter unpacks that story carefully -- distinguishing mediation from mere correlation, and introducing the competing cost that creates a tradeoff.

## Fundamental Theory

Catastrophic forgetting happens because a neural network uses overlapping sets of weights and hidden units to represent different tasks. When task B is trained, gradient updates shift the weights that were tuned for task A, and performance on A collapses. The severity of this interference depends on how much the two tasks share in representation space. If task A and task B activate completely disjoint sets of hidden units, then training on B cannot touch the weights that matter for A -- there is no pathway for interference.

Spike sparsity in an SNN creates a natural mechanism for representational separation. When only a small fraction of neurons fire for any given input, the probability that two different tasks activate the same neuron is reduced. This is not guaranteed -- two tasks could still happen to activate the same sparse subset -- but on average, sparser activity means less overlap.

The firing threshold $\vartheta$ in the LIF model (covered in Part 2) is the primary control knob. A higher threshold means a neuron requires more accumulated input before it fires. Raising $\vartheta$ uniformly across the hidden layer suppresses the overall firing rate. The hypothesis is that this suppression is the lever that reduces overlap and thereby reduces forgetting.

## Technical Explanation

The causal chain has four links:

```
Higher threshold (vartheta up)
        |
        v
Lower spiking activity (mean firing rate down)
        |
        v
Fewer active hidden units per task (sparse, task-specific subsets)
        |
        v
Lower task-representation overlap (subsets share fewer neurons)
        |
        v
Less weight interference during sequential training
        |
        v
Less catastrophic forgetting
```

Each arrow is a claim that can be tested independently. The proposal instruments the chain at two points: the endpoint (forgetting, measured by the accuracy matrix) and the mediating variable (representational overlap, measured by PCA subspace overlap and cosine similarity).

The competing cost runs in the opposite direction:

```
Higher threshold (vartheta up)
        |
        v
Lower spiking activity
        |
        v
Fewer active units available to represent task A
        |
        v
Reduced representational capacity
        |
        v
Lower within-task accuracy
```

These two effects -- reduced interference and reduced capacity -- pull in opposite directions. The result is an inverted-U relationship between sparsity and net performance: moderate sparsity is best, extreme sparsity hurts accuracy even as it reduces forgetting.

## Core Concepts

**Mediation vs. correlation.** A mediator is a variable that lies on the causal path between the independent variable (threshold) and the outcome (forgetting). Representational overlap is proposed as a mediator, not merely a correlate. The distinction matters: if overlap mediates, then (a) threshold affects overlap, (b) overlap affects forgetting, and (c) controlling for overlap should reduce or eliminate the direct threshold-forgetting relationship. A mere correlation would mean overlap and forgetting happen to move together without one causing the other.

**Stability-plasticity tradeoff.** Any continual learning system must balance stability (retaining old knowledge) and plasticity (acquiring new knowledge). Sparsity trades plasticity for stability: a very sparse network changes fewer weights per gradient step, which protects old memories but also slows new learning. The inverted-U is a manifestation of this tradeoff.

**Representational interference.** When two tasks share hidden representations, the gradient for task B points in a direction that is not orthogonal to the gradient for task A. The dot product of these gradients is positive (they interfere). Orthogonal representations -- achieved by disjoint active units -- make the gradients orthogonal and eliminate interference.

## What / Why / When / How

**What** is the hypothesis? That LIF threshold controls forgetting through the mediating variable of representational overlap.

**Why** does this matter? If the mechanism is real, it suggests a principled, biologically motivated approach to continual learning that does not require storing old data (replay) or computing expensive Fisher information matrices (EWC).

**When** does the mechanism operate? During the training of each new task, when gradient updates could overwrite old representations. The mechanism is preventive, not corrective.

**How** is it tested? By varying the threshold, measuring forgetting and overlap across tasks, and checking whether the overlap-forgetting correlation is stronger than the threshold-forgetting correlation (which would suggest overlap is the proximate cause).

## Advantages and Disadvantages

**Advantages of the sparsity-as-protection mechanism:**
- Biologically plausible (cortical neurons fire sparsely)
- No memory overhead (no replay buffer)
- No task identity required at test time (unlike PackNet)
- Computationally cheap (threshold is a scalar hyperparameter)

**Disadvantages:**
- Capacity cost at high sparsity
- Threshold calibration is difficult: the achieved activity level drifts during training (a key complication in the pilot)
- Does not scale to tasks with very different input statistics without additional mechanisms
- The mechanism may be weaker than dedicated CL methods (EWC, replay) for large task sequences

## Historical Context

The idea that sparse, distributed representations reduce interference dates to the connectionist literature of the 1980s and 1990s. Marr's theory of the hippocampus (1971) proposed that sparse coding in the dentate gyrus reduces pattern overlap and enables pattern separation. McCloskey and Cohen (1989) named catastrophic interference in neural networks. The complementary learning systems theory (McClelland et al., 1995) proposed that the hippocampus uses sparse codes for rapid, non-interfering storage while the neocortex uses dense, overlapping codes for slow, generalizing storage. The SNN literature has revisited these ideas with biologically realistic neuron models, but rigorous empirical tests of the sparsity-forgetting link in SNNs remain sparse (no pun intended).

## Comparison

| Mechanism | Reduces overlap? | Requires task labels? | Memory overhead | Capacity cost |
|-----------|-----------------|----------------------|-----------------|---------------|
| Sparsity (this work) | Yes, probabilistically | No | None | Yes, at extremes |
| EWC (Kirkpatrick et al. 2017) | No (dense) | Yes (task boundary) | O(params) | Minimal |
| Replay | No (dense) | No | O(buffer) | Minimal |
| PackNet (Mallya & Lazebnik 2018) | Yes, by construction | Yes (task boundary) | None | Yes (fixed budget) |
| Progressive Networks | Yes, by construction | Yes | O(tasks * params) | None |

## Visual Intuition

The inverted-U tradeoff:

```
  Accuracy
  (or net
  performance)
     |
 1.0 |          *
     |        *   *
 0.9 |      *       *
     |    *           *
 0.8 |  *               *
     | *                  *
 0.7 |*                     *
     +--+--+--+--+--+--+--+---> Sparsity (firing rate)
        low                high
        (dense)            (very sparse)

  <-- more forgetting -->  <-- less forgetting -->
  <-- more capacity   -->  <-- less capacity    -->
```

The mediation diagram:

```
                    [Representational Overlap]
                   /                          \
                  / (a)                    (b) \
                 /                              \
[Threshold] ---/-------------------------------> [Forgetting]
               (c) direct path (should be weak
                   if overlap fully mediates)
```

## Mathematical Foundations

Let $\vartheta$ be the LIF firing threshold, $r$ the mean firing rate of the hidden layer, $O$ the representational overlap between tasks A and B, and $F$ the mean forgetting.

The hypothesis asserts:

$$\vartheta \uparrow \;\Rightarrow\; r \downarrow \;\Rightarrow\; O \downarrow \;\Rightarrow\; F \downarrow$$

with a competing path:

$$\vartheta \uparrow \;\Rightarrow\; r \downarrow \;\Rightarrow\; \text{capacity} \downarrow \;\Rightarrow\; \text{accuracy} \downarrow$$

The net effect on a combined metric (e.g., final average accuracy, which penalizes both forgetting and low within-task accuracy) is:

$$\text{net performance}(\vartheta) = \alpha \cdot \text{accuracy}(\vartheta) - \beta \cdot F(\vartheta)$$

where $\alpha, \beta > 0$ are implicit weights. This is maximized at an interior point $\vartheta^*$, giving the inverted-U.

For the quadratic approximation, fit:

$$\text{net performance} \approx a\vartheta^2 + b\vartheta + c, \quad a < 0$$

The vertex (optimal threshold) is:

$$\vartheta^* = -\frac{b}{2a}$$

The interior-peak condition requires $\vartheta^* \in (\vartheta_{\min}, \vartheta_{\max})$, i.e., the optimum lies strictly inside the tested range, not at a boundary. If $\vartheta^*$ falls outside the range, the data are consistent with a monotone relationship and the inverted-U claim is not supported.

## Implementation Notes

- The threshold $\vartheta$ is a global scalar in the simplest implementation; per-layer or per-neuron thresholds are possible but add complexity.
- Achieved firing rate drifts during training because the weight distribution changes. Calibrate threshold on a trained (or partially trained) network, not an untrained one.
- The mediation analysis requires measuring overlap at a fixed point in training (e.g., after task A, before task B) to avoid confounding with task-B learning.
- Do not conflate the nominal threshold with the achieved activity level; report both.

## Examples

**Simple example.** Suppose a 100-unit hidden layer. With a low threshold, 80 units fire for task A and 80 for task B; expected overlap is $80 \times 80 / 100 = 64$ units. With a high threshold, 10 units fire for each task; expected overlap is $10 \times 10 / 100 = 1$ unit. The interference is proportional to the overlap.

**Worked example from the pilot.** At nominal target activity 0.10 (best condition): achieved activity 0.443, accuracy 0.977, forgetting 0.026, PCA overlap 0.676. At nominal target 0.80 (worst condition): achieved activity 0.462, accuracy 0.735, forgetting 0.319, PCA overlap 0.526. The forgetting difference (0.319 - 0.026 = 0.293) is large. The PCA overlap difference (0.676 - 0.526 = 0.150) is in the predicted direction. Across all 15 runs, PCA overlap vs. forgetting gives $r = -0.873$, strongly supporting the mechanism link.

Note the surprise: the achieved activities at the two extremes (0.654 and 0.462) are not as different as the nominal targets (0.01 and 0.80) would suggest. The threshold calibration confound (calibrated on an untrained network) caused the achieved activities to cluster in the 0.35-0.65 range. This is why the threshold -- not the achieved activity -- is the cleanly manipulated variable.

## Current Research

Recent SNN continual-learning work (e.g., Hebbian orthogonal-projection and sparse-pathway methods) has explored related ideas, but few papers rigorously instrument the representational overlap as a mediator. The field lacks a standard benchmark for SNN-specific CL evaluation. Open problems include: (1) whether the mechanism generalizes beyond Split-MNIST to more complex datasets; (2) whether per-neuron adaptive thresholds can achieve better tradeoffs; (3) how the mechanism interacts with recurrent SNN architectures.

## References

- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419-457.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

---

<a name="chapter-2"></a>
# Chapter 2: Continual-Learning Measurement

## Overview

Before you can claim that sparsity reduces forgetting, you need a precise, reproducible way to measure forgetting. This chapter defines the standard continual-learning metrics: the accuracy matrix, final average accuracy, per-task forgetting, mean forgetting, backward transfer, and forward transfer. These are the numbers that appear in every table in the proposal.

## Fundamental Theory

A continual learning experiment trains a model sequentially on $T$ tasks. After training on task $i$, you evaluate the model on all tasks $j = 1, \ldots, T$. This produces a matrix of accuracy values. The matrix encodes everything: how well the model learned each task, how much it forgot earlier tasks, and whether learning later tasks helped earlier ones.

The key insight is that a single final accuracy number is insufficient. A model that achieves 90% average accuracy by doing well on the last task and forgetting all earlier ones is very different from a model that achieves 90% by retaining all tasks equally. The matrix separates these cases.

## Technical Explanation

Define the accuracy matrix $A \in \mathbb{R}^{T \times T}$ where:

$$A[i, j] = \text{accuracy on task } j \text{ after training on task } i$$

Only the lower triangle (including the diagonal) is meaningful: you cannot evaluate on task $j$ before training on it (assuming no forward transfer measurement), and you evaluate after each task is trained.

```
         Task evaluated (j)
         1     2     3     4
       +-----+-----+-----+-----+
  1    | A11 |     |     |     |   <- after training task 1
  2    | A21 | A22 |     |     |   <- after training task 2
  3    | A31 | A32 | A33 |     |   <- after training task 3
  4    | A41 | A42 | A43 | A44 |   <- after training task 4
       +-----+-----+-----+-----+
Task
trained
(i)
```

The diagonal $A[i,i]$ is the within-task accuracy immediately after training. The off-diagonal lower entries $A[i,j]$ for $j < i$ show how much task $j$ is retained after training on tasks $i > j$.

## Core Concepts

**Final average accuracy.** After training all $T$ tasks, the average accuracy across all tasks:

$$\text{ACC} = \frac{1}{T} \sum_{j=1}^{T} A[T, j]$$

This is the primary performance metric. It penalizes both forgetting (low $A[T,j]$ for $j < T$) and poor within-task learning (low $A[T,T]$).

**Per-task forgetting.** For task $j < T$, forgetting is the drop from the best accuracy ever achieved on task $j$ to the final accuracy:

$$F_j = \max_{i \leq j} A[i, j] - A[T, j]$$

The maximum is taken over all evaluations up to and including the evaluation immediately after training task $j$ (i.e., $i \leq j$, since $A[i,j]$ is only defined for $i \geq j$). In practice, the peak is almost always $A[j,j]$ (the within-task accuracy right after training), so:

$$F_j \approx A[j, j] - A[T, j]$$

Forgetting is non-negative by definition (you cannot forget something you never knew, and the max ensures we measure the drop from the best point).

**Mean forgetting.** Average forgetting over all tasks except the last (the last task cannot be forgotten since it is trained last):

$$\bar{F} = \frac{1}{T-1} \sum_{j=1}^{T-1} F_j$$

**Backward transfer (BWT).** Measures how training on later tasks affects earlier tasks, signed:

$$\text{BWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} \left( A[T, j] - A[j, j] \right)$$

BWT is negative when forgetting occurs (final accuracy is below within-task accuracy) and positive when later training helps earlier tasks (a rare but real phenomenon called positive backward transfer). Note that $\text{BWT} = -\bar{F}$ when forgetting is defined as above.

**Forward transfer (FWT).** Measures how training on earlier tasks affects later tasks, relative to a random-initialization baseline:

$$\text{FWT} = \frac{1}{T-1} \sum_{j=2}^{T} \left( A[j-1, j] - b_j \right)$$

where $b_j$ is the accuracy on task $j$ of a randomly initialized model (zero-shot baseline). Positive FWT means earlier tasks helped later ones; negative FWT means they hurt.

## What / Why / When / How

**What** does the accuracy matrix capture? The full trajectory of performance across all tasks and all training stages.

**Why** use per-task forgetting rather than just final accuracy? Final accuracy conflates forgetting with within-task learning quality. A model with high within-task accuracy but severe forgetting looks the same as a model with moderate within-task accuracy and no forgetting, if their averages happen to match.

**When** do you compute these metrics? After each task is trained, evaluate on all tasks trained so far. Store the full matrix; compute derived metrics at the end.

**How** do you handle task identity at test time? In the class-incremental setting (harder), the model must identify which task an input belongs to without being told. In the task-incremental setting (easier, used in this proposal's Split-MNIST pilot), the task identity is provided at test time, and the model uses the appropriate output head.

## Advantages and Disadvantages

**Advantages of the accuracy matrix:**
- Complete: captures all information about the learning trajectory
- Decomposable: forgetting, BWT, FWT are all derived from the same matrix
- Comparable across methods: standard in the CL literature

**Disadvantages:**
- Quadratic in the number of tasks: $T^2$ evaluations
- Does not capture within-task learning dynamics (only snapshots after each task)
- BWT and FWT require a baseline ($b_j$) that is not always well-defined

## Historical Context

The accuracy matrix formalism was popularized by Lopez-Paz and Ranzato (2017) in the GEM paper, which also defined BWT and FWT. Before that, papers reported ad hoc metrics (e.g., "accuracy on task 1 after training task 2") that were not comparable across studies. The matrix formalism has since become the standard.

## Comparison

| Metric | What it measures | Range | Sign convention |
|--------|-----------------|-------|-----------------|
| ACC | Final average accuracy | [0,1] | Higher is better |
| $F_j$ | Per-task forgetting | [0,1] | Lower is better |
| $\bar{F}$ | Mean forgetting | [0,1] | Lower is better |
| BWT | Backward transfer | [-1,1] | Higher is better (0 = no change) |
| FWT | Forward transfer | [-1,1] | Higher is better (0 = no benefit) |

## Visual Intuition

A concrete 2-task example (Split-MNIST, tasks = digits 0/1 vs. 2/3):

```
After training task 1:  A[1,1] = 0.98  (task 1 just learned)
After training task 2:  A[2,1] = 0.65  (task 1 partially forgotten)
                        A[2,2] = 0.97  (task 2 just learned)

F_1 = A[1,1] - A[2,1] = 0.98 - 0.65 = 0.33  (33% forgetting)
BWT = A[2,1] - A[1,1] = 0.65 - 0.98 = -0.33
ACC = (A[2,1] + A[2,2]) / 2 = (0.65 + 0.97) / 2 = 0.81
```

With sparsity (target activity 0.10, from the pilot):

```
After training task 1:  A[1,1] ~ 0.98
After training task 2:  A[2,1] ~ 0.97  (almost no forgetting)
                        A[2,2] ~ 0.98

F_1 ~ 0.026
ACC ~ 0.977
```

## Mathematical Foundations

Full definitions:

$$A[i,j] \in [0,1], \quad i \geq j, \quad i,j \in \{1,\ldots,T\}$$

$$F_j = \max_{i: j \leq i \leq T} A[i,j] - A[T,j], \quad j < T$$

$$\bar{F} = \frac{1}{T-1} \sum_{j=1}^{T-1} F_j$$

$$\text{BWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} \left(A[T,j] - A[j,j]\right) = -\bar{F}$$

$$\text{FWT} = \frac{1}{T-1} \sum_{j=2}^{T} \left(A[j-1,j] - b_j\right)$$

$$\text{ACC} = \frac{1}{T} \sum_{j=1}^{T} A[T,j]$$

Note: in the pilot, $T = 5$ (five pairs of MNIST digits). The mean forgetting averages over $j = 1, 2, 3, 4$.

## Implementation Notes

- Store the full matrix, not just the final row. You will want to inspect the trajectory.
- Evaluate on the full test set for each task, not just a subset.
- Use the same evaluation protocol (same task head, same preprocessing) at every evaluation point.
- For Split-MNIST with task-incremental evaluation, the task identity is provided; the model selects the appropriate 2-way output head.
- Seed the random number generator before each training run and record the seed. You need this for reproducibility.

## Examples

**Simple example.** Three tasks, perfect retention:

```
A = [[1.0,  -,   -  ],
     [1.0, 1.0,  -  ],
     [1.0, 1.0, 1.0]]

F_1 = 1.0 - 1.0 = 0.0
F_2 = 1.0 - 1.0 = 0.0
mean_F = 0.0
BWT = 0.0
ACC = 1.0
```

**Worked example from the pilot (nominal target 0.40, worst-but-one condition):**

Pilot reports mean forgetting 0.149, accuracy 0.876. This means the final row of the accuracy matrix averages to 0.876, and the average drop from within-task peak to final is 0.149. The BWT is approximately -0.149.

## Current Research

Recent work has questioned whether mean forgetting is the right metric for class-incremental settings, where the model must also learn to distinguish between tasks without task labels. Metrics like "plasticity" (average within-task accuracy) and "stability" (average retention) are sometimes reported separately. The proposal uses the standard task-incremental formulation for the pilot.

## References

- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*.
- Chaudhry, A., et al. (2018). Riemannian walk for incremental learning. *ECCV*.

---

<a name="chapter-3"></a>
# Chapter 3: Representational Overlap Metrics

## Overview

The central mechanism claim requires measuring representational overlap -- the degree to which two tasks share the same hidden-layer representations. This chapter covers three metrics: cosine overlap between task-mean representations, PCA subspace overlap, and Centered Kernel Alignment (CKA). It also explains a critical methodological subtlety: linear CKA is degenerate for cross-task comparison when the two tasks have disjoint inputs with no row-pairing, which is exactly the situation in Split-MNIST. Understanding why CKA is disabled in this project is as important as understanding how it works.

## Fundamental Theory

A neural network's hidden layer maps inputs to a high-dimensional vector space. For a given task, the set of hidden representations forms a cloud of points in that space. Two tasks "overlap" in representation space if their clouds occupy similar regions. Overlap is not a single number -- it depends on what aspect of the clouds you compare: their centers (cosine overlap), their principal directions (PCA subspace overlap), or their pairwise similarity structure (CKA).

Each metric captures a different facet of overlap and has different assumptions, failure modes, and computational costs. Using multiple metrics and checking whether they agree is good practice. When they disagree -- as they do in the pilot -- that disagreement is itself informative.

## Technical Explanation

### Metric 1: Cosine Overlap Between Task-Mean Representations

The simplest metric. For each task, compute the mean hidden representation across all inputs in that task's test set. Then compute the cosine similarity between the two mean vectors.

Let $\mathbf{h}_A = \frac{1}{|D_A|} \sum_{x \in D_A} \mathbf{h}(x)$ and $\mathbf{h}_B = \frac{1}{|D_B|} \sum_{x \in D_B} \mathbf{h}(x)$, where $\mathbf{h}(x)$ is the hidden-layer activation vector for input $x$.

$$\text{cosine\_overlap}(A, B) = \frac{\mathbf{h}_A \cdot \mathbf{h}_B}{\|\mathbf{h}_A\| \|\mathbf{h}_B\|}$$

This is fast and interpretable but coarse: it only compares the centers of the clouds, ignoring their shape and spread.

### Metric 2: PCA Subspace Overlap

A richer metric that compares the principal directions of variation within each task's representation cloud.

For task $A$, collect the hidden representations $H_A \in \mathbb{R}^{n_A \times d}$ (rows are examples, columns are hidden units). Center the rows, compute the SVD, and take the top $k$ right singular vectors as the columns of $U_A \in \mathbb{R}^{d \times k}$. These span the $k$-dimensional subspace that captures the most variance in task A's representations.

Do the same for task $B$ to get $U_B \in \mathbb{R}^{d \times k}$.

The subspace overlap is:

$$\text{PCA\_overlap}(A, B) = \frac{\|U_A^T U_B\|_F^2}{k}$$

This is the average squared cosine between the principal directions of the two tasks. It lies in $[0, 1]$: 0 means the subspaces are orthogonal (no overlap), 1 means they are identical.

In this project, $k = 10$ (top 10 principal components).

### Metric 3: Centered Kernel Alignment (CKA)

CKA (Kornblith et al., 2019) is a more sophisticated similarity measure based on the Hilbert-Schmidt Independence Criterion (HSIC). It compares the pairwise similarity structure of two sets of representations, rather than their directions.

For two representation matrices $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$ (same $n$ rows, i.e., same inputs), compute the kernel matrices $K = XX^T$ and $L = YY^T$ (linear kernels). Center them: $\tilde{K} = HKH$ and $\tilde{L} = HLH$ where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix.

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{tr}(\tilde{K}\tilde{L})$$

$$\text{CKA}(X, Y) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

CKA is invariant to orthogonal transformations and isotropic scaling, making it more robust than raw cosine similarity. It ranges from 0 (no similarity) to 1 (identical structure).

## THE CRITICAL SUBTLETY: WHY LINEAR CKA IS DISABLED FOR CROSS-TASK SPLIT-MNIST

This is not a minor implementation detail. It is a fundamental methodological point.

CKA requires that the two representation matrices $X$ and $Y$ have the **same rows** -- that is, the same $n$ inputs, evaluated in both networks (or both layers). The kernel matrices $K = XX^T$ and $L = YY^T$ are $n \times n$ matrices of pairwise similarities, and the HSIC measures how correlated these pairwise structures are. This correlation is only meaningful if row $i$ of $X$ and row $i$ of $Y$ correspond to the same input.

In Split-MNIST, task A uses digits 0 and 1, and task B uses digits 2 and 3. These are **disjoint input sets**. There is no natural pairing between a digit-0 image and a digit-2 image. If you stack the representations of task A inputs and task B inputs into a single matrix, the row correspondence is arbitrary -- you could permute the rows of one matrix and get a completely different CKA value, even though the representations themselves have not changed.

This means linear CKA is **ill-defined** for cross-task comparison in Split-MNIST. It is not that CKA gives a wrong answer; it is that the question CKA is designed to answer ("how similar is the pairwise similarity structure of these two sets of representations?") does not make sense when the two sets have no shared inputs.

In this project, cross-task CKA is deliberately set to `NaN` and excluded from analysis. This is the correct decision. Reporting a CKA value in this setting would be misleading.

**When is CKA valid?** CKA is valid for:
- Comparing two layers within the same network on the same inputs
- Comparing two networks trained on the same dataset (same inputs, different architectures or random seeds)
- Comparing representations before and after fine-tuning on the same inputs

CKA is **not valid** for comparing representations of disjoint input sets without a meaningful pairing.

**What to use instead?** PCA subspace overlap and cosine overlap between task means do not require row-pairing. They compare the geometry of the representation clouds without needing matched inputs. This is why they are the primary overlap metrics in this project.

### Linear Probes as a Complementary Measure

A linear probe is a simple linear classifier trained on top of frozen hidden representations to predict task labels. If the representations for task A are well-preserved after training on task B, a linear probe trained on task-A representations (extracted after task-B training) should achieve high accuracy on task-A test data.

Linear probe accuracy is a representation-forgetting measure: it tells you whether the information needed to solve task A is still present in the hidden layer, even if the output head has been overwritten. It complements the accuracy matrix (which measures end-to-end performance) by isolating the representation layer.

## Core Concepts

- **Subspace overlap** captures the alignment of the principal directions of variation, not just the means.
- **CKA** captures pairwise similarity structure, requiring matched inputs.
- **Row-pairing requirement** is the key constraint that makes CKA inapplicable to cross-task Split-MNIST.
- **Linear probes** measure representation quality without requiring overlap between tasks.

## What / Why / When / How

**What** do these metrics measure? Different aspects of how similar the hidden representations of two tasks are.

**Why** use multiple metrics? Because they can disagree. In the pilot, cosine overlap and PCA overlap give opposite correlations with forgetting ($r = +0.756$ vs. $r = -0.873$). This disagreement reveals that the mean representations move in one direction while the principal subspaces move in another -- a real and interesting finding.

**When** do you compute overlap? After training task A, before training task B. You want to measure the overlap of the representations that task B's training will interfere with.

**How** do you compute PCA overlap in practice? Use `numpy.linalg.svd` on the centered representation matrix, take the top $k$ right singular vectors, and compute the Frobenius norm of their inner product matrix.

## Advantages and Disadvantages

| Metric | Advantages | Disadvantages |
|--------|-----------|---------------|
| Cosine overlap | Fast, interpretable | Only compares means; ignores spread |
| PCA subspace overlap | Captures principal directions; no row-pairing needed | Sensitive to choice of $k$; ignores non-linear structure |
| CKA | Invariant to orthogonal transforms; captures full pairwise structure | Requires row-pairing; $O(n^2)$ memory; degenerate for disjoint inputs |
| Linear probes | Measures representation quality directly | Requires training a probe; conflates representation and readout |

## Historical Context

SVCCA (Raghu et al., 2017) was an early method for comparing neural network representations using singular vector canonical correlation analysis. CKA (Kornblith et al., 2019) improved on SVCCA by being invariant to orthogonal transformations and more robust to small sample sizes. Both methods were developed for comparing representations within or across networks trained on the same data, not for cross-task comparison with disjoint inputs. The application to continual learning requires care precisely because tasks have disjoint inputs.

## Comparison

| Method | Row-pairing required? | Invariant to rotation? | Captures spread? | Complexity |
|--------|----------------------|----------------------|-----------------|------------|
| Cosine overlap | No | No | No | O(d) |
| PCA subspace | No | Yes | Partially | O(nd^2) |
| CKA (linear) | Yes | Yes | Yes | O(n^2 d) |
| SVCCA | Yes | Yes | Partially | O(n^2 d) |

## Visual Intuition

Two tasks in 2D representation space:

```
Task A representations:          Task B representations:
        *                                   +
      * * *                               + + +
        *                                   +

Mean_A = center of *s            Mean_B = center of +s
PC1_A = horizontal direction     PC1_B = horizontal direction

Cosine overlap of means: depends on angle between Mean_A and Mean_B
PCA overlap: high if PC1_A and PC1_B are aligned (both horizontal here)
```

When sparsity is high, the clouds shrink and separate:

```
Low sparsity (dense):            High sparsity (sparse):
  * * * * * * * * *                    *
  * * * * * * * * *                   * *
  + + + + + + + + +                    *
  + + + + + + + + +
  (clouds overlap)                (clouds separated)
```

The CKA degeneracy for disjoint inputs:

```
Task A inputs: [img_0, img_1, img_0', ...]   (digits 0 and 1)
Task B inputs: [img_2, img_3, img_2', ...]   (digits 2 and 3)

No natural pairing: which img_0 corresponds to which img_2?
Stacking them arbitrarily:
  Row 1: img_0 paired with img_2  (arbitrary)
  Row 2: img_1 paired with img_3  (arbitrary)
  ...
CKA value depends on this arbitrary pairing -> MEANINGLESS
```

## Mathematical Foundations

**PCA subspace overlap:**

Let $H_A \in \mathbb{R}^{n_A \times d}$ be the centered representation matrix for task A. Compute SVD: $H_A = U \Sigma V^T$. Take $U_A = V[:, :k] \in \mathbb{R}^{d \times k}$ (top $k$ right singular vectors). Similarly for task B.

$$\text{PCA\_overlap}(A, B) = \frac{\|U_A^T U_B\|_F^2}{k} \in [0, 1]$$

For $k = 10$: $\|U_A^T U_B\|_F^2 = \sum_{i=1}^{10} \sum_{j=1}^{10} (u_{A,i}^T u_{B,j})^2$, divided by 10.

**CKA:**

$$K = H_A H_A^T, \quad L = H_B H_B^T \quad (n \times n \text{ matrices, same } n)$$

$$\tilde{K} = HKH, \quad \tilde{L} = HLH, \quad H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$$

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{tr}(\tilde{K}\tilde{L})$$

$$\text{CKA}(H_A, H_B) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

**Validity condition for CKA:** Row $i$ of $H_A$ and row $i$ of $H_B$ must correspond to the same input $x_i$. If this condition is violated (disjoint inputs), CKA is undefined. Set to NaN.

## Implementation Notes

- For PCA overlap, center the representation matrix before computing SVD. Uncentered SVD gives the wrong subspace.
- Use `numpy.linalg.svd(H, full_matrices=False)` for efficiency; take the last output (right singular vectors).
- For CKA, the $O(n^2)$ kernel matrix can be large. For $n = 10000$ examples, $K$ is $10000 \times 10000$ floats = 800 MB. Use minibatch CKA for large datasets.
- Always check whether your inputs are paired before computing CKA. If tasks have disjoint inputs, set CKA to NaN and document this decision.
- For linear probes, freeze the network after task-B training, extract task-A representations, train a logistic regression, and report accuracy on the task-A test set.

## Examples

**Simple example.** Two tasks, 3 examples each, 4-dimensional hidden layer:

```
H_A = [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [1, 1, 0, 0]]  (task A uses dimensions 1 and 2)

H_B = [[0, 0, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 1]]  (task B uses dimensions 3 and 4)

PC1_A ~ [1, 0, 0, 0] or [0, 1, 0, 0]
PC1_B ~ [0, 0, 1, 0] or [0, 0, 0, 1]

PCA overlap = 0.0  (orthogonal subspaces -> no interference)
```

**Worked example from the pilot.** At nominal target 0.10 (best): PCA overlap = 0.676. At nominal target 0.80 (worst): PCA overlap = 0.526. The difference (0.150) is in the predicted direction: lower sparsity (higher activity) -> higher overlap -> more forgetting. Across 15 runs, $r(\text{PCA overlap}, \text{forgetting}) = -0.873$.

The cosine overlap tells a different story: $r(\text{cosine overlap}, \text{forgetting}) = +0.756$. This means tasks with higher cosine similarity between their mean representations have *less* forgetting. This is counterintuitive and suggests that the mean representations are not the right level of analysis -- the principal subspaces (captured by PCA overlap) are more informative about interference.

## Current Research

Recent work has proposed kernel-based and neural-network-based similarity measures that go beyond linear CKA. Platonic Representation Hypothesis work (2024) uses CKA to compare representations across modalities and architectures, always with matched inputs. The SNN literature has not yet standardized on a representation similarity metric for continual learning evaluation.

## References

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML*.
- Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. *NeurIPS*.

---

<a name="chapter-4"></a>
# Chapter 4: Confound Controls

## Overview

Observing that a sparser SNN forgets less does not prove that sparsity reduces forgetting through the representational overlap mechanism. Several alternative explanations -- confounds -- could produce the same observation. This chapter identifies the main confounds and explains the experimental controls designed to rule them out. Understanding confounds is not a formality; it is the difference between a publishable mechanistic claim and a descriptive correlation.

## Fundamental Theory

A confound is a variable that is correlated with both the independent variable (threshold/sparsity) and the dependent variable (forgetting), and that could explain the observed relationship without the proposed mechanism being true. Ruling out confounds requires either (a) holding the confound constant while varying the independent variable, or (b) measuring the confound and showing it does not account for the effect.

The two main confounds in this study are the capacity confound and the plasticity confound. A third, subtler confound is the calibration confound (threshold calibrated on an untrained network), which is discussed in Chapter 5.

## Technical Explanation

### Confound 1: The Capacity Confound

**The concern.** A sparser network may simply learn less of task A in the first place. If the network never fully learned task A, then there is less to forget -- not because sparsity protects old memories, but because the network had lower capacity and encoded task A more weakly. The observed lower forgetting is then an artifact of lower initial accuracy, not a protection effect.

**Why it matters.** If the capacity confound is not controlled, you cannot distinguish "sparsity protects memories" from "sparsity prevents learning." Both produce lower forgetting, but only the first is interesting.

**The control.** Match task-A mastery across conditions. Specifically, ensure that the within-task accuracy $A[1,1]$ (accuracy on task A immediately after training) is comparable across sparsity levels. If a sparse network achieves the same $A[1,1]$ as a dense network but lower forgetting $F_1$, the capacity confound is ruled out.

In practice, this means either (a) training each condition until task-A accuracy reaches a fixed threshold (e.g., 95%), or (b) reporting $A[1,1]$ alongside forgetting and checking that the correlation between $A[1,1]$ and forgetting is not driving the sparsity-forgetting relationship.

### Confound 2: The Plasticity Confound

**The concern.** Fewer active neurons means fewer weight updates per gradient step (because the gradient flows only through active neurons in an SNN). A network with very low firing rates may simply change its weights less during task-B training, trivially preserving task-A performance -- not because the representations are separated, but because the network is nearly frozen.

**Why it matters.** If the plasticity confound is true, then sparsity is not protecting task A through representational separation; it is protecting task A by preventing learning of task B. This is a very different mechanism, and it predicts that task-B accuracy should be lower in sparser networks (which it is, at extreme sparsity -- but the question is whether this accounts for all of the forgetting reduction).

**The control: count-matched dense-frozen baseline.** Create a dense network where the same number of weights are updated during task-B training as in the sparse SNN, but the selection of which weights to update is random (not determined by spiking activity). If the dense-frozen baseline shows similar forgetting reduction, the plasticity confound is supported. If the sparse SNN shows lower forgetting than the dense-frozen baseline, the representational separation mechanism is supported.

**The control: during-task-B sparsity ablation with fixed evaluation mask.** Train the network on task A with high sparsity (high threshold). Then, during task-B training, lower the threshold to allow dense activity (removing the sparsity protection during learning). Evaluate task-A retention using the original sparse evaluation mask. If forgetting increases when sparsity is removed during task-B training, this supports the plasticity confound. If forgetting remains low, the protection comes from the representational structure established during task-A training, not from reduced plasticity during task-B training.

### Confound 3: Representation-Space Measurement as a Control

The most direct way to test the mechanism is to measure the representational overlap and show that it mediates the sparsity-forgetting relationship. If:

1. Higher threshold -> lower overlap (threshold affects the mediator)
2. Lower overlap -> lower forgetting (mediator affects the outcome)
3. Controlling for overlap reduces the threshold-forgetting relationship (mediation)

...then the representational overlap mechanism is supported and the confounds are less plausible (because they do not predict this specific pattern of mediation).

This is why the PCA subspace overlap and cosine overlap metrics are not just descriptive -- they are the mechanism instruments that allow a causal interpretation.

## Core Concepts

- **Capacity confound:** sparser net learns less of task A -> less to forget
- **Plasticity confound:** sparser net updates fewer weights -> trivially less change
- **Matched mastery control:** equalize $A[1,1]$ across conditions
- **Count-matched dense-frozen baseline:** equalize number of weight updates
- **During-task-B ablation:** remove sparsity during task-B training, keep evaluation mask
- **Mediation analysis:** measure overlap as mediator to support causal interpretation

## What / Why / When / How

**What** are confounds? Alternative explanations that could produce the observed sparsity-forgetting correlation without the proposed mechanism.

**Why** control for them? Because a mechanistic claim ("sparsity reduces forgetting through representational separation") requires ruling out simpler explanations.

**When** do you apply controls? The matched-mastery control is applied during training (stop when task-A accuracy reaches threshold). The count-matched baseline and ablation are separate experimental conditions. The mediation analysis is applied during analysis.

**How** do you implement the count-matched baseline? Count the number of weight updates in the sparse SNN during task-B training (proportional to the mean firing rate times the number of training steps). Create a dense network and randomly mask the same fraction of weight updates during task-B training.

## Advantages and Disadvantages

**Advantages of these controls:**
- Each control targets a specific alternative explanation
- Together they triangulate the mechanism
- The mediation analysis provides a positive test (not just ruling out alternatives)

**Disadvantages:**
- The count-matched baseline requires careful implementation
- The during-task-B ablation changes the experimental condition in a way that may introduce new confounds
- Mediation analysis is correlational, not experimental; it cannot prove causation

## Historical Context

The capacity and plasticity confounds are well-known in the continual learning literature. Kirkpatrick et al. (2017) addressed the plasticity confound in EWC by showing that the Fisher information weighting (not just reduced plasticity) was necessary for the forgetting reduction. Zenke et al. (2017) similarly showed that the online importance weighting in SI was not equivalent to simple weight decay. The proposal follows this tradition of mechanistic controls.

## Comparison

| Control | Confound targeted | Experimental cost | Strength of evidence |
|---------|------------------|-------------------|---------------------|
| Matched task-A mastery | Capacity | Low (training criterion) | Moderate |
| Count-matched dense-frozen | Plasticity | High (new baseline) | Strong |
| During-task-B ablation | Plasticity | Moderate (new condition) | Strong |
| Mediation analysis (overlap) | Both | Low (analysis only) | Moderate (correlational) |

## Visual Intuition

The capacity confound:

```
Dense network:
  Task A training: learns well (A[1,1] = 0.98)
  Task B training: overwrites task A (A[2,1] = 0.65)
  Forgetting = 0.33

Sparse network (capacity confound):
  Task A training: learns poorly (A[1,1] = 0.75)  <- lower capacity
  Task B training: overwrites task A (A[2,1] = 0.72)
  Forgetting = 0.03  <- looks like protection, but task A was never well-learned

Sparse network (true protection):
  Task A training: learns well (A[1,1] = 0.97)  <- matched mastery
  Task B training: task A preserved (A[2,1] = 0.95)
  Forgetting = 0.02  <- genuine protection
```

The plasticity confound:

```
Sparse SNN:
  Task B gradient flows through 10% of neurons
  -> 10% of weights updated
  -> task A weights mostly unchanged
  -> low forgetting (but is this protection or just frozen weights?)

Count-matched dense baseline:
  Task B gradient flows through all neurons
  -> randomly mask 90% of weight updates
  -> same number of weights updated as sparse SNN
  -> if forgetting is similar: plasticity confound supported
  -> if forgetting is higher: representational separation supported
```

## Mathematical Foundations

**Matched mastery condition:**

$$A[1,1]^{\text{sparse}} \approx A[1,1]^{\text{dense}} \pm \epsilon$$

where $\epsilon$ is a tolerance (e.g., 0.02). If this holds and $F_1^{\text{sparse}} < F_1^{\text{dense}}$, the capacity confound is ruled out.

**Plasticity confound formalization.** Let $\Delta W_B$ be the weight change during task-B training. The plasticity confound predicts:

$$\|\Delta W_B^{\text{sparse}}\|_F \ll \|\Delta W_B^{\text{dense}}\|_F$$

and that this difference in $\|\Delta W_B\|_F$ accounts for the forgetting difference. The count-matched baseline tests this by equalizing $\|\Delta W_B\|_F$ while removing the representational structure.

**Mediation analysis.** Let $T$ = threshold, $O$ = overlap, $F$ = forgetting. The mediation model is:

$$O = \alpha_0 + \alpha_1 T + \epsilon_1$$
$$F = \beta_0 + \beta_1 T + \beta_2 O + \epsilon_2$$

If $\beta_1$ is significantly reduced (or becomes non-significant) when $O$ is included in the model, overlap mediates the threshold-forgetting relationship. The indirect effect is $\alpha_1 \beta_2$.

## Implementation Notes

- For matched mastery, use early stopping with a task-A accuracy criterion, not a fixed number of epochs. Different sparsity levels may require different numbers of epochs to reach the same accuracy.
- For the count-matched baseline, the random mask should be re-sampled each gradient step (not fixed), to avoid introducing structured sparsity.
- For the during-task-B ablation, lower the threshold to the dense baseline value during task-B training, then restore it for evaluation. Document exactly when the threshold is changed.
- For mediation analysis, use bootstrapped confidence intervals for the indirect effect (the Sobel test is underpowered for small samples).

## Examples

**Simple example.** Suppose at threshold $\vartheta = 1.0$ (sparse), $A[1,1] = 0.97$ and $F_1 = 0.03$. At threshold $\vartheta = 0.5$ (dense), $A[1,1] = 0.98$ and $F_1 = 0.30$. The mastery levels are matched (0.97 vs. 0.98), so the capacity confound is ruled out. The forgetting difference (0.27) is attributable to the sparsity mechanism.

**Worked example from the pilot.** At nominal target 0.10 (best): accuracy 0.977, forgetting 0.026. At nominal target 0.80 (worst): accuracy 0.735, forgetting 0.319. The accuracy difference (0.977 - 0.735 = 0.242) suggests the capacity confound may be partially active: the dense network also learned task A better. The pilot does not fully control for this; it is a limitation acknowledged in the reframed claim. The full study will implement matched-mastery controls.

## Current Research

Recent continual learning papers have become more careful about confound controls. Pham et al. (2021) showed that many CL methods' advantages disappear when capacity is matched. The SNN-CL literature has not yet systematically applied these controls, which is a gap this proposal begins to address.

## References

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*.
- Pham, Q., Liu, C., Hoi, S., & Zhang, C. (2021). DualPrompt: Complementary prompting for rehearsal-free continual learning. *ECCV*.

---

<a name="chapter-5"></a>
# Chapter 5: Experimental Methodology and Reproducibility

## Overview

This chapter describes the actual experimental setup: the Split-MNIST pilot, the LIF-SNN architecture, the threshold calibration procedure, the nominal activity targets, and -- critically -- the honest complications that emerged. It then describes the reframed claim that the pilot supports, and the expansion ladder for the full study. Reproducibility is treated as a first-class concern throughout.

## Fundamental Theory

A good experiment manipulates one variable (the independent variable), holds everything else constant, and measures the outcome. In this study, the independent variable is the LIF firing threshold $\vartheta$. Everything else -- architecture, dataset, training procedure, evaluation protocol -- is held constant across conditions. The outcome variables are forgetting, accuracy, and representational overlap.

The challenge is that $\vartheta$ does not directly control the firing rate; it controls the threshold for firing, and the actual firing rate depends on the input statistics and the learned weights. This indirect relationship creates the calibration problem that is the central complication of the pilot.

## Technical Explanation

### The Split-MNIST Benchmark

Split-MNIST divides the MNIST dataset (70,000 images of handwritten digits 0-9) into five sequential tasks:
- Task 1: digits 0 vs. 1
- Task 2: digits 2 vs. 3
- Task 3: digits 4 vs. 5
- Task 4: digits 6 vs. 7
- Task 5: digits 8 vs. 9

Each task is a binary classification problem. The model is trained on tasks 1 through 5 sequentially, with no access to previous tasks' data during later training (naive sequential, no replay). The task identity is provided at test time (task-incremental setting), so the model uses the appropriate 2-way output head.

Split-MNIST is a standard CL benchmark because it is simple enough to run quickly (useful for pilot studies) but exhibits clear catastrophic forgetting in naive sequential training.

### The LIF-SNN Architecture

The network is a feedforward LIF-SNN:
- Input layer: 784 units (flattened 28x28 MNIST pixels, rate-coded as spike trains)
- Hidden layer: 256 LIF neurons (the layer where sparsity is controlled)
- Output layer: 2 units per task (task-specific heads)

The LIF neuron model (covered in Part 2) uses a fixed firing threshold $\vartheta$ across all hidden neurons. Surrogate gradients (straight-through estimator) enable backpropagation through the spike function.

### Threshold-Controlled Sparsity

The firing threshold $\vartheta$ is the primary experimental manipulation. Five nominal activity targets are tested: 1%, 10%, 20%, 40%, and 80% of hidden neurons firing per time step. For each target, the threshold is calibrated by bisection to achieve the target activity on the untrained network.

**Bisection calibration procedure:**
1. Initialize the network with random weights.
2. Pass a batch of task-1 training data through the network.
3. Measure the mean firing rate of the hidden layer.
4. If the firing rate is above the target, increase $\vartheta$; if below, decrease $\vartheta$.
5. Repeat until the firing rate is within a tolerance of the target.

This calibration is performed once, before training begins. The threshold is then fixed for the entire training run.

### The Calibration Confound: The Central Complication

Here is the honest complication that the pilot revealed.

The threshold is calibrated on an **untrained network** with random weights. During training, the weight distribution changes substantially. The learned weights produce different input currents to the hidden neurons, which changes the firing rate -- even with the same threshold. As a result, the **achieved activity level during and after training** drifts away from the nominal target.

The pilot results show this clearly:

```
Nominal target | Achieved activity (mean over 3 seeds)
0.01 (1%)      | 0.654   <- should be ~0.01, actually 0.654
0.10 (10%)     | 0.443   <- should be ~0.10, actually 0.443
0.20 (20%)     | 0.367   <- should be ~0.20, actually 0.367
0.40 (40%)     | 0.357   <- should be ~0.40, actually 0.357
0.80 (80%)     | 0.462   <- should be ~0.80, actually 0.462
```

The achieved activities cluster in the range 0.35-0.65, far from the nominal targets. The extreme conditions (1% and 80%) converge to similar achieved activities (0.654 and 0.462). This means the experiment did not successfully manipulate the firing rate over the intended range.

**Consequence for analysis.** If you key the analysis on the achieved activity (the actual firing rate), the effect of sparsity on forgetting disappears: $r(\text{achieved activity}, \text{forgetting}) = -0.068$ (essentially zero). The achieved activities are too similar across conditions to show a clean relationship.

**What was successfully manipulated.** The firing threshold $\vartheta$ was successfully varied across conditions. The threshold is the cleanly manipulated variable, even though the achieved activity is not. The threshold still produces different forgetting outcomes (forgetting ranges from 0.026 to 0.319), and the PCA overlap still correlates strongly with forgetting ($r = -0.873$). The mechanism is visible, but the causal chain runs through the threshold, not through a precisely controlled activity level.

### The Cosine-vs-PCA Disagreement

A second complication: cosine overlap and PCA overlap give opposite correlations with forgetting.

- PCA overlap vs. forgetting: $r = -0.873$ (higher overlap -> more forgetting, as predicted)
- Cosine overlap vs. forgetting: $r = +0.756$ (higher cosine similarity -> less forgetting, opposite to prediction)

This disagreement is real and informative. The mean representations (captured by cosine overlap) move in a direction that is not predictive of forgetting. The principal subspaces (captured by PCA overlap) are predictive. This suggests that the interference mechanism operates at the level of the representational subspace, not the mean activation direction.

One interpretation: tasks with similar mean representations may be "semantically close" (e.g., both involve curved strokes), and semantic closeness may help rather than hurt retention (positive transfer). The PCA subspace captures the directions of variation within each task, which is more directly related to the weight updates that cause interference.

### The Reframed Claim

Given these complications, the honest claim supported by the pilot is:

> "Increasing the LIF firing threshold (suppressing dense activity) reduced catastrophic forgetting on Split-MNIST under naive sequential training, co-varying with reduced PCA-subspace overlap. The cleanly manipulated variable is the spike threshold, not a precisely controlled activity level."

What this claim does NOT say:
- SNNs solve catastrophic forgetting (they do not; forgetting is reduced, not eliminated)
- Sparsity always helps (it hurts accuracy at extreme sparsity)
- LIF generalizes to all SNN architectures (it may not)
- The proxy metric (PCA overlap) equals hardware energy efficiency (it does not)
- The mechanism is proven (it is supported by correlation, not experimental manipulation of the mediator)

### Experimental Conditions Summary

```
Condition | Nominal target | Achieved activity | Accuracy | Forgetting | PCA overlap
----------|---------------|-------------------|----------|------------|------------
1         | 0.01          | 0.654             | 0.942    | 0.068      | 0.701
2         | 0.10 (BEST)   | 0.443             | 0.977    | 0.026      | 0.676
3         | 0.20          | 0.367             | 0.952    | 0.056      | 0.647
4         | 0.40          | 0.357             | 0.876    | 0.149      | 0.594
5         | 0.80 (WORST)  | 0.462             | 0.735    | 0.319      | 0.526
```

All values are means over 3 seeds. The pilot uses >= 3 seeds; the full study will use >= 5 seeds.

### The Full-Study Expansion Ladder

The pilot establishes the basic phenomenon. The full study expands along four dimensions:

**More baselines:**
- MLP (dense, no spikes): establishes whether the effect is SNN-specific
- ConvNet (dense): establishes whether the effect is architecture-specific
- Conv-SNN (convolutional LIF): establishes whether the effect generalizes within SNNs

**More CL methods:**
- Replay: does sparsity add benefit on top of replay?
- EWC (Kirkpatrick et al., 2017): does sparsity complement Fisher-weighted regularization?
- SI (Zenke et al., 2017): synaptic intelligence comparison
- LwF (Li & Hoiem, 2017): knowledge distillation comparison
- PackNet (Mallya & Lazebnik, 2018): structured sparsity comparison

**More datasets:**
- Permuted-MNIST: same digits, different pixel permutations per task (tests input-level separation)
- Split-CIFAR-10: more complex images, tests generalization beyond MNIST

**More sparsity mechanisms:**
- Winner-Take-All (WTA): only the top-k neurons fire per time step (hard sparsity)
- Activity regularization: L1 penalty on firing rates (soft sparsity)
- Comparison with threshold-controlled sparsity (this pilot)

## Core Concepts

- **Nominal vs. achieved activity:** the target firing rate vs. the actual firing rate after training
- **Calibration confound:** threshold calibrated on untrained network; achieved activity drifts during training
- **Reframed claim:** threshold is the cleanly manipulated variable, not activity level
- **Expansion ladder:** systematic extension to more baselines, methods, datasets, mechanisms

## What / Why / When / How

**What** is the pilot? A minimal experiment to establish the basic phenomenon and identify complications before committing to the full study.

**Why** use Split-MNIST? It is fast, well-understood, and exhibits clear forgetting. It is the right scale for a pilot.

**When** do you run the pilot? Before the full study, to identify methodological issues (like the calibration confound) that would otherwise invalidate the full study.

**How** do you handle the calibration confound in the full study? Options include: (a) calibrate on a partially trained network; (b) use online threshold adaptation to maintain a target activity level during training; (c) accept the threshold as the independent variable and report achieved activity as a covariate.

## Advantages and Disadvantages

**Advantages of the pilot design:**
- Fast iteration (Split-MNIST trains in minutes)
- Clear ground truth (MNIST is well-understood)
- Reveals methodological issues early

**Disadvantages:**
- Split-MNIST is too simple to generalize to real-world CL
- The calibration confound limits the precision of the activity manipulation
- 3 seeds is insufficient for reliable statistics (full study uses >= 5)

## Historical Context

The use of Split-MNIST as a CL benchmark dates to Zenke et al. (2017) and has been standard since. Its simplicity is both a strength (fast, interpretable) and a weakness (may not reflect real-world complexity). The calibration problem for threshold-controlled sparsity in SNNs is not widely discussed in the literature, making it a genuine methodological contribution of this work.

## Comparison

| Benchmark | Tasks | Complexity | Training time | Forgetting severity |
|-----------|-------|-----------|---------------|---------------------|
| Split-MNIST | 5 | Low | Minutes | High (naive) |
| Permuted-MNIST | 10+ | Low | Hours | Moderate |
| Split-CIFAR-10 | 5 | Medium | Hours | High |
| Split-CIFAR-100 | 20 | High | Days | Very high |

## Visual Intuition

The calibration confound:

```
Nominal target: 1%                    Nominal target: 80%
                                      
Untrained net:  fires at 1%           Untrained net:  fires at 80%
                |                                     |
                | training                            | training
                v                                     v
Trained net:    fires at 65%          Trained net:    fires at 46%
                                      
Both converge toward 50% range!
The threshold manipulation is real, but the activity manipulation is not.
```

The expansion ladder:

```
PILOT (this work)
  |-- Split-MNIST
  |-- LIF-SNN only
  |-- Naive sequential only
  |-- Threshold-controlled sparsity only
  |-- 3 seeds
  
FULL STUDY
  |-- + Permuted-MNIST, Split-CIFAR
  |-- + MLP, ConvNet, Conv-SNN baselines
  |-- + Replay, EWC, SI, LwF, PackNet
  |-- + WTA, activity regularization
  |-- + >= 5 seeds
```

## Mathematical Foundations

**Bisection calibration.** Let $r(\vartheta)$ be the mean firing rate as a function of threshold $\vartheta$. We want $r(\vartheta^*) = r_{\text{target}}$. Bisection:

1. Initialize $\vartheta_{\text{lo}} = 0$, $\vartheta_{\text{hi}} = \vartheta_{\text{max}}$.
2. Set $\vartheta_{\text{mid}} = (\vartheta_{\text{lo}} + \vartheta_{\text{hi}}) / 2$.
3. If $r(\vartheta_{\text{mid}}) > r_{\text{target}}$, set $\vartheta_{\text{lo}} = \vartheta_{\text{mid}}$; else set $\vartheta_{\text{hi}} = \vartheta_{\text{mid}}$.
4. Repeat until $|r(\vartheta_{\text{mid}}) - r_{\text{target}}| < \epsilon$.

This converges in $O(\log(\vartheta_{\text{max}} / \epsilon))$ steps.

**Achieved activity drift.** Let $r_0(\vartheta)$ be the firing rate on the untrained network and $r_t(\vartheta)$ be the firing rate after $t$ training steps. The calibration confound is:

$$r_t(\vartheta^*) \neq r_0(\vartheta^*) = r_{\text{target}}$$

The drift $\Delta r = r_t(\vartheta^*) - r_0(\vartheta^*)$ is not controlled and varies across conditions.

## Implementation Notes

- Use a fixed random seed for weight initialization and data shuffling. Record the seed.
- Calibrate the threshold on a fresh forward pass with a representative batch (e.g., 1000 examples from task 1).
- After training, measure the achieved activity on the same batch and report it alongside the nominal target.
- For the full study, consider online threshold adaptation: after each epoch, adjust $\vartheta$ to bring the firing rate closer to the target. This reduces (but does not eliminate) the calibration confound.
- Use >= 5 seeds for the full study. Report mean and standard deviation across seeds.

## Examples

**Simple example.** Calibration for 10% target:
- Untrained net with $\vartheta = 0.5$: firing rate = 45%
- Increase $\vartheta$ to 1.0: firing rate = 12%
- Increase $\vartheta$ to 1.5: firing rate = 8%
- Set $\vartheta = 1.2$: firing rate = 10% (target achieved)
- After training: firing rate = 44% (drift due to weight changes)

**Worked example from the pilot.** The nominal target 0.10 condition achieves the best performance: accuracy 0.977, forgetting 0.026, PCA overlap 0.676. The nominal target 0.80 condition achieves the worst: accuracy 0.735, forgetting 0.319, PCA overlap 0.526. Despite the calibration confound (achieved activities 0.443 and 0.462 are similar), the forgetting difference is large (0.293). This suggests the threshold itself -- not the achieved activity -- is the active variable.

## Current Research

Recent SNN continual-learning work (e.g., Hebbian orthogonal-projection and sparse-pathway methods) has explored related ideas but has not systematically addressed the calibration confound. Online threshold adaptation methods from the SNN literature (homeostatic plasticity) could be adapted to maintain a target activity level during training, which would resolve the confound in future work.

## References

- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*.
- Li, Z., & Hoiem, D. (2017). Learning without forgetting. *TPAMI*, 40(12), 2935-2947.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

---

<a name="chapter-6"></a>
# Chapter 6: Statistics for This Study

## Overview

Statistical rigor is what separates a compelling result from an anecdote. This chapter covers the specific statistical tools used in this study: sample size and seed requirements, descriptive statistics, effect sizes, hypothesis tests, multiple-comparison correction, regression for the inverted-U, bootstrap confidence intervals, and correlation analysis for the mechanism links. Each tool is explained from first principles, then applied to the pilot data.

## Fundamental Theory

The goal of statistical analysis is to distinguish signal from noise. In this study, the "signal" is the effect of the firing threshold on forgetting and representational overlap. The "noise" is the variability across random seeds (different weight initializations, different data orderings). With only 3 seeds in the pilot, the noise estimate is unreliable; the full study uses >= 5 seeds to improve it.

Statistical inference answers the question: "If there were no true effect, how likely would we be to observe a result at least as extreme as what we saw?" A small p-value means the result is unlikely under the null hypothesis of no effect. But p-values alone are insufficient; effect sizes tell you how large the effect is, and confidence intervals tell you the range of plausible true values.

## Technical Explanation

### Sample Size and Seeds

Each experimental condition (nominal activity target) is run with >= 3 seeds in the pilot and >= 5 seeds in the full study. A "seed" here means a different random initialization of the network weights and a different random ordering of the training data. The seed controls all sources of randomness in the experiment.

With 5 conditions and 3 seeds each, the pilot has 15 data points total. This is sufficient for exploratory correlation analysis but not for reliable hypothesis testing. The full study with 5 seeds per condition has 25 data points per comparison, which is marginal but acceptable for the primary comparisons.

### Descriptive Statistics

For each condition, report:
- Mean across seeds: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$
- Standard deviation: $s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2}$
- Report as: $\bar{x} \pm s$

The standard deviation (not standard error) is reported because it describes the variability of individual runs, which is what a practitioner would experience when reproducing the experiment.

### Effect Sizes: Cohen's d

Cohen's d measures the standardized difference between two conditions:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

where $s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$.

Interpretation (Cohen, 1988):
- $|d| < 0.2$: negligible
- $0.2 \leq |d| < 0.5$: small
- $0.5 \leq |d| < 0.8$: medium
- $|d| \geq 0.8$: large

Effect sizes are reported alongside p-values because a statistically significant result with a tiny effect size is not practically meaningful.

### Hypothesis Tests

**Paired t-test.** Used when comparing two conditions across the same seeds (e.g., forgetting at threshold $\vartheta_1$ vs. $\vartheta_2$, with the same 5 seeds in both conditions). The pairing removes between-seed variability.

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where $\bar{d}$ is the mean of the pairwise differences and $s_d$ is their standard deviation. Degrees of freedom: $n - 1$.

**Wilcoxon signed-rank test.** A non-parametric alternative to the paired t-test, used when the normality assumption is questionable (which it often is with $n = 5$ seeds). The Wilcoxon test ranks the absolute differences and tests whether the signed ranks are symmetric around zero.

For the pilot (3 seeds), neither test has adequate power. Report effect sizes and confidence intervals instead of p-values for the pilot; reserve hypothesis testing for the full study.

### Multiple-Comparison Correction

When testing multiple hypotheses simultaneously, the probability of at least one false positive increases. With $m$ tests at significance level $\alpha = 0.05$, the expected number of false positives is $m \times 0.05$.

**Holm-Bonferroni correction (Holm, 1979).** A step-down procedure that is more powerful than the Bonferroni correction while still controlling the family-wise error rate (FWER). Procedure:

1. Sort the $m$ p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$.
2. For $k = 1, 2, \ldots, m$: reject $H_{(k)}$ if $p_{(k)} \leq \alpha / (m - k + 1)$.
3. Stop at the first non-rejection; all subsequent hypotheses are retained.

**Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995).** Controls the false discovery rate (FDR) -- the expected proportion of false positives among all rejections -- rather than the FWER. More powerful than Holm-Bonferroni when many hypotheses are tested.

1. Sort p-values: $p_{(1)} \leq \ldots \leq p_{(m)}$.
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$.
3. Reject all $H_{(1)}, \ldots, H_{(k)}$.

In this study, Holm-Bonferroni is used for the primary comparisons (few hypotheses, FWER control desired) and Benjamini-Hochberg is used for the exploratory correlation analyses (many hypotheses, FDR control acceptable).

### Quadratic Regression for the Inverted-U

To test the inverted-U hypothesis, fit a quadratic regression of net performance (or forgetting) on the threshold (or nominal activity target):

$$y = a x^2 + b x + c + \epsilon, \quad a < 0 \text{ for inverted-U}$$

The vertex (optimal threshold) is:

$$x^* = -\frac{b}{2a}$$

**Interior-peak condition.** The inverted-U claim requires that $x^*$ lies strictly inside the tested range $[x_{\min}, x_{\max}]$:

$$x_{\min} < x^* < x_{\max}$$

If $x^*$ falls outside this range, the data are consistent with a monotone relationship (the quadratic fit is just approximating a line), and the inverted-U claim is not supported. This is a hard requirement, not a soft preference.

**Reporting.** Report $a$, $b$, $c$, $x^*$, the 95% confidence interval for $x^*$ (via bootstrap), and whether the interior-peak condition is satisfied.

### Bootstrap Confidence Intervals

Bootstrap CIs are used when the sampling distribution of a statistic is unknown or non-normal (e.g., the vertex $x^*$ of the quadratic fit, or the indirect effect in mediation analysis).

Procedure:
1. Resample the data with replacement $B = 10000$ times.
2. Compute the statistic of interest on each resample.
3. The 95% CI is the 2.5th and 97.5th percentiles of the bootstrap distribution.

For the pilot (3 seeds), bootstrap CIs are very wide and should be interpreted cautiously. They are reported for completeness and to motivate the larger sample size in the full study.

### Correlation Analysis for Mechanism Links

Pearson correlation $r$ measures the linear relationship between two variables:

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}$$

$r \in [-1, 1]$: $r = 1$ is perfect positive correlation, $r = -1$ is perfect negative correlation, $r = 0$ is no linear relationship.

For the mechanism links, the key correlations are:
- $r(\text{PCA overlap}, \text{forgetting})$: should be positive (higher overlap -> more forgetting)
- $r(\text{cosine overlap}, \text{forgetting})$: predicted positive, observed negative in pilot
- $r(\text{achieved activity}, \text{forgetting})$: should be positive (higher activity -> more forgetting)

Significance of $r$ is tested with a t-test: $t = r\sqrt{(n-2)/(1-r^2)}$ with $n-2$ degrees of freedom. With $n = 15$ runs, $|r| > 0.514$ is significant at $\alpha = 0.05$ (two-tailed).

## Core Concepts

- **Seed:** a random initialization; the unit of replication in this study
- **Cohen's d:** standardized effect size; $|d| \geq 0.8$ is large
- **Holm-Bonferroni:** step-down FWER correction; more powerful than Bonferroni
- **Benjamini-Hochberg:** FDR correction; more powerful than Holm for many tests
- **Interior-peak condition:** $x^* \in (x_{\min}, x_{\max})$; required for inverted-U claim
- **Bootstrap CI:** non-parametric confidence interval; appropriate for non-normal statistics

## What / Why / When / How

**What** statistics are used? Descriptive (mean, SD), effect sizes (Cohen's d), hypothesis tests (paired t-test, Wilcoxon), multiple-comparison correction (Holm-Bonferroni, BH-FDR), quadratic regression (inverted-U), bootstrap CIs, Pearson correlation.

**Why** use multiple statistical tools? Each tool answers a different question. P-values answer "is this likely by chance?" Effect sizes answer "how large is the effect?" CIs answer "what is the plausible range?" Regression answers "what is the shape of the relationship?"

**When** do you apply multiple-comparison correction? Whenever you test more than one hypothesis in the same study. In this study, the primary comparisons (best vs. worst condition) are corrected with Holm-Bonferroni; the exploratory correlations are corrected with BH-FDR.

**How** do you report results? Follow the APA format: $t(df) = t\text{-value}$, $p = p\text{-value}$, $d = \text{Cohen's d}$, 95% CI $[\text{lower}, \text{upper}]$.

## Advantages and Disadvantages

**Advantages of this statistical approach:**
- Effect sizes provide practical significance beyond p-values
- Multiple-comparison correction reduces false positives
- Bootstrap CIs are valid without normality assumptions
- Interior-peak condition prevents overclaiming the inverted-U

**Disadvantages:**
- Small sample size (3-5 seeds) limits power
- Pearson correlation assumes linearity; the true relationship may be non-linear
- Bootstrap CIs are wide with small samples
- The interior-peak condition may not be satisfied if the tested range is too narrow

## Historical Context

Cohen (1988) established the conventions for effect size interpretation that are now standard in psychology and increasingly in machine learning. Holm (1979) introduced the step-down correction that bears his name. Benjamini and Hochberg (1995) introduced FDR control, which has become the standard in genomics and is increasingly used in ML. The use of bootstrap CIs in ML papers is growing but not yet universal.

## Comparison

| Method | Controls | Power | Assumption |
|--------|----------|-------|------------|
| Bonferroni | FWER | Low | None |
| Holm-Bonferroni | FWER | Medium | None |
| Benjamini-Hochberg | FDR | High | Independence (approx.) |
| Paired t-test | -- | High | Normality, paired |
| Wilcoxon | -- | Medium | Symmetry |

## Visual Intuition

The inverted-U with interior-peak condition:

```
  Accuracy
     |
 1.0 |          * <- vertex x* (interior peak)
     |        *   *
 0.9 |      *       *
     |    *           *
 0.8 |  *               *
     +--+--+--+--+--+--+---> Threshold
        ^                 ^
        x_min             x_max
        
  x* must be strictly between x_min and x_max.
  If x* is at or beyond x_max, the data show a monotone decrease,
  not an inverted-U.
```

The Holm-Bonferroni procedure:

```
Sorted p-values: p(1)=0.001, p(2)=0.012, p(3)=0.045, p(4)=0.089
m = 4, alpha = 0.05

k=1: threshold = 0.05/4 = 0.0125. p(1)=0.001 < 0.0125 -> REJECT
k=2: threshold = 0.05/3 = 0.0167. p(2)=0.012 < 0.0167 -> REJECT
k=3: threshold = 0.05/2 = 0.025.  p(3)=0.045 > 0.025  -> RETAIN (stop)
k=4: (stopped) -> RETAIN
```

## Mathematical Foundations

**Cohen's d:**

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

**Paired t-test:**

$$t = \frac{\bar{d}}{s_d/\sqrt{n}}, \quad df = n-1$$

**Quadratic regression vertex:**

$$y = ax^2 + bx + c, \quad x^* = -\frac{b}{2a}, \quad \text{interior-peak: } x_{\min} < x^* < x_{\max}$$

**Pearson correlation:**

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2 \sum_i (y_i-\bar{y})^2}}, \quad t = r\sqrt{\frac{n-2}{1-r^2}}, \quad df = n-2$$

**Holm-Bonferroni:** reject $H_{(k)}$ if $p_{(k)} \leq \alpha/(m-k+1)$, stopping at first non-rejection.

**Benjamini-Hochberg:** reject $H_{(1)}, \ldots, H_{(k^*)}$ where $k^* = \max\{k: p_{(k)} \leq k\alpha/m\}$.

## Implementation Notes

- Use `scipy.stats.ttest_rel` for paired t-tests and `scipy.stats.wilcoxon` for Wilcoxon tests.
- Use `statsmodels.stats.multitest.multipletests` for Holm-Bonferroni and BH-FDR correction.
- For quadratic regression, use `numpy.polyfit(x, y, 2)` and check the sign of the leading coefficient.
- For bootstrap CIs, use `numpy.random.choice` with `replace=True` and repeat 10000 times.
- Always check the interior-peak condition before claiming an inverted-U. Report $x^*$ and its CI.
- With 3 seeds, do not report p-values as primary evidence. Report effect sizes and CIs.

## Examples

**Simple example.** Two conditions, 5 seeds each:

```
Condition A (sparse): forgetting = [0.02, 0.03, 0.04, 0.02, 0.03]
Condition B (dense):  forgetting = [0.28, 0.32, 0.30, 0.29, 0.31]

mean_A = 0.028, sd_A = 0.0084
mean_B = 0.300, sd_B = 0.0158

s_pooled = sqrt((4*0.0084^2 + 4*0.0158^2) / 8) = 0.0130
d = (0.028 - 0.300) / 0.0130 = -20.9  (very large effect)

Paired differences: [-0.26, -0.29, -0.26, -0.27, -0.28]
mean_d = -0.272, sd_d = 0.0130
t = -0.272 / (0.0130/sqrt(5)) = -46.8, df=4, p << 0.001
```

**Worked example from the pilot.** Across 15 runs (5 conditions x 3 seeds):

- PCA overlap vs. forgetting: $r = -0.873$. With $n = 15$: $t = -0.873 \times \sqrt{13/0.238} = -6.17$, $p < 0.001$. This is significant even without multiple-comparison correction.
- Cosine overlap vs. forgetting: $r = +0.756$. $t = 0.756 \times \sqrt{13/0.428} = 4.23$, $p = 0.001$. Also significant, but in the opposite direction.
- Achieved activity vs. forgetting: $r = -0.068$. $t = -0.068 \times \sqrt{13/0.995} = -0.245$, $p = 0.81$. Not significant; no relationship.

The quadratic regression of accuracy on nominal target (coded as 1, 2, 3, 4, 5):

```
Accuracy: [0.942, 0.977, 0.952, 0.876, 0.735]
Fit: a = -0.0185, b = 0.0895, c = 0.875
x* = -0.0895 / (2 * -0.0185) = 2.42

Nominal target 2.42 corresponds to between target 0.10 and 0.20.
x_min = 1, x_max = 5.
x* = 2.42 is interior -> inverted-U condition satisfied for accuracy.
```

## Current Research

The machine learning community has increasingly adopted rigorous statistical practices following critiques of p-value misuse (e.g., Bouthillier et al., 2021, on reproducibility in deep learning). Effect sizes and confidence intervals are now expected in top venues. The SNN literature lags behind in statistical rigor, making this study's approach a methodological contribution in itself.

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

---

---

<a name="capstone"></a>
# Capstone: Putting It All Together

This final section traces the complete path from the LIF neuron to the reframed empirical claim, connecting every concept in this guide into a single coherent narrative.

## The Full Path

**Step 1: The LIF neuron and the threshold.**
A Leaky Integrate-and-Fire neuron accumulates input current and fires a spike when its membrane potential crosses a threshold $\vartheta$. Raising $\vartheta$ makes the neuron harder to fire. In a hidden layer of 256 LIF neurons, raising $\vartheta$ uniformly suppresses the mean firing rate -- fewer neurons fire for any given input. This is the starting point: a single scalar hyperparameter that controls the sparsity of the network's internal representations.

**Step 2: Sparsity and representational separation.**
When only a small fraction of neurons fire for any given input, different inputs tend to activate different subsets of neurons. Two tasks trained sequentially -- say, digits 0/1 and digits 2/3 -- will activate partially overlapping subsets if the network is dense, and more disjoint subsets if the network is sparse. The PCA subspace overlap metric captures this: it measures the alignment of the principal directions of variation in the hidden representations of the two tasks. Lower overlap means more disjoint subspaces.

**Step 3: Representational overlap and catastrophic forgetting.**
When task B is trained, gradient updates flow through the active neurons and modify the weights. If task A and task B share active neurons, the updates for task B will modify the weights that task A depends on, causing forgetting. If the active sets are disjoint, task B's updates cannot reach task A's weights. The accuracy matrix captures the outcome: $F_j = A[j,j] - A[T,j]$ measures how much task $j$'s performance drops after all subsequent tasks are trained.

**Step 4: The competing cost and the inverted-U.**
Raising $\vartheta$ too high leaves too few active neurons to represent any task well. The network's capacity is reduced. This creates a tradeoff: moderate sparsity reduces forgetting without sacrificing capacity; extreme sparsity reduces forgetting but also reduces accuracy. The result is an inverted-U relationship between the threshold and net performance, with an interior optimum at $\vartheta^* = -b/2a$ (from the quadratic fit).

**Step 5: The pilot experiment and its complications.**
The Split-MNIST pilot tested five nominal activity targets (1%, 10%, 20%, 40%, 80%) with a LIF-SNN trained sequentially on five binary classification tasks. The threshold was calibrated by bisection on the untrained network. The key complication: the achieved activity drifted during training, clustering in the 0.35-0.65 range regardless of the nominal target. The threshold was successfully varied; the activity level was not.

**Step 6: The pilot results.**
Despite the calibration confound, the results show a clear pattern. The best condition (nominal target 10%, achieved activity 0.443) achieved accuracy 0.977 and forgetting 0.026. The worst condition (nominal target 80%, achieved activity 0.462) achieved accuracy 0.735 and forgetting 0.319. PCA subspace overlap correlated strongly with forgetting ($r = -0.873$), supporting the mechanism. Cosine overlap correlated in the opposite direction ($r = +0.756$), suggesting that mean representations are not the right level of analysis. Achieved activity showed no relationship with forgetting ($r = -0.068$), confirming the calibration confound.

**Step 7: The reframed claim.**
The honest, defensible claim is: "Increasing the LIF firing threshold (suppressing dense activity) reduced catastrophic forgetting on Split-MNIST under naive sequential training, co-varying with reduced PCA-subspace overlap. The cleanly manipulated variable is the spike threshold, not a precisely controlled activity level."

This claim is supported. It does not overclaim. It does not say SNNs solve forgetting, or that sparsity always helps, or that the mechanism is proven. It says: we varied the threshold, we observed reduced forgetting, and we observed reduced PCA overlap, and these two observations are consistent with the proposed mechanism.

**Step 8: The path forward.**
The full study will address the calibration confound (online threshold adaptation), add confound controls (matched mastery, count-matched baseline), expand to more datasets and architectures, and apply rigorous statistics (>= 5 seeds, effect sizes, Holm-Bonferroni correction, interior-peak test for the inverted-U). The goal is to move from "consistent with the mechanism" to "the mechanism is supported by multiple converging lines of evidence."

## What You Can Now Do

Having read all three parts of this guide, you can:

1. **Understand** the LIF neuron model, surrogate gradients, and spike sparsity (Part 2).
2. **Understand** the continual learning problem and why naive sequential training causes forgetting (Part 1).
3. **Understand** the causal chain from threshold to forgetting, and the competing capacity cost (Chapter 1).
4. **Measure** forgetting using the accuracy matrix, per-task forgetting, BWT, and FWT (Chapter 2).
5. **Measure** representational overlap using PCA subspace overlap and cosine similarity, and know when NOT to use CKA (Chapter 3).
6. **Control** for the capacity and plasticity confounds (Chapter 4).
7. **Reproduce** the pilot experiment and understand its complications (Chapter 5).
8. **Analyze** results with appropriate statistics: effect sizes, multiple-comparison correction, quadratic regression with interior-peak test, bootstrap CIs, and Pearson correlation (Chapter 6).
9. **Extend** the study along the expansion ladder: more baselines, methods, datasets, and sparsity mechanisms.

---

<a name="glossary"></a>
# Glossary

**Accuracy matrix** $A[i,j]$: The accuracy on task $j$ after training on task $i$. The lower triangle (including diagonal) is the primary data structure for continual learning evaluation.

**Achieved activity**: The actual mean firing rate of the hidden layer during or after training, as opposed to the nominal target activity used for threshold calibration.

**Backward transfer (BWT)**: $\frac{1}{T-1}\sum_{j=1}^{T-1}(A[T,j] - A[j,j])$. Negative when forgetting occurs.

**Benjamini-Hochberg (BH) correction**: A procedure for controlling the false discovery rate (FDR) when testing multiple hypotheses. More powerful than Holm-Bonferroni when many tests are performed.

**Bisection calibration**: An iterative procedure for finding the threshold $\vartheta$ that achieves a target firing rate, by repeatedly halving the search interval.

**Capacity confound**: The alternative explanation that a sparser network forgets less because it learned less of the first task, not because it protects old memories.

**Catastrophic forgetting**: The tendency of a neural network to lose performance on previously learned tasks when trained on new tasks, due to overwriting of shared weights.

**Centered Kernel Alignment (CKA)**: A similarity measure for neural network representations based on HSIC. Requires matched inputs (same rows in both representation matrices). Degenerate for cross-task comparison with disjoint inputs.

**Cohen's d**: A standardized effect size measure: $d = (\bar{x}_1 - \bar{x}_2) / s_{\text{pooled}}$. Conventions: small $|d| \geq 0.2$, medium $|d| \geq 0.5$, large $|d| \geq 0.8$.

**Cosine overlap**: The cosine similarity between the mean hidden representations of two tasks. Fast but coarse; only compares the centers of the representation clouds.

**False discovery rate (FDR)**: The expected proportion of false positives among all rejected hypotheses. Controlled by the Benjamini-Hochberg procedure.

**Family-wise error rate (FWER)**: The probability of at least one false positive among all tested hypotheses. Controlled by Bonferroni and Holm-Bonferroni procedures.

**Final average accuracy (ACC)**: $\frac{1}{T}\sum_{j=1}^T A[T,j]$. The primary performance metric in continual learning.

**Forward transfer (FWT)**: $\frac{1}{T-1}\sum_{j=2}^T (A[j-1,j] - b_j)$. Measures how earlier tasks help later tasks.

**HSIC (Hilbert-Schmidt Independence Criterion)**: A measure of statistical dependence between two random variables, used as the basis for CKA.

**Interior-peak condition**: The requirement that the vertex $x^* = -b/2a$ of a quadratic fit lies strictly inside the tested range $[x_{\min}, x_{\max}]$. Required for the inverted-U claim to be supported.

**Inverted-U tradeoff**: The relationship between sparsity and net performance, where moderate sparsity is optimal and both extremes (too dense, too sparse) are suboptimal.

**LIF neuron**: Leaky Integrate-and-Fire neuron. A simplified spiking neuron model that integrates input current, leaks over time, and fires a spike when the membrane potential crosses a threshold.

**Linear probe**: A linear classifier trained on top of frozen hidden representations to measure how much task-relevant information is retained in the representation layer.

**Mean forgetting** $\bar{F}$: $\frac{1}{T-1}\sum_{j=1}^{T-1} F_j$. The average per-task forgetting over all tasks except the last.

**Mediation**: A variable $M$ mediates the relationship between $X$ and $Y$ if $X$ affects $M$, $M$ affects $Y$, and the $X$-$Y$ relationship is reduced when $M$ is controlled. Representational overlap is proposed as a mediator between threshold and forgetting.

**Naive sequential training**: Training a model on tasks one after another, with no mechanism to prevent forgetting. The baseline condition in this study.

**Nominal activity target**: The target firing rate used for threshold calibration (e.g., 10% of neurons firing). May differ substantially from the achieved activity after training.

**PCA subspace overlap**: $\|U_A^T U_B\|_F^2 / k$, where $U_A$ and $U_B$ are the top-$k$ principal components of the hidden representations of tasks A and B. Ranges from 0 (orthogonal subspaces) to 1 (identical subspaces). Uses $k = 10$ in this project.

**Per-task forgetting** $F_j$: $\max_{i \leq j} A[i,j] - A[T,j]$. The drop from the best accuracy ever achieved on task $j$ to the final accuracy.

**Plasticity confound**: The alternative explanation that a sparser network forgets less because it updates fewer weights during task-B training, not because its representations are separated.

**Plasticity-stability tradeoff**: The tension between learning new tasks quickly (plasticity) and retaining old tasks (stability). Sparsity trades plasticity for stability.

**Reframed claim**: The honest, defensible claim supported by the pilot: "Increasing the LIF firing threshold reduced catastrophic forgetting on Split-MNIST, co-varying with reduced PCA-subspace overlap. The cleanly manipulated variable is the spike threshold, not a precisely controlled activity level."

**Row-pairing requirement**: The requirement that row $i$ of representation matrix $X$ and row $i$ of representation matrix $Y$ correspond to the same input. Required for CKA to be well-defined.

**Seed**: A random initialization of the network weights and data ordering. The unit of replication in this study.

**Split-MNIST**: A continual learning benchmark that divides MNIST into five sequential binary classification tasks (0/1, 2/3, 4/5, 6/7, 8/9).

**Stability-plasticity tradeoff**: See plasticity-stability tradeoff.

**Surrogate gradient**: A smooth approximation to the derivative of the spike function, used to enable backpropagation through the non-differentiable spike threshold.

**Task-incremental setting**: A continual learning setting where the task identity is provided at test time, allowing the model to use the appropriate output head.

**Threshold** $\vartheta$: The membrane potential at which a LIF neuron fires a spike. The primary experimental manipulation in this study.

**Wilcoxon signed-rank test**: A non-parametric alternative to the paired t-test, used when normality cannot be assumed.

---

<a name="further-reading"></a>
# Further Reading

## Continual Learning

- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *International Conference on Machine Learning (ICML)*.
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12), 2935-2947.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

## Representational Similarity

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *International Conference on Machine Learning (ICML)*.
- Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. *Advances in Neural Information Processing Systems (NeurIPS)*.

## Statistics

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289-300.

## Spiking Neural Networks

- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.
- Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*. Cambridge University Press. (Available free online at neuronaldynamics.epfl.ch)
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51-63.

## Biological Motivation

- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419-457.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

---

*End of Companion Learning Guide -- Part 3.*
*Parts 1 and 2 cover deep learning foundations, continual learning background, spiking neural networks, the LIF model, surrogate gradients, and spike sparsity.*

