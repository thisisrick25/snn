# Methodology

## 1. Overview and Pipeline

Our methodology is designed to systematically isolate and measure the effect of spike sparsity on catastrophic forgetting. The pipeline consists of three main components:

1. **Sparsity Control**: Mechanisms to precisely vary spike sparsity.
2. **Continual Learning Protocols**: Standard and modified training procedures.
3. **Evaluation**: Metrics capturing accuracy, forgetting, and energy efficiency.

## 2. Models and Architecture

### 2.1 ANN Baseline
To establish a performance reference, we use a Multi-Layer Perceptron (MLP) as our ANN baseline.
- **Architecture**: Input layer → Hidden layer(s) → Output layer.
- **Activation**: ReLU non-linearity.
- **Output**: Softmax for classification.

### 2.2 SNN Baseline
We use a Spiking Neural Network based on the Leaky Integrate-and-Fire (LIF) neuron model.
- **Architecture**: Input layer → LIF Hidden layer(s) → LIF Output layer.
- **Neuron Model**: Leaky Integrate-and-Fire (LIF) as defined below.
- **Synaptic Operations**: Modeled as weighted summations of incoming spikes.

#### 2.2.1 Leaky Integrate-and-Fire (LIF) Neuron Dynamics
The membrane potential $U(t)$ of a neuron evolves according to:

$$\tau_m \frac{dU(t)}{dt} = -(U(t) - U_{rest}) + R \cdot I(t)$$

Where:
- $\tau_m$ is the membrane time constant.
- $وي- $U_{rest}$ is the resting potential.
- $R$ is the membrane resistance.
- $I(t)$ is the input current.

A spike is generated when $U(t)$ exceeds a threshold $\theta$. After spiking, the potential is reset to $U_{reset}$.

## 3. Sparsity Manipulation Mechanisms

We employ three distinct mechanisms to control the level of spike sparsity in the SNN.

### 3.1 Spike Threshold Adjustment
By increasing the firing threshold $\theta$, we directly reduce the probability of a neuron spiking.
- **Implementation**: A scaling factor is applied to the base threshold.
- **Effect**: A higher threshold leads to fewer spikes, increasing sparsity.

### 3.2 Winner-Take-All (WTA) Inhibition
This mechanism enforces sparsity by allowing only a fixed percentage of neurons in a layer to remain active after inhibition.
- **Implementation**: After computing spikes for a layer, only the top-$k$ neurons with the highest membrane potentials are allowed to fire. All others are suppressed.
- **Effect**: Direct control over the percentage of active neurons, regardless of threshold.

### 3.3 Activity Regularization
We add a regularization term to the loss function to penalize high overall neural activity.
- **Loss Function**: $\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{activity}$
- **Activity Penalty**: $\mathcal{L}_{activity} = \frac{1}{N}\sum_{i=1}^{N} s_i$, where $s_i$ is the spike count of neuron $i$ over a time window.
- **Effect**: The hyperparameter $\lambda$ controls the trade-off between task performance and sparsity.

## 4. Continual Learning Protocols

To evaluate forgetting, we employ several standard continual learning strategies.

### 4.1 Naive Sequential Learning
Tasks are presented sequentially, and the model is fine-tuned on each new task without any mechanism to prevent forgetting. This serves as a lower bound to measure the maximum extent of forgetting.

### 4.2 Replay Buffer
A fixed-size buffer stores a subset of data points from previously learned tasks. During training on a new task, the model is also trained on data sampled from this buffer, acting as a strong baseline for mitigating forgetting.

### 4.3 Elastic Weight Consolidation (EWC)
This method identifies and protects synaptic weights that are important for previous tasks. It approximates the posterior distribution of the weights given the previous task data, penalizing changes to critical weights.

### 4.4 Synaptic Intelligence (SI)
A biologically inspired method that tracks the importance of each synapse for the tasks learned so far. Unlike EWC, which relies on Fisher information, SI estimates importance online during training.

## 5. Evaluation Metrics

To comprehensively evaluate the performance of our models, we define the following metrics:

### 5.1 Task Accuracy
The average classification accuracy across all tasks learned so far, measured after training on the final task.

$$A_t = \frac{1}{T} \sum_{i=1}^{T} Acc_i$$

### 5.2 Forgetting Score
A measure of performance degradation on previous tasks.

$$F_t = \frac{1}{t-1} \sum_{i=1}^{t-1} \max_{j<i}(Acc_{i,j}) - Acc_{i,t}$$

Where $Acc_{i,j}$ is the accuracy on task $i$ after training on task $j$.

### 5.3 Spike Rate
The average number of spikes generated per neuron per time step, providing a direct measure of network activity.

### 5.4 Sparsity Index
The percentage of neurons that remain inactive (do not spike) over a given input presentation.

$$SI = \frac{\text{Number of inactive neurons}}{\text{Total number of neurons}} \times 100\%$$

### 5.5 Energy Proxy
An estimate of the computational energy cost, approximated by the product of the total spike count and the number of synaptic operations.

$$E = \sum_{t} (\text{Spike Count}_t \times \text{Synaptic Operations}_t)$$
