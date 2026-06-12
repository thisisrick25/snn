# Related Work

## 1. Catastrophic Forgetting in Artificial Neural Networks

Catastrophic forgetting, first systematically characterized by McCloskey and Cohen [3], describes the rapid degradation of previously learned knowledge when neural networks are sequentially trained on new tasks. This phenomenon remains one of the most significant barriers to deploying neural networks in real-world continual learning scenarios. To mitigate this, researchers have proposed various strategies including regularization-based methods, architecture-based methods, and rehearsal-based methods [1, 2].

### 1.1 Regularization-Based Methods
Elastic Weight Consolidation (EWC) [4] estimates the importance of parameters for previous tasks and penalizes changes to these weights during new task training. Similarly, Synaptic Intelligence (SI) [5] estimates the importance of synapses online during training. While effective, these methods often require explicit computation of parameter importance, which can be computationally expensive and may not generalize well to complex architectures.

### 1.2 Architecture-Based Methods
Architecture-based methods such as Progressive Neural Networks [6] and dynamically expandable networks [7] solve forgetting by allocating separate resources for each task. While effective, they often suffer from scalability issues as the architecture grows with each new task, making them impractical for lifelong learning scenarios.

### 1.3 Rehearsal-Based Methods
Replay Buffer methods [8] store and replay data from previous tasks during new task training. While these methods are highly effective, they raise concerns about storage requirements and privacy, especially when original data cannot be retained.

## 2. Spiking Neural Networks and Neuromorphic Computing

Spiking Neural Networks (SNNs) have emerged as a compelling third-generation neural network paradigm that more closely mimics the dynamics of biological neurons [9, 10]. Unlike ANNs, which use continuous-valued activations, SNNs communicate via discrete spikes, resulting in inherently sparse and event-driven computation.

### 2.1 Biological Plausibility and Sparse Activity
Biological neurons exhibit sparse activity patterns, with only a small fraction of neurons firing in response to any given stimulus [11, 12]. This sparsity is not just an emergent property but appears to be functionally important for memory consolidation and interference reduction [13, 14]. SNNs, by design, naturally manifest similar sparse activity patterns [15].

### 2.2 Energy Efficiency of SNNs
The event-driven nature of SNNs leads to significantly lower energy consumption compared to ANNs. Studies have shown that SNNs can achieve up to orders of magnitude lower energy consumption on neuromorphic hardware such as Intel's Loihi and IBM's TrueNorth [16, 17]. This makes them particularly attractive for edge computing and mobile applications.

### 2.3 Training SNNs
Training SNNs has historically been challenging due to the non-differentiable nature of the spike generation function. However, recent advances such as surrogate gradient methods [18] and frameworks like snnTorch [19] have made training deep SNNs more accessible, enabling their application to complex tasks such as image classification and speech recognition.

## 3. Continual Learning in Spiking Neural Networks

While SNNs have shown promise for various tasks, their application to continual learning remains underexplored. A few key works have begun to bridge this gap.

### 3.1 SNN-Specific Approaches
Some studies have proposed SNN-specific continual learning approaches, such as spike-timing-dependent plasticity (STDP) based methods [20] and neuromorphic replay mechanisms [21]. These approaches leverage the temporal dynamics of SNNs to improve memory retention.

### 3.2 Transferring ANN Methods to SNNs
Other works have focused on adapting existing ANN continual learning methods (like EWC and SI) to the SNN domain [22, 23]. While these adaptations show some success, they often do not fully exploit the unique properties of SNNs, such as their natural sparsity.

## 4. Sparsity and Representation Learning

The relationship between sparsity and learning has been extensively studied in the context of ANNs.

### 4.1 Sparse Representations in ANNs
Sparse representations, often induced via L1 regularization or dropout, have been shown to improve generalization and reduce overfitting in ANNs [24, 25]. The Lottery Ticket Hypothesis [26] further suggests that sparse subnetworks can be found within dense networks that are capable of training effectively. However, the direct impact of sparsity on catastrophic forgetting in the context of continual learning has not been systematically studied.

### 4.2 Sparsity and Interference
Theoretical work suggests that sparse representations can reduce interference between different tasks by activating non-overlapping sets of neurons [27, 28]. This principle, known as "sparse distributed memory," is biologically motivated and suggests that sparsity may play a crucial role in protecting against catastrophic forgetting.

## 5. Research Gap

Despite the significant progress in both continual learning and spiking neural networks, the direct relationship between spike sparsity and catastrophic forgetting remains unexplored. Existing SNN continual learning work focuses on architectural innovations or adapting ANN techniques, but none have systematically investigated how manipulating sparsity affects forgetting. Our work aims to fill this gap by providing the first empirical study on this topic, showing that sparsity is not just a side effect of SNN dynamics but a tunable property that can be leveraged to improve continual learning performance.
