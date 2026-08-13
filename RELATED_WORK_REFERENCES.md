# Related work reference citations

This file collects papers related to the research idea: investigating whether spike sparsity reduces catastrophic forgetting in continual learning spiking neural networks (SNNs). It is organized by how each reference supports the paper.

Some recent 2025/2026 preprints were found through web search with incomplete metadata. Those entries are marked **verify before final citation**.

## 1. Closest prior work: SNN continual learning with sparsity, thresholds, gating, or spike budgets

These papers are the most important for positioning the novelty of this project. They show that sparse activation, threshold modulation, spike budgets, and gating have already been used in SNN continual learning. The contribution of this project should therefore be framed as a controlled causal study of spike sparsity, representation overlap, and forgetting.

1. Shen, J., Ni, W., Xu, Q., & Tang, H. (2024). Efficient Spiking Neural Networks with Sparse Selective Activation for Continual Learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 38*(1), 611-619. https://doi.org/10.1609/aaai.v38i1.27817
   - Relevance: Closest prior work. Uses trace-based K-Winner-Take-All and variable thresholds to create sparse selective activation for continual learning in SNNs.

2. Hammouamri, I., Masquelier, T., & Wilson, D. G. (n.d.). Mitigating Catastrophic Forgetting in Spiking Neural Networks through Threshold Modulation. *OpenReview*. https://openreview.net/forum?id=15SoThZmtU
   - Relevance: Directly related to the threshold-control part of this project. Uses firing-threshold modulation to reduce forgetting in SNNs.

3. Meem, A. T., Nadid, M. H., & Mia, M. Z. A. (2026). Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks for Neuromorphic Vision. *arXiv:2602.12236*. https://arxiv.org/abs/2602.12236
   - Relevance: Directly related to spike-rate control, sparsity as regularization, and energy-aware continual learning in SNNs. Combines replay, learnable LIF neurons, and an adaptive spike scheduler (energy-aware spike budgeting).
   - Status: Provisionally verified (2026-02-12). Confirm on the arXiv abstract page before final citation.

4. Active Dendrites Enable Efficient Continual Learning in Time-To-First-Spike Neural Networks. (2024). *arXiv:2404.19419*. https://arxiv.org/abs/2404.19419
   - Relevance: Uses highly sparse time-to-first-spike coding and task-specific gating/subnetwork behavior to reduce forgetting.
   - Status: Verify full author list before final citation.

5. TACOS: Task Agnostic Continual Learning in Spiking Neural Networks. (2024). *arXiv:2409.00021*. https://arxiv.org/abs/2409.00021
   - Relevance: SNN continual learning method using synaptic consolidation, metaplasticity, and neuromodulation without explicit task awareness.
   - Status: Verify full author list before final citation.

6. Roy, Kobayashi, Chakraborty, Talukder, & Alam. (2026). Gradient-Free Continual Learning in Spiking Neural Networks via Inter-Spike Interval Regularization. *arXiv:2604.16496*. https://arxiv.org/abs/2604.16496
   - Relevance: Uses the coefficient of variation of inter-spike intervals (ISI-CV) as a gradient-free measure of synaptic importance, relevant to spike timing and firing regularity as protection signals.
   - Status: Provisionally verified (2026-04-14). Confirm full author list on the arXiv abstract page before final citation.

7. Zhang, Cao, Jiang, Du, Yu, Lv, & Deng. (2025). SAFA-SNN: Sparsity-Aware On-Device Few-Shot Class-Incremental Learning with Fast-Adaptive Structure of Spiking Neural Network. *arXiv:2510.03648*. https://arxiv.org/abs/2510.03648
   - Relevance: Sparsity-aware dynamics and fast-adaptive structure for few-shot class-incremental learning in SNNs.
   - Status: Provisionally verified title. Confirm full author list on the arXiv abstract page before final citation.

8. Nagabhushana, Agrawal, & Borthakur. (2026). CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds. *arXiv:2603.15184*. https://arxiv.org/abs/2603.15184
   - Relevance: Combines continual learning with spiking transformers using dynamic-threshold LIF neurons to prevent forgetting.
   - Status: Provisionally verified title. Confirm full author list on the arXiv abstract page before final citation.

9. SD2-SNN: Self-Distillation and Structural Decomposition Framework for SNNs in Continual Learning. (2026). *Neural Networks / ScienceDirect result*.
   - Relevance: Reported to combine sparse encoding, self-distillation, structural decomposition, and sparse activation subspace consistency.
   - Status: Verify full bibliographic metadata before final citation.

10. Astrocyte-gated multi-timescale plasticity for online continual learning in deep spiking neural networks. (2025/2026). *Frontiers in Neuroscience*.
    - Relevance: Biologically inspired SNN continual learning method using multi-timescale gating/plasticity.
    - Status: Verify full author list, year, and DOI before final citation.

## 2. Foundational continual learning references

These papers establish the standard continual learning problem, methods, and evaluation context.

11. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of Learning and Motivation* (Vol. 24, pp. 109-165). Academic Press.
    - Relevance: Foundational source for catastrophic interference in sequential neural-network learning.

12. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521-3526. https://doi.org/10.1073/pnas.1611835114
    - Relevance: Introduces Elastic Weight Consolidation (EWC), a core regularization baseline.

13. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *Proceedings of the 34th International Conference on Machine Learning*, 3987-3995.
    - Relevance: Introduces Synaptic Intelligence (SI), a core continual learning baseline.

14. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems, 30*.
    - Relevance: Important replay/projection baseline and source for common forgetting measurements.

15. Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 40*(12), 2935-2947.
    - Relevance: Standard distillation-based continual learning baseline.

16. Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 7765-7773.
    - Relevance: Parameter-isolation baseline relevant to sparsity and task-specific capacity.

17. Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020). Orthogonal gradient descent for continual learning. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics*, 3762-3773.
    - Relevance: Interference-reduction method based on orthogonalizing gradient updates.

18. Saha, G., Garg, I., & Roy, K. (2021). Gradient projection memory for continual learning. *International Conference on Learning Representations*.
    - Relevance: Uses task subspaces to reduce interference, important for the representational-overlap mechanism.

19. Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks, 113*, 54-71. https://doi.org/10.1016/j.neunet.2019.01.012
    - Relevance: Standard survey for continual learning methods and problem framing.

20. van de Ven, G. M., & Tolias, A. S. (2019). Three scenarios for continual learning. *arXiv:1904.07734*.
    - Relevance: Defines task-incremental, domain-incremental, and class-incremental scenarios.

## 3. Mechanism references: sparsity, representation overlap, subspaces, and interference

These papers support the proposed causal chain: spike sparsity reduces representational overlap, which reduces interference and forgetting.

21. Ramasesh, V. V., Dyer, E., & Raghu, M. (2020). Anatomy of catastrophic forgetting: Hidden representations and task semantics. *International Conference on Learning Representations*.
    - Relevance: Studies forgetting through hidden representations, task semantics, and subspace similarity.

22. Kaushik, P., Kortylewski, A., Gain, A., & Yuille, A. (2021). Understanding catastrophic forgetting and remembering in continual learning with optimal relevance mapping. https://www.cs.jhu.edu/~alanlab/Pubs21/kaushik2021understanding.pdf
    - Relevance: Introduces the idea of optimal representational overlap for balancing forgetting and remembering.
    - Status: Verify venue before final citation.

23. Doan, T., Bennani, M. A., Mazoure, B., Rabusseau, G., & Alquier, P. (2021). A theoretical analysis of catastrophic forgetting through the NTK overlap matrix. *Proceedings of Machine Learning Research*. https://proceedings.mlr.press/v130/doan21a/doan21a.pdf
    - Relevance: Treats task overlap/similarity as central to forgetting; useful for PCA/subspace-overlap motivation.

24. Abbasi, A., et al. (2022). Sparsity and heterogeneous dropout for continual learning. *Proceedings of Machine Learning Research*. https://proceedings.mlr.press/v199/abbasi22a/abbasi22a.pdf
    - Relevance: Uses k-winner sparse activations and heterogeneous dropout to reduce overlap between task representations.
    - Status: Verify full author list and exact title before final citation.

25. Hu, X., et al. (2024). Task-aware orthogonal sparse network for exploring shared knowledge in continual learning. *Proceedings of the 41st International Conference on Machine Learning*. https://proceedings.mlr.press/v235/hu24b.html
    - Relevance: Uses orthogonal sparse network partitioning to reduce interference and share useful knowledge.
    - Status: Verify full author list before final citation.

26. Kim, J., Kim, Y., & Sohn, K. (n.d.). Measuring representational shifts in continual learning: A linear transformation perspective. *OpenReview*.
    - Relevance: Provides representation-shift metrics relevant to representation drift and forgetting.
    - Status: Verify venue/year before final citation.

27. Keep Moving: Identifying task-relevant subspaces to maximise plasticity for newly learned tasks. (2023). *arXiv:2310.04741*. https://arxiv.org/abs/2310.04741
    - Relevance: Separates task-relevant subspaces from plasticity-preserving subspaces, useful for mechanism analysis.
    - Status: Verify full author list before final citation.

28. Low-coherence Subspace Projection for Continual Learning. (n.d.). *OpenReview*.
    - Relevance: Uses low-coherence rather than strictly orthogonal subspaces to reduce task interference while preserving capacity.
    - Status: Verify bibliographic metadata before final citation.

29. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *Proceedings of the 36th International Conference on Machine Learning*, 3519-3529.
    - Relevance: Standard reference for CKA-style representation similarity.

30. Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. *Advances in Neural Information Processing Systems, 30*.
    - Relevance: Standard representation comparison method, useful as an alternative to cosine similarity/PCA overlap.

## 4. Spike sparsity, sparse firing, and energy-efficiency references

These papers support the spike sparsity controls, sparse firing regularization, and energy-efficiency discussion.

31. Yan, Z., Bai, Z., & Wong, W.-F. (2024). Reconsidering the energy efficiency of spiking neural networks. *arXiv:2409.08290*. https://arxiv.org/abs/2409.08290
    - Relevance: Cautions that SNN energy efficiency depends on time window, sparsity, memory access, and data movement; important for avoiding simplistic energy claims.

32. Sparse-firing regularization methods for spiking neural networks with time-to-first-spike coding. (2023). *Scientific Reports*. https://www.nature.com/articles/s41598-023-50201-5
    - Relevance: Directly studies sparse-firing regularization in TTFS-coded SNNs.
    - Status: Verify full author list before final citation.

33. Backpropagation with sparsity regularization for spiking neural network learning. (2022). *Frontiers in Neuroscience*. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.760298/full
    - Relevance: Introduces spiking and synaptic sparsity regularization during SNN training.
    - Status: Verify full author list before final citation.

34. Optimizing the energy consumption of spiking neural networks for neuromorphic applications. (2020). *Frontiers in Neuroscience*. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full
    - Relevance: Uses loss terms to reduce activations/synaptic operations for SNN energy efficiency.
    - Status: Verify full author list before final citation.

35. High-performance deep spiking neural networks with 0.3 spikes per neuron. (2024). *Nature Communications*. https://www.nature.com/articles/s41467-024-51110-5
    - Relevance: Shows that high-performance SNNs can operate with very sparse spiking.
    - Status: Verify full author list before final citation.

36. Wei, C., Duan, B., Guo, C., Zhang, J., Song, Q., Li, H., & Chen, Y. (2025). Phi: Leveraging pattern-based hierarchical sparsity for high-efficiency spiking neural networks. *arXiv:2505.10909*. https://arxiv.org/abs/2505.10909
    - Relevance: Exploits hierarchical sparsity in SNN activations for hardware efficiency.

37. Shapero, S., Charles, A. S., Rozell, C. J., & Hasler, P. (2017). Low power sparse approximation on reconfigurable analog hardware. *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*.
    - Relevance: Related to spiking/sparse coding and low-power sparse approximation.
    - Status: The search also surfaced arXiv:1705.05475 on spiking LCA and sparse coding. Verify the exact citation intended before final use.

38. Yin, H., et al. (2021). Energy-efficient models for high-dimensional spike train classification using sparse spiking neural networks. *NSF Public Access Repository*.
    - Relevance: Studies sparse spatiotemporal coding for resource-efficient SNNs.
    - Status: Verify full author list and venue before final citation.

## 5. Biological sparse coding and motivation references

These sources motivate the biological plausibility of sparse activity. They should be used carefully as motivation, not as proof that the simplified SNN model is biologically faithful.

39. Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature, 381*, 607-609. https://doi.org/10.1038/381607a0
    - Relevance: Foundational sparse coding reference.

40. Buzsaki, G. (2006). *Rhythms of the Brain*. Oxford University Press.
    - Relevance: Broad neuroscience background on neural rhythms and temporal neural activity.

41. Softky, W. R., & Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs. *Journal of Neuroscience, 13*(1), 334-350.
    - Relevance: Useful for discussion of irregular firing, spike timing, and inter-spike intervals.

42. Lennie, P. (2003). The cost of cortical computation. *Current Biology, 13*(6), 493-497. https://doi.org/10.1016/S0960-9822(03)00135-0
    - Relevance: Classic biological energy-constraint reference.

## 6. Must-read shortlist

Start with these before revising the related work section:

1. Shen et al. (2024), Efficient SNNs with Sparse Selective Activation for Continual Learning
2. Hammouamri, Masquelier, and Wilson, Threshold Modulation for Catastrophic Forgetting in SNNs
3. Meem et al. (2026), Energy-Aware Spike Budgeting for Continual Learning in SNNs, arXiv:2602.12236
4. Active Dendrites Enable Efficient Continual Learning in TTFS SNNs, arXiv:2404.19419
5. Ramasesh et al. (2020), Anatomy of Catastrophic Forgetting
6. Kaushik et al. (2021), Optimal Relevance Mapping / optimal representational overlap
7. Abbasi et al. (2022), sparse activations and heterogeneous dropout for continual learning
8. Kirkpatrick et al. (2017), EWC
9. Zenke et al. (2017), SI
10. Yan, Bai, and Wong (2024), Reconsidering SNN energy efficiency

## 7. Framing note for the paper

The literature suggests that the novelty should not be framed as simply "sparse SNNs reduce forgetting." Prior work already uses sparse activation, threshold modulation, gating, and spike budgets for SNN continual learning.

A stronger framing is:

> Prior SNN continual-learning work uses sparse activation, threshold modulation, spike budgeting, or gating to reduce forgetting, but the causal relationship between controlled spike sparsity levels, representational overlap, and catastrophic forgetting remains under-characterized. This study systematically varies spike sparsity across multiple control mechanisms and tests whether reduced representational overlap mediates the sparsity-forgetting relationship.
