# Learning Resources to Execute This Research

A curated, verified list of resources to go from beginner to research-level on
everything needed to run this SNN + continual-learning study. All links were
checked as real and reachable.

The five knowledge areas you need:
1. Deep learning + PyTorch foundations (MLP, backprop, Adam, cross-entropy, MNIST)
2. Spiking Neural Networks + snntorch (LIF, surrogate gradients, encoding, T=25)
3. Continual Learning + catastrophic forgetting (Split-MNIST, EWC/SI/replay, forgetting metrics)
4. Representation-overlap metrics (CKA, PCA subspace overlap, linear probes)
5. Statistics for the study (effect sizes, paired tests, corrections, bootstrap)

## Suggested learning order (the critical path)
1. DL/PyTorch foundations
2. Spiking neural networks + snntorch
3. Continual learning + catastrophic forgetting
4. Representation-overlap metrics
5. Statistics for the analysis

---

## 1. Deep Learning + PyTorch Foundations

### YouTube / video
- 3Blue1Brown - Neural Networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi - best intuition for backprop/gradients.
- Andrej Karpathy - Neural Networks: Zero to Hero: https://karpathy.ai/zero-to-hero.html - build nets + backprop from scratch in code.
- fast.ai - Practical Deep Learning: https://course.fast.ai/ - learn-by-doing.

### Docs / tutorials
- PyTorch Tutorials: https://pytorch.org/tutorials/
- PyTorch 60-Minute Blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html - tensors, autograd, first image classifier.
- Stanford CS231n notes: https://cs231n.stanford.edu/

### GitHub
- pytorch/examples: https://github.com/pytorch/examples - includes the MNIST classifier (your exact baseline shape).
- pytorch/tutorials: https://github.com/pytorch/tutorials

### MOOCs
- DeepLearning.AI Deep Learning Specialization: https://www.coursera.org/specializations/deep-learning
- fast.ai: https://course.fast.ai/

### University courses
- Stanford CS231n: https://cs231n.stanford.edu/
- NYU Deep Learning (LeCun / Canziani): https://atcold.github.io/

### People to follow
Andrej Karpathy (https://karpathy.ai/), Andrew Ng (https://www.deeplearning.ai/), Yann LeCun, Jeremy Howard.

---

## 2. Spiking Neural Networks + snntorch (your core tooling)

### YouTube / video
- snnTorch intro - Jason Eshraghian (ICONS 2021): https://www.youtube.com/watch?v=O2-mT291ygg - from the library's author.
- Hands-on with snnTorch - Eshraghian (UCSC): https://www.youtube.com/watch?v=aUjWRpisRRg - practical Colab walkthrough.
- Training SNNs Using Lessons From Deep Learning: https://www.youtube.com/watch?v=zldal7b7sJ4 - surrogate gradients / BPTT.
- Cosyne 2022 SNN Tutorial: https://www.youtube.com/watch?v=GTXTQ_sOxak - deeper theory.

### Docs / tutorials (primary learning path)
- snnTorch tutorials index: https://snntorch.readthedocs.io/en/latest/tutorials/index.html
- Tut 1: Spike Encoding (rate/latency/delta): https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html
- Tut 2: The LIF Neuron: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html
- Tut 5: Training SNNs (BPTT + surrogate): https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
- Tut 6: Surrogate Gradient in a Conv-SNN: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
- snntorch.surrogate reference (ATan, fast_sigmoid - you use fast_sigmoid): https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
- snn.Leaky reference (exact LIF constructor you use): https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html

### GitHub
- snnTorch: https://github.com/jeshraghian/snntorch - your library.
- Zenke's spytorch: https://github.com/fzenke/spytorch - classic surrogate-gradient reference.
- Norse: https://github.com/norse/norse
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- BindsNET: https://github.com/BindsNET/bindsnet
- Tonic (event datasets): https://github.com/neuromorphs/tonic

### LIF theory (the beta = exp(-dt/tau) foundation)
- Gerstner "Neuronal Dynamics" - free book + lectures (EPFL): https://neuronaldynamics.epfl.ch/lectures.html - gold standard for LIF, spike trains, coding.
- EPFL Biological Modeling of Neural Networks: https://www.epfl.ch/labs/lcn/teaching-html/nnbm/

### People to follow
Jason Eshraghian (snntorch), Wulfram Gerstner (LIF theory), Friedemann Zenke (surrogate gradients), Emre Neftci (surrogate-gradient learning), Sander Bohte.

---

## 3. Continual Learning + Catastrophic Forgetting

### YouTube / video
- ContinualAI channel: https://www.youtube.com/channel/UCD9_bqN3gX-TLxcr47vvMmA - best single CL channel.
- "Understanding Catastrophic Forgetting" lecture: https://www.youtube.com/watch?v=UnCAdBtvZhc - explicitly uses Split/Permuted-MNIST (your benchmarks).

### Blogs / written
- ContinualAI Course - Catastrophic Forgetting lecture notes: https://course.continualai.org/lectures/understanding-catastrophic-forgetting
- EWC + Synaptic Intelligence tutorial (Brenndoerfer): https://mbrenndoerfer.com/writing/regularization-methods-ewc-synaptic-intelligence-continual-learning - clear walkthrough of two methods you'll use.
- van de Ven "Three scenarios for continual learning" (task/domain/class-incremental - the backbone of Fix #3): arXiv:1904.07734.

### GitHub
- Avalanche (ContinualAI): https://github.com/ContinualAI/avalanche - benchmarks + EWC/SI/LwF/GEM/replay + forgetting metrics in one place.
  - 5-min getting started: https://avalanche.continualai.org/getting-started/learn-avalanche-in-5-minutes
- Mammoth: https://github.com/aimagelab/mammoth - large menu of CL methods + datasets.
- Sequoia: https://github.com/lebrice/Sequoia

### MOOC / course
- ContinualAI Course "On Machines that can Learn Continually": https://course.continualai.org/master.md - grad/PhD-aimed, open-access, recordings on YouTube.

### People to follow
Gido van de Ven (scenarios/taxonomy), Vincenzo Lomonaco (Avalanche), German Parisi, Tinne Tuytelaars, James Kirkpatrick (EWC), Friedemann Zenke (SI), Marc'Aurelio Ranzato (GEM).

---

## 4. Representation-Overlap Metrics (CKA, PCA, linear probes)

- Kornblith et al. 2019 "Similarity of NN Representations Revisited" (CKA):
  - Project page: https://cka-similarity.github.io/
  - PDF: http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf
  - The canonical CKA reference (already cited in the proposal).
- Google Research CKA reference code + Colab demo:
  - Repo: https://github.com/google-research/google-research/tree/master/representation_similarity
  - Demo: https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
- Linear probing mindset: covered in CS231n transfer-learning material - https://cs231n.stanford.edu/ (no single authoritative "linear probe" tutorial; the transfer-learning/feature-reuse sections teach the exact setup).

Project note: `overlap.py` deliberately disables cross-task linear CKA (disjoint
Split-MNIST inputs, no row-pairing). Study CKA to understand why it is ill-defined
there and where it is valid (matched-input comparisons), plus PCA subspace overlap
as the primary mechanism metric (the r = -0.873 signal).

---

## 5. Statistics for the Study

- StatQuest (Josh Starmer): https://statquest.org/ - clear on t-tests, effect sizes, bootstrap, multiple comparisons (covers most of the analysis plan).
- Learning Statistics with R (free book): https://learningstatisticswithr.com/ - hypothesis testing, CIs, effect sizes, regression.
- Statistical Rethinking (McElreath): https://www.statisticalrethinking.com/ - deeper modeling perspective.

Maps to the plan: Cohen's d, paired t-test/Wilcoxon, Holm-Bonferroni +
Benjamini-Hochberg FDR, bootstrap CIs, quadratic regression for the inverted-U
(H3). StatQuest alone covers most of these.

---

## Shortest high-value path (essentials only)
1. 3Blue1Brown NN playlist -> PyTorch 60-min Blitz -> pytorch/examples MNIST
2. snntorch Tutorials 1, 2, 5, 6 (+ Eshraghian's ICONS talk)
3. Gerstner "Neuronal Dynamics" (LIF chapters only)
4. ContinualAI catastrophic-forgetting lecture + EWC/SI tutorial + van de Ven scenarios paper
5. Avalanche 5-minute intro
6. CKA paper + Google Research demo
7. StatQuest (t-tests, effect sizes, bootstrap, multiple comparisons)
