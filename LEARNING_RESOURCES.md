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

---

## Study Roadmap: Understanding This Experiment and Its Results

The five sections above are external links. This section is different: it maps the concepts you need to the two in-repo documents, then gives a realistic one-day plan.

The two documents:
- **COMPANION_GUIDE.md** — 15-chapter handbook, beginner to research-level, reads linearly.
- **RESEARCH_REPORT.md** — 15-section deep-dive into the study design, mechanism, and analysis plan.

---

### Part A: Topic Roadmap

These ~15 topics take you from zero to being able to understand the experimental design and interpret the pilot results. Each maps to a chapter of COMPANION_GUIDE.md and/or a section of RESEARCH_REPORT.md.

#### A.1 Topics to understand the experiment (the design)

| # | Topic | Where to read |
|---|-------|---------------|
| 1 | Neural network fundamentals — MLP, backprop, Adam, cross-entropy | COMPANION_GUIDE Ch.1 |
| 2 | MNIST family and Split-MNIST construction | COMPANION_GUIDE Ch.2 |
| 3 | Continual learning settings — task- / domain- / class-incremental | COMPANION_GUIDE Ch.3 + RESEARCH_REPORT §1 |
| 4 | Catastrophic forgetting and the stability-plasticity dilemma | COMPANION_GUIDE Ch.4 |
| 5 | Spiking neurons and the LIF model (beta = exp(-1/tau), membrane/threshold/reset dynamics) | COMPANION_GUIDE Ch.5-6 + RESEARCH_REPORT §2 |
| 6 | Encoding and time in SNNs — rate coding, choosing the number of timesteps T | COMPANION_GUIDE Ch.7 |
| 7 | Surrogate-gradient training — why spikes are non-differentiable and how fast-sigmoid gets around it | COMPANION_GUIDE Ch.8 |
| 8 | Spike sparsity and the threshold mechanism — the active-neuron metric, calibration, calibration drift, and the ~38% activity ceiling of this architecture | COMPANION_GUIDE Ch.9 + RESEARCH_REPORT §6 |

#### A.2 Topics to interpret the results

| # | Topic | Where to read |
|---|-------|---------------|
| 9 | The central hypothesis and mechanism — sparsity -> less representational overlap -> less forgetting | COMPANION_GUIDE Ch.10 + RESEARCH_REPORT §§3-4 |
| 10 | Continual-learning metrics — the accuracy matrix, forgetting score, backward transfer | COMPANION_GUIDE Ch.11 + RESEARCH_REPORT §8 |
| 11 | Representational-overlap metrics — cosine similarity vs CKA, why centering matters (this explains the cosine ~0.95 vs CKA ~0.02 gap seen in the pilot) | COMPANION_GUIDE Ch.12 |
| 12 | Mediation vs correlation — path a, path b, the indirect effect a*b, bootstrap CI; this is the study's actual novelty | RESEARCH_REPORT §5 + RESEARCH_IDEA_REFINED §3.5 |
| 13 | Confound controls — capacity vs plasticity vs genuine sparse-coding, and the six controls | COMPANION_GUIDE Ch.13 + RESEARCH_REPORT §6 |
| 14 | The inverted-U / quadratic hypothesis and the interior-peak requirement (H3) | RESEARCH_REPORT §4 |
| 15 | Statistics for the study — seeds, effect sizes, why the pilot is screening-only and not confirmatory | COMPANION_GUIDE Ch.15 |

---

**Reading orders:**

- **FAST path** (enough to interpret the current pilot results): topics 3 -> 4 -> 5 -> 8 -> 10 -> 11 -> 9.
- **FULL path**: COMPANION_GUIDE Ch.1-15 in order, then RESEARCH_REPORT end to end.

---

**The three most load-bearing concepts for this pilot:**

- **CKA and centering (Topic 11)** — explains the cosine-vs-CKA gap in the results. Cosine similarity and linear CKA are both computed as mean pairwise overlap across the five Split-MNIST task representations (hidden-layer-2 spike counts); they diverge because CKA centers the representations first.
- **Threshold -> activity -> the ~38% ceiling (Topic 8)** — explains the whole calibration saga and why targets were redefined to ~1-35%. The architecture hits a hard ceiling before you reach the high-sparsity regime the hypothesis needs.
- **Mediation is not correlation (Topic 12)** — the line between this being novel work and a rediscovery of existing sparse-SNN results. CKA tracking forgetting across conditions and seeds is the mechanism signal; whether that path is causal is what the mediation model tests.

---

### Part B: What's Achievable in a Single Day

Reading everything in a day is possible. Genuinely understanding all 15 topics is not — that is roughly a two-day effort. The FAST path alone is enough to interpret the pilot results, and it fits in one focused day (~6-8 hours).

**One-day plan (FAST path):**

| Block | Time | Topics |
|-------|------|--------|
| Morning 1 | ~1.5h | Continual-learning settings (topic 3) + catastrophic forgetting (topic 4) |
| Morning 2 | ~2h | The LIF neuron (topic 5) + spike sparsity / threshold / activity metric / ~38% ceiling (topic 8) |
| Afternoon 1 | ~1h | CL metrics — accuracy matrix, forgetting score, backward transfer (topic 10) |
| Afternoon 2 | ~1.5h | Overlap metrics — cosine vs CKA and centering (topic 11); this is the one that explains the cosine ~0.95 vs CKA ~0.02 result |
| Afternoon 3 | ~1h | The central mechanism — sparsity -> less overlap -> less forgetting (topic 9) |

---

**Save these three for a second day (do not rush them):**

- Surrogate-gradient training (topic 7)
- Mediation vs correlation, the formal model and bootstrap CI (topic 12) — the novelty; it deserves real time
- The full confound-control logic and the six controls (topic 13)

---

One focused day on the FAST path is enough to read and interpret the pilot the moment it finishes. The rigor layer — surrogate gradients, mediation statistics, confound design — is what you need before the full study and the paper. That is a second day, not today's bottleneck.
