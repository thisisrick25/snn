# Investigating the Relationship Between Spike Sparsity and Catastrophic Forgetting in Continual Learning Spiking Neural Networks

**Author:** Graduate Research Project  
**Date:** July 2026  
**Status:** Pilot complete — three-model progression: (a) flatten-MLP on Split-MNIST and CIFAR (§11.8–11.9); (b) Conv-SNN on CIFAR (§11.10, the fair capacity-pressure test); (c) combined cross-architecture conclusions. H2 (extreme sparsity hurts accuracy) is now supported, visible only with the competent Conv-SNN on CIFAR (~10-point penalty at ~1% activity vs. the ~0.71 plateau). H3 (inverted-U sweet-spot) is not supported on any model or benchmark — the real pattern is a saturating curve, not an interior peak. H4 (representational-overlap mediation of forgetting) is not supported across two architectures and two benchmarks (four negative screens total, including a Conv-SNN CIFAR mediation with a-path running in the wrong direction). All pilot results are exploratory (3 seeds, not pre-registered); full study pending with 8–10 seeds, wider decoupled activity range, and multi-mechanism design  
**Software stack:** Python 3.x, PyTorch 2.12.0, torchvision 0.27.0, snntorch 0.9.4, numpy 2.4.6, scikit-learn 1.9.0, matplotlib 3.11.0, pandas 3.0.3, scipy 1.17.1

---

## Abstract

Catastrophic forgetting — the tendency of neural networks to lose performance on earlier tasks when trained on new ones — remains one of the central unsolved problems in machine learning. Biological neural systems, by contrast, learn over long timescales while maintaining sparse, event-driven activity patterns. This project investigates whether spike sparsity in spiking neural networks (SNNs) reduces catastrophic forgetting in continual learning settings, and whether that reduction is explained by a specific mechanism: lower representational overlap between sequentially learned task representations. The central claim is a mediation hypothesis — sparsity reduces forgetting *through* reduced overlap, not merely alongside it — which distinguishes this work from prior SNN continual learning studies that use sparse activation as a tool without characterizing the causal pathway. The study proceeds in two stages. A pilot experiment on Split-MNIST with threshold-controlled sparsity and three random seeds screens the correlation precondition: does moderate sparsity (roughly 20–40% active neurons) associate with lower overlap and lower forgetting? If the precondition holds, a full study runs a formal mediation model across three sparsity mechanisms (spike threshold, winner-take-all, activity regularization), two continual learning settings (task-incremental and class-incremental), and multiple CL methods (naive, replay, EWC, SI, LwF), with 8–10 confirmatory seeds and a pre-registered confirmatory test hierarchy. The novelty survives only if the indirect effect of sparsity on forgetting through overlap is significantly nonzero after conditioning on confounds; a significant total effect with a nonsignificant indirect effect would indicate that sparsity acts through reduced plasticity or capacity rather than through representational separation, which would substantially weaken the contribution. The pilot has since run a three-model progression — flatten-MLP on Split-MNIST and CIFAR, then a Conv-SNN on CIFAR as the fair capacity-pressure test — whose combined findings are: H2 (extreme sparsity hurts accuracy) is supported but only visible with a competent model; H3 (the inverted-U sweet-spot) is not supported on any model, with the real pattern being a saturating curve rather than an interior peak; and H4 (overlap-mediation of forgetting) is not supported across two architectures and two benchmarks (four negative screens). These are exploratory pilot results at three seeds and redirect the full study rather than settling the questions.

**Framing update (post-pilot).** The pilot proceeded in three stages that together form a model progression. First, a fixed-window k-winner-take-all (k-WTA) gate replaced the frozen firing threshold, which had proved unable to hold activity at a target level in a trained network (Section 11.8). On the resulting controlled activity axis (roughly 1-33% active neurons on Split-MNIST with a flatten-MLP), the predicted inverted-U did **not** appear: accuracy rose and forgetting fell monotonically, with no interior optimum. H3 is **not supported on Split-MNIST**. A subsequent exploratory mediation analysis on those 18 k-WTA conditions (Section 11.9) found no evidence that representational overlap (CKA) mediates the sparsity-forgetting relationship beyond activity co-variation: the indirect effect $a \times b$ was +0.128 with a 95% bootstrap CI of [-0.586, +0.994], which includes zero, and the b-path coefficient was weakly negative rather than the positive value the hypothesis requires. H4 is **not supported on Split-MNIST** -- the first two negative screens.

Second, the flatten-MLP was tested on CIFAR-10, where it reached only approximately 0.62 accuracy and was nearly flat across the activity axis. A model that cannot learn the task cannot create real capacity pressure, so this was an unfair test of H3. Third, a Conv-SNN with a three-layer spiking convolutional frontend (channels 3 to 16 to 32 to 64, k-WTA and activity metric on the two FC hidden layers only) was run on CIFAR-10 (Section 11.10). The Conv-SNN is genuinely competent, reaching approximately 0.71 final average accuracy -- a real nine-point lift over the flatten-MLP. On this competent model, H2 (extreme sparsity hurts accuracy) is **now clearly supported**: accuracy at approximately 1% activity is 0.609 versus a plateau near 0.71, a ten-point penalty the weak MLP could not reveal. H3 remains **not supported**: the curve rises then saturates, with no interior peak. H4 is **not supported** for a third and fourth time: forgetting is flat across the activity axis, and a formal exploratory mediation on the Conv-SNN CIFAR data (n = 18) gives total effect $c = -0.003$, a-path $= +0.815$ (positive -- more activity associates with *more* overlap, the opposite of the hypothesis), indirect effect $a \times b = +0.030$ with 95% bootstrap CI [-0.973, +0.729].

The pilot has thus produced four negative screens for H4 across two architectures and two benchmarks, and has established that the real sparsity-accuracy pattern is a saturating curve rather than an inverted-U. These are not refutations of the underlying idea; they are the expected output of underpowered screening runs (three seeds, exploratory, not pre-registered). The contribution of the pilot is a rigorously controlled activity axis, a three-model progression that fairly tests the hypotheses at different capacity levels, and four honest negative screens that redirect the full study: more seeds (8-10), a wider and better-decoupled activity range, and a multi-mechanism design where the full mediation model can be estimated with adequate precision. The overlap-mediation mechanism remains the central hypothesis; it is simply a hypothesis the full study must test, not a finding the pilot has established.

---

## 1. Problem Background

### 1.1 The Stability-Plasticity Dilemma

Any learning system that must acquire knowledge over time faces a fundamental tension. To learn new information, the system must modify its internal representations — it must be *plastic*. But modifying those representations risks destroying knowledge already encoded — it must also be *stable*. Grossberg (1987) named this the stability-plasticity dilemma, and it has since become the organizing concept for both computational neuroscience and machine learning research on sequential learning.

In artificial neural networks, the dilemma manifests acutely. A network trained on a sequence of tasks with shared parameters will, under naive gradient descent, rapidly lose performance on earlier tasks as it adapts to later ones. McCloskey and Cohen (1989) documented this phenomenon in connectionist networks and called it catastrophic interference — the word "catastrophic" chosen deliberately to contrast with the gradual, partial forgetting seen in human memory. The severity is striking: a network that achieves 99% accuracy on Task 1 can fall to near-chance performance on that same task after training on Task 2, even when the tasks are superficially similar.

### 1.2 Formal Setup

Let there be $T$ tasks indexed $t = 1, \ldots, T$. Each task $t$ is associated with a data distribution $p_t(x, y)$ over inputs $x$ and labels $y$. The learner processes tasks sequentially: it trains on data from $p_1$, then $p_2$, and so on, without free access to earlier data once a new task begins. The ideal continual learning objective after all $T$ tasks is:

$$\mathcal{L}_{\text{CL}}(\theta) = \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}_t(\theta)$$

where $\mathcal{L}_t(\theta) = \mathbb{E}_{(x,y) \sim p_t}[\ell(f_\theta(x), y)]$ is the expected loss on task $t$. Naive sequential training minimizes only $\mathcal{L}_T(\theta)$ at the final step, ignoring all previous tasks. The gradient update at step $t$:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_t(\theta)$$

moves parameters in a direction that reduces loss on task $t$ but carries no information about tasks $1, \ldots, t-1$. If the gradient of $\mathcal{L}_t$ is not orthogonal to the gradient of $\mathcal{L}_{t'}$ for $t' < t$, the update will increase loss on earlier tasks. Over many gradient steps, this accumulates into catastrophic forgetting.

The standard forgetting metric formalizes this. Let $\text{Acc}_{t'}(\theta)$ denote accuracy on task $t'$ with parameters $\theta$, and let $\theta_K$ denote the final parameters after training on all $K$ tasks. The mean forgetting is:

$$\mathcal{F} = \frac{1}{K-1} \sum_{t=1}^{K-1} \left[ \max_{t' \leq t} \text{Acc}_{t'}(\theta_{t'}) - \text{Acc}_{t'}(\theta_K) \right]$$

The inner term captures the drop from the best accuracy ever achieved on task $t'$ to the accuracy at the end of training. This is the primary outcome variable throughout this project.

### 1.3 Why Shared Weights Cause Forgetting

The mechanism is straightforward once stated. After training on Task 1, the parameters $\theta_1$ encode a solution: the weights are configured so that the network correctly classifies inputs from $p_1$. When training begins on Task 2, the optimizer computes gradients of $\mathcal{L}_{T_2}$ with respect to $\theta$ and updates accordingly. These gradients have no information about $\mathcal{L}_{T_1}$. If the direction that reduces $\mathcal{L}_{T_2}$ also increases $\mathcal{L}_{T_1}$ — which happens whenever the two loss landscapes conflict — the update harms Task 1 performance. After ten epochs of Task 2 training, the parameters may be far from any region of parameter space that solves Task 1.

The gradient interference can be quantified. Let $g_1 = \nabla_\theta \mathcal{L}_{T_1}(\theta)$ and $g_2 = \nabla_\theta \mathcal{L}_{T_2}(\theta)$. The update $\theta \leftarrow \theta - \eta g_2$ increases Task 1's loss by approximately $\eta \langle g_1, g_2 \rangle$. When $\langle g_1, g_2 \rangle < 0$, the update helps Task 2 but hurts Task 1. Gradient Episodic Memory (Lopez-Paz and Ranzato, 2017) explicitly constrains updates to satisfy $\langle g_1, g_2 \rangle \geq 0$, which is one principled approach to the problem. The present project takes a different angle: rather than constraining the gradient, it asks whether the network's activation structure can naturally reduce the degree of gradient conflict.

### 1.4 Continual Learning Settings

The severity of forgetting depends strongly on what information is available at test time and how the output space is structured. Van de Ven and Tolias (2019) identified three canonical scenarios that have since become the standard taxonomy.

**Task-incremental learning (Task-IL)** provides the task identity $t$ at both training and test time. The model can therefore use separate output heads — one per task — and route each test input to the correct head using the provided label. Forgetting is bounded to the shared body (feature extractor), because the per-task output heads are never overwritten by subsequent task training. This is the easiest scenario and the primary setting for this project's mechanism analysis.

**Domain-incremental learning (Domain-IL)** withholds the task identity at test time but keeps the output space fixed across tasks. The model must classify correctly without knowing which domain the input comes from. Permuted-MNIST is the canonical example: each task applies a different pixel permutation, but the label set is always digits 0–9.

**Class-incremental learning (Class-IL)** is the hardest scenario. The task identity is withheld at test time, and new classes are added with each task. The model must maintain a single classifier over all classes seen so far, implicitly identifying which task an input belongs to and classifying it correctly — all without any task label. For Split-MNIST in the class-incremental setting, the output head grows from 2 neurons after Task 1 to 10 neurons after Task 5, and the model must discriminate among all 10 digits at test time.

The structural reason Task-IL bounds forgetting while Class-IL does not is worth stating precisely. In Task-IL with per-task heads, the output layer weights $W_{\text{head},t}$ for task $t$ are never updated after task $t$ completes. Only the shared body weights are subject to gradient updates from later tasks, and even there, the gradient signal is more focused because the task-specific head provides a clean loss signal. In Class-IL, the output layer is a single growing matrix $W_{\text{head}} \in \mathbb{R}^{C_t \times d}$ where $C_t$ grows with $t$. Training on task $t$ updates all rows of $W_{\text{head}}$ (or at least the rows for new classes), directly overwriting the decision boundaries for old classes. This is a structural source of forgetting that exists even if the shared body is perfectly preserved. Class-IL forgetting is therefore structurally unbounded in a way that Task-IL forgetting is not.

This project evaluates both settings. Task-IL is the primary setting for mechanism analysis because it isolates forgetting in the shared body, which is where the sparsity mechanism is hypothesized to operate. Class-IL is the harder test: if the sparsity effect holds there, the result is substantially more convincing.


---

## 2. Spiking Neural Networks

### 2.1 Biological Motivation

The brain computes using electrical pulses called action potentials or spikes. Each spike is a stereotyped, all-or-nothing event: a neuron either fires or it does not. This binary, event-driven communication is radically different from the continuous, always-on activations used in conventional artificial neural networks. The membrane potential $V_m(t)$ — the voltage across the cell membrane — integrates incoming signals over time. When it crosses a threshold, the neuron emits a spike and resets. The information transmitted is not in the amplitude of the spike (which is fixed) but in the timing and rate of spikes.

Two features of biological neural activity are particularly relevant here. First, cortical neurons are sparse: at any given moment, only a small fraction of neurons in a given area are active. Estimates from Lennie (2003) suggest that fewer than 1–4% of cortical neurons are active simultaneously, a constraint driven partly by metabolic cost. Olshausen and Field (1996) showed that sparse coding of natural images produces basis functions resembling the receptive fields of simple cells in primary visual cortex, suggesting that sparsity is not merely an energy constraint but a representational strategy. Second, biological systems learn over long timescales without catastrophic forgetting — a property that has motivated decades of research into whether the computational properties of spiking neurons, including their sparse activity, might contribute to this robustness.

The present project treats biological sparsity as motivation for the hypothesis, not as evidence that the simplified SNN model is biologically realistic. The sparsity controls used here — threshold tuning, winner-take-all selection, activity regularization — are computational approximations of the mechanisms that produce sparse activity in biology (inhibitory circuits, homeostatic plasticity, metabolic constraints). They are not direct models of those mechanisms.

### 2.2 The Leaky Integrate-and-Fire Neuron

The leaky integrate-and-fire (LIF) model is the workhorse of modern SNN research. It captures the two most essential features of a biological neuron — integration of input over time and threshold-triggered spiking — while discarding the biophysical complexity of Hodgkin-Huxley-style models.

The continuous-time dynamics are derived from an RC circuit analogy. The membrane has capacitance $C_m$ and leak conductance $g_L$. Applying Kirchhoff's current law and defining $\tau_{mem} = C_m / g_L$:

$$\tau_{mem} \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)$$

Setting $V_{rest} = V_{reset} = 0$ and absorbing $R_m$ into the input (so the input is already in voltage units):

$$\tau_{mem} \frac{dV_m}{dt} = -V_m + u(t)$$

This is the standard LIF ODE. The spike condition and reset rule complete the model: if $V_m(t) \geq V_{th}$, emit a spike $s(t) = 1$ and reset $V_m \leftarrow 0$ (reset-to-zero, the snntorch default).

### 2.3 Discrete-Time Dynamics

To simulate the LIF neuron on a computer, time is discretized into steps of size $dt$. The homogeneous solution of the ODE decays exponentially, so over one timestep the membrane potential decays by a factor:

$$\beta = \exp\left(-\frac{dt}{\tau_{mem}}\right)$$

This is the membrane decay constant. Adding the input term (treated as constant over the interval $[t, t+dt]$) and incorporating the reset:

$$U[t] = \beta \cdot U[t-1] \cdot (1 - S[t-1]) + I[t]$$
$$S[t] = \begin{cases} 1 & \text{if } U[t] \geq V_{th} \\ 0 & \text{otherwise} \end{cases}$$

where $U[t]$ is the membrane potential at timestep $t$, $S[t]$ is the spike output, $I[t] = W \mathbf{s}_{pre}[t]$ is the weighted sum of presynaptic spikes (or the encoded input current for the first layer), and the term $(1 - S[t-1])$ implements the reset-to-zero: if the neuron spiked at the previous step, its membrane is zeroed before the new input is added.

In this project, $\tau_{mem} = 20$ ms and $dt = 1$ ms, giving:

$$\beta = \exp\left(-\frac{1}{20}\right) \approx 0.9512$$

Over $T = 25$ timesteps, the decay factor compounds to $\beta^{25} \approx 0.28$, meaning that a membrane potential set at $t=0$ with no further input retains only about 28% of its value by $t=25$. This sets the effective temporal integration window: inputs arriving more than roughly 25 timesteps apart are nearly independent. The firing threshold is $V_{th} = 1.0$, the resting potential $V_{rest} = 0$, and the reset potential $V_{reset} = 0$.

### 2.4 The Non-Differentiability Problem and Surrogate Gradients

Training SNNs with backpropagation requires computing the gradient of the loss with respect to the network parameters. The spike function $S[t] = \Theta(U[t] - V_{th})$ is a Heaviside step function, which has zero derivative almost everywhere and is undefined at the threshold. This makes the standard backpropagation algorithm inapplicable: the gradient of the loss with respect to the pre-spike membrane potential is zero everywhere except at the threshold, where it is undefined.

The surrogate gradient method resolves this by replacing the true derivative of the spike function with a smooth approximation during the backward pass only. The forward pass uses the true Heaviside function (so spikes are genuinely binary), but the backward pass uses a surrogate that provides a useful gradient signal. The fast-sigmoid surrogate is the most common choice:

$$\frac{dS}{dU}\bigg|_{\text{surrogate}} = \frac{1}{(1 + k|U - V_{th}|)^2}$$

where $k$ controls the sharpness of the approximation. This is the surrogate used in snntorch. The gradient is largest near the threshold and decays away from it, which provides a sensible learning signal: neurons whose membrane potential is close to the threshold receive the strongest gradient, while neurons far from threshold receive little gradient.

The surrogate gradient approach is a pragmatic approximation, not a biologically principled one. It works well in practice and has enabled the training of deep SNNs on standard benchmarks, but it introduces a mismatch between the forward and backward passes that has no clean theoretical justification. For the purposes of this project, surrogate gradients are treated as a standard engineering tool.

### 2.5 Input Encoding and Output Decoding

MNIST images are static, not temporal, so they must be encoded into a spike train for the SNN to process. Two encoding schemes are common. **Rate coding** converts each pixel value to a firing probability: a pixel with value $p \in [0,1]$ generates a Bernoulli spike at each timestep with probability $p$. Over $T$ timesteps, the expected number of spikes is $pT$. **Direct current injection** (also called constant current encoding) treats the normalized pixel value as a constant input current $I[t] = p$ at every timestep, which the LIF neuron integrates over time. Direct encoding is simpler and avoids the stochasticity of rate coding; it is the approach used in this project.

For output decoding, the network runs for $T = 25$ timesteps and the output spikes from the final layer are summed over time. The class with the highest total spike count is the prediction:

$$\hat{y} = \arg\max_c \sum_{t=1}^{T} S_c[t]$$

Cross-entropy loss is applied to the summed spike counts (treated as logits), which provides a differentiable training signal through the surrogate gradient mechanism.


---

## 3. The Central Idea and Its Novelty

### 3.1 Spike Sparsity as a Mechanism

The term "sparsity" in this project refers to *activation sparsity* — the fraction of neurons that are active (i.e., that spike at least once) during a given forward pass — not weight sparsity (the fraction of zero-valued parameters). These are distinct concepts with different implications. Weight sparsity is a property of the parameter matrix; activation sparsity is a property of the network's behavior on a given input. An SNN with dense weights can produce sparse activations if the threshold is high enough, and a network with sparse weights can produce dense activations if the surviving weights are large.

The primary sparsity metric throughout this project is the active-neuron percentage: the fraction of hidden neurons that spike at least once over the $T = 25$ timestep simulation window, averaged over a batch of inputs. This is the quantity that is controlled (via threshold tuning, winner-take-all selection, or activity regularization) and the quantity that is used in all statistical models. The calibration target is a desired active-neuron percentage; the observed active-neuron percentage (which may differ from the target due to imperfect calibration) is what enters the analysis.

### 3.2 The Mediation Hypothesis

The central claim of this project is a mediation hypothesis, stated precisely as follows:

> Moderate spike sparsity reduces catastrophic forgetting in SNNs, and this reduction is at least partially mediated by reduced representational overlap between sequentially learned task representations.

The word "mediated" carries specific statistical meaning. It is not enough to show that sparsity correlates with lower forgetting, or even that sparsity correlates with lower overlap and lower overlap correlates with lower forgetting. Mediation requires demonstrating that the effect of sparsity on forgetting operates *through* overlap — that is, that the indirect path (sparsity $\to$ overlap $\to$ forgetting) accounts for a significant portion of the total effect of sparsity on forgetting, after conditioning on confounds.

The distinction between correlation and mediation matters enormously for the scientific contribution. If sparsity reduces forgetting but not through overlap, then the mechanism is something else — reduced plasticity (fewer weight updates), reduced capacity (less to learn and therefore less to forget), or some other pathway. In that case, the representational overlap story is wrong, and the contribution collapses to "sparsity helps forgetting," which is already established. The mediation claim is what makes this project novel.

### 3.3 Novelty Assessment: Occupied and Open Territory

The literature on SNN continual learning has grown substantially in recent years, and several components of this project's design have precedents. An honest novelty assessment requires distinguishing what is already occupied from what remains open.

**Occupied territory.** Shen, Ni, Xu, and Tang (2024, AAAI) used trace-based K-Winner-Take-All and variable thresholds to create sparse selective activation for continual learning in SNNs — this is the closest prior work and directly occupies the "sparse activation reduces forgetting in SNNs" space. Hammouamri, Masquelier, and Wilson (TMLR, OpenReview 15SoThZmtU) used firing-threshold modulation specifically to reduce forgetting in SNNs, directly occupying the threshold-control component. Meem, Nadid, and Mia (2026, arXiv:2602.12236) combined replay, learnable LIF neurons, and an adaptive spike scheduler for energy-aware continual learning in SNNs, occupying the spike-budget control space. On the ANN side, Ramasesh, Dyer, and Raghu (2020, ICLR) studied forgetting through hidden representations and task semantics, establishing the ANN-side link between representational overlap and forgetting.

**Open territory.** What none of these papers does is run a controlled study that (a) systematically varies sparsity across multiple distinct mechanisms, (b) measures representational overlap as a primary outcome, and (c) tests whether overlap *mediates* the sparsity-forgetting relationship using a formal mediation model with bootstrap confidence intervals. The full package — multiple controlled sparsity mechanisms in CL-SNNs, representational overlap as a mediator rather than a correlate, and the formal statistical test of mediation — is the contribution.

The novelty survives only if mediation stays central. If the mediation analysis fails (the indirect effect is not significantly nonzero), the contribution degrades to "sparsity helps forgetting in SNNs," which is already Shen et al. (2024). The project is designed to test this honestly, including the possibility of a null or negative result.

---

## 4. Research Questions and Hypotheses

### 4.1 Research Questions

**RQ1 (Primary):** Does increasing spike sparsity reduce catastrophic forgetting in spiking neural networks?

**RQ2:** How does spike sparsity affect retention of previously learned tasks across different continual learning protocols (task-incremental vs. class-incremental, naive vs. replay vs. regularization)?

**RQ3:** What sparsity level gives the best tradeoff among classification accuracy (target: at least 90% of baseline), forgetting rate (target: no more than 20% degradation from peak), and energy efficiency (estimated from spike count and synaptic operations)?

**RQ4:** Do sparse SNNs at their best sparsity level outperform equally parameterized ANN baselines on continual learning benchmarks?

**RQ5:** What mechanism links spike sparsity to reduced catastrophic forgetting?

### 4.2 Hypotheses

**H1: Moderate spike sparsity reduces forgetting.**

*Prediction:* SNNs with 20–40% active neurons will show at least a 30% reduction in mean forgetting compared with dense baselines (active-neuron percentage near 100%).

*Mechanism:* Sparse activity reduces overlap between task representations. When fewer neurons are active, fewer weight updates during Task B training interfere with the representations learned for Task A.

*Test:* Measure the forgetting score (best accuracy minus current accuracy) across sparsity levels from 1% to 95%. Compare the 20–40% range against the dense baseline using paired t-tests (or Wilcoxon signed-rank if normality is violated), with seeds as the unit of replication. Report Cohen's $d$ with a confidence interval alongside every pairwise comparison.

*Evidence required:* Forgetting score at 20–40% activity is less than 0.7 times the forgetting score at 100% activity, with $p < 0.05$ after Holm-Bonferroni correction. The reduction must persist after matching task-A peak accuracy across sparsity levels (ruling out the capacity confound) and after comparing against a count-matched dense-frozen control (ruling out the plasticity confound).

**H2: Extreme sparsity harms accuracy.**

*Prediction:* SNNs with less than 5% active neurons will show at least a 50% accuracy drop compared with dense baselines.

*Mechanism:* If too few neurons are active, the network lacks the representational capacity to encode task-specific features. The information bottleneck becomes too severe.

*Test:* Measure classification accuracy at 1%, 5%, and 10% activity. Compare against the dense baseline.

*Evidence required:* Accuracy at 1% and 5% activity is less than 0.5 times baseline accuracy, with $p < 0.05$ after Holm-Bonferroni correction.

**H3: The sparsity-performance relationship is non-linear.**

*Prediction:* Continual learning performance follows an inverted-U pattern as a function of active-neuron percentage, with the best performance near 20–40% activity.

*Test:* Fit a quadratic regression to accuracy (or forgetting) as a function of observed active-neuron percentage. Test for non-linearity with an F-test at $p < 0.05$ after correction.

*Interior-peak requirement:* A significant quadratic term is necessary but not sufficient. The fitted vertex (the peak of the inverted-U) must lie strictly inside the tested activity range — inside (5%, 95%), ideally within the predicted 20–40% band. A curve that merely flattens or bends at an extreme can also produce a significant quadratic coefficient and does not count as support for an inverted-U. The peak location is reported with a bootstrap confidence interval (resampled over seeds) that must exclude the range boundaries.

*Mechanism-separation requirement:* The inverted-U must be established per sparsity mechanism (threshold, winner-take-all, activity regularization) before any pooled curve is reported. Mechanistically different interventions are not pooled unless their individual curves agree in shape and direction.

**H4: Representational overlap mediates the sparsity-forgetting relationship.**

*Prediction:* The effect of spike sparsity on forgetting is at least partially mediated by reduced representational overlap between tasks. Specifically: increasing sparsity reduces overlap (path $a$), and lower overlap predicts lower forgetting conditional on sparsity (path $b$), such that the indirect effect $a \times b$ is significantly different from zero.

*Rationale:* This is the central novelty of the project. Prior SNN continual-learning work establishes that sparse activation, threshold modulation, and spike budgeting can reduce forgetting; what remains uncharacterized is *why*. Establishing overlap as a mediator — rather than merely a correlate — is what distinguishes this study from occupied territory.

*Test:* A formal mediation model estimating the indirect effect with a bootstrap confidence interval, conditional on confounds (see Section 6). A significant negative correlation between overlap and forgetting is a necessary precondition but is not, by itself, accepted as evidence of mediation.

*Decision rule:* H4 is supported only if the indirect effect $a \times b$ is significantly different from zero (bootstrap CI excludes zero) after conditioning on the covariates. A significant total effect with a non-significant indirect effect indicates sparsity acts through a non-overlap pathway (reduced plasticity or capacity), which would weaken the central novelty claim.


---

## 5. Mechanism Measurement and the Formal Mediation Model

### 5.1 Representational Overlap Metrics

Measuring representational overlap requires choosing a metric that captures the degree to which two tasks' hidden representations occupy the same subspace. Several metrics are available, and they are not interchangeable — they measure related but distinct properties.

**Cosine similarity** between task-mean representations is the simplest measure. Let $\bar{h}_A$ and $\bar{h}_B$ be the mean hidden-layer activations (spike counts over the simulation window, averaged over a held-out evaluation set) for tasks A and B respectively. The cosine similarity is:

$$\text{cos}(\bar{h}_A, \bar{h}_B) = \frac{\bar{h}_A \cdot \bar{h}_B}{\|\bar{h}_A\| \|\bar{h}_B\|}$$

This captures the alignment of the mean activation vectors but ignores the structure of the full activation distributions.

**PCA subspace overlap** projects the hidden representations of each task onto their principal components and measures the overlap between the resulting subspaces. Let $U_A \in \mathbb{R}^{d \times k}$ and $U_B \in \mathbb{R}^{d \times k}$ be the top-$k$ principal components of the task-A and task-B activation matrices. The subspace overlap is:

$$\text{SubspaceOverlap}(A, B) = \frac{1}{k} \|U_A^T U_B\|_F^2$$

This ranges from 0 (orthogonal subspaces, no overlap) to 1 (identical subspaces, complete overlap).

**Centered Kernel Alignment (CKA)** (Kornblith, Norouzi, Lee, and Hinton, 2019) is a more principled similarity measure that is invariant to orthogonal transformations and isotropic scaling. For two activation matrices $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$ (where $n$ is the number of examples), the linear CKA is:

$$\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \cdot \|Y^T Y\|_F}$$

CKA ranges from 0 (no similarity) to 1 (identical representations up to linear transformation). It is more robust than cosine similarity to differences in the scale and orientation of representations, and it has become the standard tool for comparing neural network representations across layers, architectures, and training conditions. In this project, CKA is used both as a cross-task overlap measure (comparing task-A and task-B representations at the same layer) and as a representation drift measure (comparing task-A representations before and after task-B training).

**Linear probe accuracy** measures whether task information remains linearly recoverable from the hidden representations. After training task A, the feature extractor is frozen and a linear classifier is trained on the hidden representations to predict task-A labels. After training task B, the same frozen feature extractor is re-probed with a new linear classifier. A drop in probe accuracy indicates that the representation itself degraded, independent of the output head.

### 5.2 Three Quantities That Must Not Be Conflated

A critical methodological point: three distinct quantities must be reported separately and never collapsed into a single "overlap" or "forgetting" measure.

**Cross-task representational overlap** is the similarity between the representations of *different* tasks (task A vs. task B), measured at a fixed layer. This is the proposed mediator. High cross-task overlap implies that the two tasks share coding subspace, so weight updates for task B are more likely to interfere with task A's representations.

**Representation drift** is the change in task A's *own* representation before versus after task-B training. This is measured by computing CKA between task-A hidden representations (on task-A inputs) before task-B training begins and after task-B training completes. High drift means task A's internal code was overwritten; low drift means it was preserved.

**Decodability** is the linear-probe accuracy on frozen features. This measures whether task information remains linearly recoverable regardless of drift. A representation can drift substantially yet remain decodable (if the drift is a linear transformation that preserves the task-relevant structure), or stay stable yet lose decodability (if the task-relevant dimensions are suppressed). These are not interchangeable, and conflating them leads to incorrect conclusions about the mechanism.

The mediation model uses cross-task overlap as the primary mediator. Representation drift and decodability are secondary outcomes that help interpret the mechanism.

### 5.3 The Formal Mediation Model

The formal mediation model is the core of the project's statistical contribution. It estimates the indirect effect of sparsity on forgetting through representational overlap, conditional on confounds.

**Variables:**
- $X$ = observed active-neuron percentage (the treatment; always observed, never the calibration target)
- $M$ = primary cross-task representational overlap between task A and task B, measured at the primary hidden layer *during task-B training* (the mediator; predefined before analysis)
- $Y$ = mean forgetting score $\mathcal{F}$ over prior tasks (the outcome)
- Covariates: sparsity mechanism (threshold / WTA / activity-reg), task setting (task-IL / class-IL), task-A mastery (matched peak accuracy, loss, and probe accuracy), with seed as a random effect

**Path a** (sparsity to overlap): Regress $M$ on $X$ plus covariates plus a random intercept for seed:

$$M = a_0 + a_1 X + a_2 \cdot \text{mechanism} + a_3 \cdot \text{setting} + a_4 \cdot \text{mastery} + u_{\text{seed}} + \epsilon_M$$

The coefficient $a_1$ is path $a$: the effect of sparsity on overlap, conditional on covariates.

**Path b and direct effect** (overlap and sparsity to forgetting): Regress $Y$ on $M$ plus $X$ plus covariates plus random seed effect:

$$Y = b_0 + b_1 M + c' X + b_2 \cdot \text{mechanism} + b_3 \cdot \text{setting} + b_4 \cdot \text{mastery} + u_{\text{seed}} + \epsilon_Y$$

The coefficient $b_1$ is path $b$: the effect of overlap on forgetting, conditional on sparsity and covariates. The coefficient $c'$ is the direct effect of sparsity on forgetting, conditional on overlap and covariates.

**Indirect effect:** The mediated effect is $a_1 \times b_1$, estimated with a bootstrap confidence interval resampled at the seed/run level. The proportion of the total effect that is mediated is $(a_1 \times b_1) / (a_1 \times b_1 + c')$.

**Reporting:** Path $a$, path $b$, direct effect $c'$, indirect effect $a \times b$ with bootstrap CI, and proportion mediated are all reported. H4 is supported only if the bootstrap CI for $a \times b$ excludes zero after conditioning on the covariates above.

**Decision logic:** A significant total effect ($a \times b + c' \neq 0$) with a non-significant indirect effect ($a \times b \approx 0$) indicates that sparsity acts through a non-overlap pathway — reduced plasticity (fewer weight updates) or reduced capacity (less to learn) — rather than through representational separation. This would weaken the central novelty claim substantially, because it would mean the mechanism story is wrong even if the performance effect is real.

**Mechanism separation:** The mediation model is estimated per sparsity mechanism first. Mechanisms are pooled only if their individual $a$, $b$, and indirect effects are consistent in direction and magnitude. This requirement prevents a pooled estimate from masking mechanism-specific heterogeneity.


---

## 6. Confound Controls and Rigor

### 6.1 The Confound Problem

A lower forgetting score at moderate sparsity could arise for reasons entirely unrelated to reduced representational interference. Two confounds are particularly serious and must be ruled out before any causal claim about the mechanism.

The **capacity confound** arises because a sparser network may simply learn *less* of task A, so there is less to forget. Lower forgetting would then reflect weaker initial learning, not better retention. This confound is especially insidious because it can produce a forgetting reduction that looks exactly like the hypothesized effect.

The **plasticity confound** arises because with fewer active neurons, fewer weights are updated during task B. Lower forgetting would then reflect *fewer updates* — a trivial plasticity effect — rather than sparse coding reducing representational interference. This confound is also plausible: if only 20% of neurons are active, then roughly 80% of the weights connected to those neurons receive no gradient from task B, which mechanically reduces the degree to which task B can overwrite task A.

The study controls for both confounds, and for several additional confounds, through a set of carefully designed control conditions.

### 6.2 Matched Task-A Mastery (Control 3.3.1)

Forgetting comparisons across sparsity levels are only valid conditional on equal task-A mastery. If a sparse network achieves 85% peak accuracy on task A while a dense network achieves 99%, comparing their forgetting scores is misleading: the sparse network has less to forget by construction.

The control requires reporting task-A peak accuracy (immediately after training on task A, before any task-B training) at every sparsity level. Where peak accuracies differ, forgetting is reported as a function of task-A mastery — by stratifying or covarying on peak accuracy — rather than comparing raw forgetting scores at unequal starting points. The matching is done on peak accuracy, loss, and linear-probe accuracy to ensure that the comparison is valid at multiple levels of the representation.

If sparsity reduces interference, lower forgetting should persist at moderate sparsity even after task-A mastery is matched. If the forgetting reduction disappears after matching, it was driven by the capacity confound.

### 6.3 Forgetting in Representation Space (Control 3.3.2)

Output-level accuracy conflates two distinct effects: degradation of the internal representation and reshuffling of the readout. A network can forget at the output level because its hidden representations degraded (genuine representational forgetting) or because the output head was retrained in a way that no longer correctly reads out the preserved representations (readout reshuffling). These have different implications for the mechanism.

The linear probe protocol separates them. After training task A, the feature extractor is frozen and a linear probe is trained on each task. After training task B, the same frozen-then-probed protocol is applied. A drop in probe accuracy indicates the representation itself degraded, independent of the output head. CKA between task-A hidden representations before and after task-B training provides a complementary measure: high CKA with low output accuracy indicates "representation preserved, readout reshuffled"; low CKA indicates genuine representational forgetting.

If sparsity reduces representational interference, moderate sparsity should show higher pre/post-B CKA and smaller probe-accuracy drops than dense baselines.

### 6.4 Count-Matched Dense-Frozen Baseline (Control 3.3.3)

This control isolates "fewer weights updated" (a plasticity effect) from "sparse distributed coding reduces overlap" (the proposed mechanism). A dense network is constructed in which a random subset of weights is frozen during task B, matched in count to the number of weights the sparse network leaves unupdated at each sparsity level.

If the count-matched dense-frozen network forgets as little as the sparse network, the effect is attributable to fewer updates, not to sparse coding. If the sparse network forgets less than the count-matched dense-frozen control, this supports a genuine sparse-coding mechanism.

The limitation of this control is important to state: it matches the *quantity* of unupdated weights but not the *geometry* of which weights are updated or how large those updates are. The following controls address geometry and update magnitude.

### 6.5 Update-Norm-Matched Control (Control 3.3.4)

Sparse activation reduces not only how many parameters are updated but also the *magnitude* of gradients flowing through inactive units. Lower forgetting could therefore reflect smaller effective weight updates rather than sparse coding reducing overlap.

This control matches the total per-task update norm (sum of squared weight deltas, or per-layer update norm) between the sparse network and a dense comparison network, by scaling the dense network's learning rate or gradient norm so that its cumulative update magnitude equals the sparse network's at each sparsity level. Per-task and per-layer gradient norms, update norms, and cross-task gradient cosine similarity are logged for every run.

If forgetting differences vanish once update norm is matched, the mechanism is update-magnitude driven, not overlap driven.

### 6.6 Activation-Dropout Control (Control 3.3.5)

Sparsity mechanisms deactivate units. A dense network with random unit dropout can reproduce the same fraction of active units without any learned, input-dependent sparse code. This control isolates "fewer units active" (a stochastic capacity effect) from "structured, input-dependent sparse coding."

Random per-unit dropout is applied to a dense network, calibrated to match the *observed* active-neuron percentage of the sparse network at each level. If activation-dropout at matched active% reproduces the forgetting reduction, the effect does not require a learned sparse code. If the sparse network still forgets less, this supports structured sparse coding.

### 6.7 Structured Neuron-Block Freezing Control (Control 3.3.6)

The count-matched control (3.3.3) freezes a *random* subset of weights. Sparse coding may instead protect *contiguous, neuron-aligned* subnetworks (capacity partitioning), which random freezing does not emulate. This control freezes structured neuron-aligned blocks (whole units or channels) during task B, matched to the effective unupdated-unit count of the sparse network.

Comparing random-weight freezing (3.3.3), structured-neuron freezing (3.3.6), and true sparsity separates "which units are protected" from "how the code is distributed." If structured-neuron freezing matches the sparse network's forgetting reduction, the mechanism is capacity partitioning (protecting whole neurons) rather than distributed sparse coding (reducing overlap across shared neurons).

### 6.8 Ablation Logic: During-Task-B Manipulation

Interference is created *while* task B is being learned, not after. The ablation therefore manipulates sparsity during task-B training rather than masking the network after task A. This is a critical design choice.

The procedure: train task A to a fixed mastery level, then increase or decrease the active-unit sparsity *during* task-B training, which is when weight updates from task B can overwrite task-A structure. The task-A evaluation mask is kept fixed across all conditions, so that the measurement probe for task-A accuracy does not change when the training-time sparsity changes.

Post-hoc masking — masking the network after task A and then evaluating — changes which units the readout reads from (a readout effect) without changing the interference that occurred during task-B learning. It is therefore not a valid test of the interference mechanism. The during-task-B manipulation with a fixed evaluation mask isolates the interference mechanism the hypothesis is about.

Expected pattern: if sparsity has a causal role in reducing interference, increasing active-unit overlap during task-B training should increase task-A forgetting, and increasing sparsity during task-B training should reduce it, with the task-A evaluation mask held constant.

### 6.9 Learning-Rate Control

The main learning rate is fixed at 0.001 across all sparsity levels. A sensitivity analysis repeats key experiments with learning rates of 0.001, 0.0001, and 0.00001 to verify that the sparsity effect is not an artifact of the learning rate interacting with the effective gradient magnitude.

### 6.10 Mechanism Non-Equivalence

A critical point that is easy to overlook: the three sparsity mechanisms are distinct interventions, not interchangeable ways of reaching "the same sparsity." Matching them on a single scalar (percentage of active neurons) does not make them equivalent, because they differ in *which* neurons are silenced and *how* the surviving code is distributed.

**Spike threshold** raises the firing bar uniformly. Whichever neurons exceed the threshold fire, so the active set is input-driven and can be highly overlapping across inputs. Different inputs may activate largely the same neurons if they share common features.

**Winner-take-all (top-k)** enforces a hard cardinality per step: exactly $k$ neurons fire regardless of input magnitude. This produces competition and often more decorrelated codes, because neurons must compete to be among the top-$k$ rather than simply exceeding a fixed bar.

**Activity regularization** applies a soft penalty during training. Sparsity emerges as a learned equilibrium and may concentrate in particular units (some neurons become permanently inactive, others permanently active), producing a different kind of sparse code than either threshold or WTA.

Because these produce different *kinds* of sparsity at the same active-neuron percentage, results are analyzed per mechanism first. No pooled sparsity-forgetting or sparsity-overlap curve is reported unless the individual per-mechanism curves agree in shape and direction.

Per-mechanism reporting metrics for every (mechanism, activity level, seed) condition:
- Active-neuron percentage (observed, not target)
- Total spike count and mean spike rate
- Firing-rate entropy across neurons (how evenly activity is distributed vs. concentrated in a few units)
- Lifetime sparsity (per-neuron fraction of inputs on which the neuron is active)
- Per-neuron participation and dead-neuron fraction (how many units are effectively unused)


---

## 7. Methodology

### 7.1 Models

**ANN baseline: Multi-layer perceptron.** The ANN baseline is a fully connected MLP with architecture 784 $\to$ 256 $\to$ 256 $\to$ output, ReLU activations, and approximately 260,000 parameters. The input is a flattened 28$\times$28 MNIST image (784 dimensions). The two hidden layers each have 256 units. The output layer has 2 units per task in the task-incremental setting (per-task binary heads) or 10 units in the class-incremental setting (single growing head). The MLP is chosen over a CNN because it makes it easier to isolate sparsity effects without convolutional feature sharing; a simplified ConvNet can be used as a secondary validation model.

The MLP is trained with Adam ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$), batch size 128, and cross-entropy loss. For Split-MNIST, 10 epochs are trained per task.

**SNN baseline: Leaky integrate-and-fire network.** The SNN has the same architecture as the MLP (784 $\to$ 256 $\to$ 256 $\to$ output) but with LIF neurons in the hidden layers, implemented using snntorch 0.9.4. Parameters: $\tau_{mem} = 20$ ms, $dt = 1$ ms, $\beta = \exp(-1/20) \approx 0.9512$, $V_{th} = 1.0$, $V_{rest} = 0$, $V_{reset} = 0$, $T = 25$ timesteps per forward pass. The output layer uses summed spike counts as logits for cross-entropy loss. The surrogate gradient is the fast-sigmoid approximation.

The SNN is trained with the same Adam optimizer, learning rate, batch size, and epoch count as the MLP, to ensure that differences in forgetting are attributable to the spiking dynamics rather than to training differences.

### 7.2 Sparsity Manipulation

Three sparsity mechanisms are implemented, each calibrated to the same observed active-neuron percentage metric.

| Mechanism | How sparsity is controlled | Target activity levels |
|---|---|---|
| Spike threshold | Higher $V_{th}$ produces fewer spikes | 1%, 5%, 10%, 20%, 30%, 40%, 60%, 80%, 95% |
| Winner-take-all | Only the top-$k$ neurons fire per step | Matched to the same activity levels |
| Activity regularization | A penalty discourages high activity | Regularization strength tuned to the same levels |

The calibration procedure for threshold-based sparsity: train or warm up on the first task for a short calibration pass, sweep threshold values, select the threshold that produces the closest observed activity to the target level, keep that threshold fixed during the continual learning run, and record the observed activity level. The analysis always uses observed activity, never the calibration target.

The activity grid includes a 30% point (1%, 5%, 10%, 20%, 30%, 40%, 60%, 80%, 95%) to give the inverted-U fit adequate resolution near the hypothesized 20–40% optimum.

### 7.3 Continual Learning Benchmarks

**Split-MNIST** divides the 10-digit MNIST dataset into five binary classification tasks: digits 0 vs. 1 (Task 1), 2 vs. 3 (Task 2), 4 vs. 5 (Task 3), 6 vs. 7 (Task 4), and 8 vs. 9 (Task 5). Each task uses only the subset of training and test images belonging to its two classes, with labels remapped to $\{0, 1\}$ within each task for the task-incremental setting. The task order is fixed for reproducibility; a seed-level order sensitivity check is included in the statistical analysis.

**Permuted-MNIST** applies a different fixed random pixel permutation to all images for each of 10 tasks. The label set remains digits 0–9 across all tasks, making this a domain-incremental benchmark. It is included as an expansion-ladder benchmark after the Split-MNIST pilot produces interpretable results.

**CIFAR-10/100** (optional extension) is included only after MNIST-scale findings are stable, to test whether the effect survives harder visual inputs.

### 7.4 Continual Learning Methods

| Method | Type | Reason for inclusion |
|---|---|---|
| Naive sequential learning | Baseline | Reference condition for maximum forgetting |
| Replay buffer (size = 200) | Rehearsal | Strong practical baseline |
| Elastic Weight Consolidation | Regularization | Fisher-based parameter importance |
| Synaptic Intelligence | Regularization | Online parameter-importance method |
| Learning without Forgetting | Distillation | Standard distillation-based baseline |
| PackNet | Parameter isolation | Optional architecture-based comparison |

Naive sequential learning is the primary baseline for the pilot and the confirmatory analysis. The other methods are added in the full study to test whether sparsity still matters when explicit forgetting-prevention mechanisms are active.

### 7.5 Training Details

- **Optimizer:** Adam with $\eta = 0.001$
- **Batch size:** 128 for MNIST experiments, 64 for CIFAR experiments
- **Epochs:** 10 per task for MNIST, 20 per task for CIFAR
- **Seeds:** 8–10 for confirmatory conditions, at least 5 for secondary and exploratory conditions, 3 for the pilot (feasibility only — pilot p-values are never reported as confirmatory evidence)
- **Seed variation:** Seeds vary network initialization, data shuffling, and task order (for the order-sensitivity check)
- **Reporting:** All reported numbers are mean $\pm$ standard deviation across seeds; single-run point estimates are not reported

---

## 8. Evaluation Metrics

### 8.1 Forgetting Metrics

**Backward transfer (BWT)** measures the average effect of learning later tasks on earlier task performance:

$$\text{BWT} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left[ A_{T,t} - A_{t,t} \right]$$

where $A_{i,j}$ is the accuracy on task $j$ after training on task $i$. Negative BWT indicates forgetting; positive BWT (rare) indicates that later tasks improved earlier task performance.

**Forgetting score** $F_t$ for task $t$ is the drop from the best accuracy ever achieved on task $t$ to the final accuracy:

$$F_t = \max_{t' \leq t} A_{t',t} - A_{T,t}$$

The mean forgetting $\bar{F} = \frac{1}{T-1} \sum_{t=1}^{T-1} F_t$ is the primary outcome variable.

**Forward transfer (FWT)** measures the effect of previously learned tasks on the speed or quality of learning a new task:

$$\text{FWT} = \frac{1}{T-1} \sum_{t=2}^{T} \left[ A_{t-1,t} - b_t \right]$$

where $b_t$ is the accuracy on task $t$ when trained from scratch (without any prior task training).

### 8.2 Accuracy Metrics

**Average task accuracy** is the mean accuracy across all tasks after sequential training:

$$\bar{A} = \frac{1}{T} \sum_{t=1}^{T} A_{T,t}$$

**Final average accuracy** is the same quantity, emphasizing that it is measured at the end of all training.

### 8.3 Sparsity and Efficiency Metrics

**Spike rate** is the average number of spikes per neuron per timestep, averaged over a batch of inputs.

**Sparsity index** is the percentage of inactive neurons (neurons that do not spike at all over the $T$ timestep window).

**Energy proxy** is estimated as spike count $\times$ synaptic operations, following common neuromorphic efficiency estimates. This is a computational estimate, not a hardware energy measurement. It should be described as such and never used to claim hardware energy efficiency without neuromorphic hardware validation. The energy proxy is useful for comparing conditions within this study but cannot be extrapolated to real hardware without accounting for memory access patterns, data movement, and chip-specific characteristics (see Yan, Bai, and Wong, 2024, for a careful analysis of these issues).

### 8.4 Mechanistic Metrics

**Cosine similarity** between task-specific hidden representations (cross-task overlap measure).

**PCA subspace overlap** between task representation subspaces.

**Synaptic overlap** measured as correlation in weight updates between tasks (cross-task gradient cosine similarity, logged per run).

**CKA** between task-A representations before and after task-B training (representation drift measure), and between task-A and task-B representations at the same layer (cross-task overlap measure).

**Linear-probe accuracy** on frozen features per task, measured before and after task-B training, to quantify representation-space forgetting independent of the output head.


---

## 9. Statistical Plan

### 9.1 Confirmatory Test Hierarchy

To avoid an uncontrolled multiple-comparison surface, tests are partitioned into three predefined families, declared before analysis begins.

**Primary confirmatory family** covers the core hypotheses (H1, H2, H3, and the H4 mediation claim) evaluated on ONE predefined primary configuration: Split-MNIST, task-incremental setting, naive sequential training, spike-threshold sparsity mechanism. This is the family that decides whether the central claims are supported. Corrected with Holm-Bonferroni within-family.

**Secondary mechanism family** covers the same hypotheses re-tested across the other sparsity mechanisms (winner-take-all, activity regularization) to test mechanism generality. Corrected with Holm-Bonferroni within-family.

**Exploratory family** covers everything else: additional datasets (Permuted-MNIST, CIFAR), additional CL methods (replay, EWC, SI, LwF, PackNet), the ANN comparison (RQ4), and alternative overlap metrics. Corrected with Benjamini-Hochberg FDR and reported as exploratory, not confirmatory.

### 9.2 Seed Guidance

The pilot uses 3 seeds for feasibility only. Pilot p-values are never reported as confirmatory evidence and exist only to guide design decisions. Confirmatory conditions in the full study use 8–10 seeds; secondary and exploratory conditions use at least 5. Seeds vary initialization, data shuffling, and task order. All reported numbers are mean $\pm$ std with no single-run estimates.

### 9.3 Effect Sizes and Confidence Intervals

Cohen's $d$ is reported alongside every pairwise comparison, in every family. Forgetting-reduction claims (for example "30% reduction") are accompanied by both a p-value and an effect size with a confidence interval. The magnitude of an effect is always reported, not only its significance.

### 9.4 Non-Linearity Test (H3)

Quadratic regression is fit to accuracy (or forgetting) as a function of observed active-neuron percentage, per mechanism. The F-test for the quadratic term is the primary test of non-linearity. The interior-peak requirement is enforced: the fitted vertex must lie strictly inside (5%, 95%), ideally within 20–40%, with a bootstrap CI on the peak location that excludes the range boundaries. Per-mechanism fits are reported before any pooled fit.

### 9.5 Mediation Analysis (H4)

The formal mediation model (Section 5.3) is estimated per sparsity mechanism first. The indirect effect $a \times b$ is reported with a bootstrap CI resampled at the seed/run level. Mechanisms are pooled only if per-mechanism estimates agree in direction and magnitude. The proportion mediated is reported alongside the indirect effect.

### 9.6 Observed vs. Target Activity

All statistical models and plots use the OBSERVED active-neuron percentage per condition, never the calibration target. This is non-negotiable: the calibration target is a design parameter, not a measurement.

---

## 10. Two-Stage Plan and the Pilot

### 10.1 The Pilot's Role

The pilot is a feasibility screen, not a confirmatory study. Its purpose is to determine whether the correlation precondition for mediation holds at small scale: does moderate sparsity associate with lower overlap and lower forgetting? If the precondition fails at pilot scale, the mediation hypothesis is not worth the full study.

The pilot cannot support a mediation claim. It uses 3 seeds (too few for confirmatory inference), a single sparsity mechanism (threshold only), and a single dataset and setting (Split-MNIST, task-incremental). Its p-values are never reported as confirmatory evidence.

### 10.2 Pilot Scope

| Component | Pilot choice | Reason |
|---|---|---|
| Dataset | Split-MNIST | Simple, standard CL benchmark |
| Setting | Task-incremental | Clear first setting with task labels available |
| Model | LIF-SNN only | Tests the core SNN hypothesis first |
| CL method | Naive sequential | Exposes forgetting without mitigation |
| Sparsity mechanism | Spike-threshold only | Simplest direct control |
| Sparsity levels | 1%, 10%, 20%, 40%, 80% | Covers extreme sparse, moderate, and dense-ish |
| Seeds | 3 | Minimum to avoid seed-specific conclusions |

### 10.3 Pilot Measurements

After each task, the model is evaluated on all tasks learned so far. The accuracy matrix $A \in \mathbb{R}^{5 \times 5}$ is stored, where $A_{i,j}$ is the accuracy on task $j$ after training task $i$. Spike counts and active-neuron percentages are logged per task and per sparsity level. Hidden representations (spike counts over the simulation window, averaged over a held-out evaluation set) are saved for overlap analysis.

### 10.4 The Five Pilot Plots

The pilot produces five plots that together screen the correlation precondition:

1. **Accuracy matrix per sparsity level:** A heatmap of $A_{i,j}$ for each sparsity condition, showing how accuracy on each task evolves as later tasks are trained. This reveals the forgetting pattern and whether it differs across sparsity levels.

2. **Observed activity vs. final average accuracy:** A scatter plot with one point per (sparsity level, seed) combination, showing whether moderate sparsity achieves competitive accuracy. This screens H2 (extreme sparsity hurts accuracy) and the accuracy component of H3.

3. **Observed activity vs. mean forgetting:** A scatter plot showing whether moderate sparsity reduces forgetting. This screens H1 and the forgetting component of H3.

4. **Observed activity vs. representational overlap:** A scatter plot showing whether higher sparsity reduces cross-task overlap. This screens path $a$ of the mediation model.

5. **Representational overlap vs. mean forgetting:** A scatter plot showing whether lower overlap predicts lower forgetting. This screens path $b$ of the mediation model.

### 10.5 Pilot Decision Criteria

**Continue to full experiments if** most of the following are true across at least 3 seeds:
1. Forgetting is lower at moderate sparsity than at dense or near-dense activity.
2. Extreme sparsity hurts accuracy.
3. Representational overlap decreases as sparsity increases.
4. Representational overlap correlates with forgetting.
5. The pattern is visible across at least 3 seeds.

**Revise the hypothesis if** any of the following happen:
1. Sparsity changes spike rate but not forgetting.
2. Sparsity changes forgetting but not representational overlap.
3. The best result occurs only at extreme sparsity.
4. Results differ strongly across seeds.
5. The model fails to learn Split-MNIST reliably.

**Stop or reframe if** the pilot shows no reliable relationship among sparsity, overlap, and forgetting. In that case, the paper may still become a negative result or a study of why spike sparsity alone is insufficient for continual learning.

### 10.6 Full Study Expansion

The full study adds, in order: (1) ANN MLP baseline with matched parameter count; (2) winner-take-all and activity-regularization sparsity mechanisms; (3) replay, EWC, SI, LwF, and optionally PackNet; (4) Permuted-MNIST; (5) class-incremental setting; (6) optionally CIFAR. Each expansion is conditional on the previous stage producing interpretable results.

The full study runs the formal mediation model (Section 5.3) across mechanisms with 8–10 confirmatory seeds. The final mediation claim rests on the full study, not on the pilot.


---

## 11. Pilot Findings

### 11.1 Overview and Scope

The pilot experiment was a minimal screening run whose sole purpose was to test the correlation precondition for the mediation hypothesis: does representational overlap between task representations track catastrophic forgetting as spike activity varies? It was not designed to run the formal mediation model, to confirm any hypothesis, or to characterize the full sparsity-forgetting relationship. The design was Split-MNIST, task-incremental setting, LIF-SNN, naive sequential training, threshold-based sparsity control, and 3 random seeds — the minimum configuration described in Section 10.2.

The two-stage logic is worth restating precisely. The mediation hypothesis (H4) requires, as a necessary precondition, that representational overlap and forgetting co-vary with spike activity in the expected direction. If that correlation is absent at pilot scale, the full mediation model is not worth running. The pilot screens this precondition; the full study will run the formal mediation model with the complete seed count, multiple sparsity mechanisms, and the statistical machinery described in Sections 5 and 9. Nothing in this section constitutes confirmatory evidence.

### 11.2 A Methodological Finding: Calibration Drift and the Corrected Design

The original calibration procedure, described in Section 7.2, trained the network on a short warmup pass over Task 0, swept threshold values to find the one producing the target active-neuron percentage, and then froze that threshold for the remainder of the continual learning run. This procedure failed in a systematic and instructive way.

The problem is that a threshold calibrated on an underfit network is not meaningful once the network is trained. During the warmup pass, weights are near their random initialization and the network has not yet learned to represent the input distribution. Membrane potentials are correspondingly modest, and a relatively low threshold is needed to produce any given activity level. As training proceeds and weights grow, more neurons cross the same frozen threshold on each forward pass. The result is substantial activity drift: a threshold calibrated to produce approximately 7% active neurons during the warmup pass produced approximately 42–64% active neurons during the actual continual learning run, once the network had trained for several epochs. The calibration target and the observed activity diverged by a factor of six to nine.

This is not a numerical accident but a structural consequence of the calibration design. With a frozen global threshold, spike activity is not a directly controllable variable. It is an emergent property of the trained weights interacting with a fixed threshold. Calibrating on an underfit network and then freezing the threshold produces a threshold that is systematically too low for the trained network.

The corrected design abandons the calibration-target framing entirely. Rather than asking "what threshold produces X% activity?" and then freezing that threshold, the corrected design treats the threshold as the intervention and activity as the measured outcome. The experiment sets the LIF threshold to a fixed value $\theta$ drawn from a predetermined sweep, trains the network, and records the activity level that emerges. This is more honest about what is actually being controlled: the intervention is "we set the threshold to $\theta$"; the response is "the network exhibited $X$% activity during continual learning." The analysis then uses the observed activity, as Section 9.6 requires, rather than the nominal target.

This reframing has a practical consequence for the statistical plan. The sparsity axis in the corrected design is defined by the threshold sweep, not by a target activity grid. The observed activity values at each threshold are the independent variable in all downstream analyses.

That said, the fixed-threshold sweep itself turned out not to be the final design. As documented in §11.8, a subsequent 27-condition sweep across nine threshold values revealed that a single frozen global threshold cannot control spike activity in a trained network: observed activity clustered in a bimodal band regardless of the threshold value, with no stable moderate-activity regime. The corrected design described here was a necessary intermediate step — it removed the calibration-drift problem and made the intervention well-defined — but it exposed a deeper limitation of threshold-based sparsity control. The sparsity mechanism was subsequently replaced by k-winner-take-all (k-WTA), which directly sets the over-window active fraction as a dial rather than relying on a threshold to produce a target activity level.

### 11.3 A Model-Property Finding: The ~38% Activity Ceiling

Under the specific configuration used in this project — LIF neurons with reset-to-zero, $\tau_{mem} = 20$ ms ($\beta \approx 0.951$), $T = 25$ timesteps, and direct current-injection encoding that repeats the normalized pixel value as a constant input at every timestep — no threshold, however low, drives more than approximately 37–38% of hidden neurons to fire over the simulation window. The reachable sparsity axis is therefore roughly 1–38% active neurons, not the 1–80% range originally planned.

The empirically measured threshold-to-activity map on a trained network is as follows:

| Threshold $\theta$ | Observed active-neuron % |
|---|---|
| 1.5 | ~35% |
| 3.0 | ~30% |
| 5.0 | ~23% |
| 8.0 | ~17% |
| 16 | ~14% |
| 24 | ~11% |
| 32 | ~8% |
| 48 | ~3.4% |
| 64 | ~0.9% (near-dead) |

The ceiling near 38% is a property of the encoding and neuron configuration, not a failure of the threshold sweep. With direct encoding, each pixel value is injected as a constant current for all 25 timesteps. A neuron that receives sufficient current will fire on the first timestep and then reset; whether it fires again depends on whether the residual membrane potential, after the reset, accumulates enough from the continued input to cross the threshold again within the remaining timesteps. For most neurons receiving moderate input, the answer is no: the reset-to-zero rule clears the membrane, and the remaining timesteps are insufficient to recharge past the threshold. The result is that the maximum achievable activity is bounded well below 100%, and that bound sits near 38% for this particular combination of $\beta$, $T$, and encoding scheme.

This ceiling should be revisited if the full study requires a denser activity regime. Increasing the number of timesteps, switching to rate coding, or using a different reset rule (reset-by-subtraction rather than reset-to-zero) would each raise the ceiling. For the present study, the reachable axis of approximately 1–38% is the operative range, and the activity grid and hypotheses should be interpreted accordingly.

### 11.4 The Dead-Network Cliff

At the highest tested threshold ($\theta = 64$), the network fired essentially zero spikes across all three seeds. Training loss remained frozen at $\ln(2) \approx 0.693$ — the value expected when the network outputs uniform logits and learns nothing — and final accuracy sat at approximately 51.6%, consistent with chance performance on binary classification tasks. The network was, in effect, dead: no gradient flowed through the spike function because no spikes were produced, and the surrogate gradient was zero everywhere.

This is a degenerate boundary condition, not a graded decline. The transition from near-zero activity to zero activity is a cliff rather than a slope: at $\theta = 48$ the network still fires at approximately 3.4% and learns, while at $\theta = 64$ it fires at approximately 0.9% and fails to learn at all. The cliff is consistent with the spirit of H2 — extreme sparsity harms accuracy — but it does not characterize the *shape* of the accuracy decline in the extreme-sparse regime, because the network collapses to chance rather than degrading gracefully. Conditions meeting the dead-network criterion (training loss $\geq 0.68$ after the first task, or final accuracy $\leq 55\%$ on binary tasks) are now automatically flagged and excluded from mechanism analysis. The shape of the accuracy-versus-sparsity relationship in the extreme-sparse regime below approximately 3% activity therefore remains uncharacterized and is deferred to future work.

### 11.5 The Mechanism Signal: Representational Overlap Tracks Forgetting

The central positive result of the pilot is that linear CKA between task representations tracked catastrophic forgetting monotonically and reproducibly across all three seeds. Conditions with the lowest CKA values (approximately 0.006–0.007) had the lowest forgetting, and conditions with the highest CKA values (approximately 0.013–0.015) had the worst forgetting, with mean forgetting ranging from approximately 0.03 at the sparse end to approximately 0.44 at the dense end of the reachable activity axis. This monotonic relationship held across seeds without exception.

This is the correlation precondition the mediation hypothesis requires. The pilot was designed to test whether this precondition holds, and it does. The result survived even on a compressed and imperfect activity axis — the calibration drift described in Section 11.2 meant that the first screening runs covered a narrower observed-activity range (approximately 0.38–0.56) than intended, and the relationship was still visible.

Two clarifications are essential. First, this is a correlation result at pilot scale, not a mediation claim. Demonstrating that CKA and forgetting co-vary does not establish that overlap is the pathway through which sparsity affects forgetting. That requires the formal mediation model, with the indirect effect estimated conditional on confounds, which is the work of the full study. Second, the pilot used 3 seeds and a single sparsity mechanism; the result is reproducible within this narrow scope but cannot be generalized beyond it.

A secondary observation concerns the contrast between cosine similarity and CKA as overlap metrics. Cosine similarity between raw task-mean representations remained high throughout — approximately 0.97–0.99 — while centered CKA was near zero. This is expected and not a contradiction. Cosine similarity on raw mean vectors measures the alignment of the average activation patterns, which tends to be high whenever both tasks activate a broadly similar set of neurons (as is the case for related digit-classification tasks sharing the same input space). CKA on centered activation matrices measures the similarity of the *relational structure* of the representations — how examples within each task relate to one another — which is a more sensitive and appropriate measure of representational overlap for the mediation hypothesis. CKA is the correct mediator metric, and the near-zero CKA values at low activity indicate that the task representations are structurally distinct even when their mean vectors point in similar directions.

### 11.6 What the Pilot Did Not Establish

Several questions the full study must answer remain open after the pilot.

The inverted-U relationship predicted by H3 could not be characterized from the initial screening runs. The observed-activity axis in those runs was compressed (approximately 0.38–0.56) due to calibration drift, and the relationship between the nominal threshold target and the observed activity was non-monotonic. The corrected fixed-threshold sweep was intended to produce a clean, monotonic activity axis, but it did not: as documented in §11.8, the sweep revealed that a frozen global threshold cannot control activity in a trained network, and no stable moderate-activity regime emerged. The sparsity mechanism was therefore replaced by k-WTA, which did produce a controlled, monotonic activity axis spanning approximately 1–33%. On that controlled axis, the inverted-U predicted by H3 was not observed: accuracy increased roughly monotonically with activity and forgetting decreased monotonically, with no interior optimum. The H3 characterization is thus complete for this configuration, and the result is a negative one — the inverted-U did not materialize on Split-MNIST with this LIF setup.

The shape of the accuracy decline in the extreme-sparse regime is uncharacterized. The dead-network cliff at $\theta = 64$ establishes that the network collapses to chance at near-zero activity, but it does not describe how accuracy degrades between approximately 3% and 0% activity. Whether the decline is gradual or abrupt, and whether it is consistent across seeds, requires additional threshold values in that range.

None of the pilot results are confirmatory. Three seeds are insufficient for confirmatory inference, and the pilot was explicitly designed as a screening exercise. The p-values from pilot-scale analyses are not reported here, consistent with the policy stated in Section 9.2.

### 11.7 Net Verdict: Proceed to the Full Study

The pilot produced three findings that together justify proceeding to the full study. The calibration machinery has been simplified: activity is now a measured outcome of a fixed-threshold sweep rather than a calibration target, which removes the drift problem and makes the intervention well-defined. The reachable activity axis has been characterized: approximately 1–38% under the current configuration, with a dead-network cliff below approximately 1%. And the mechanism signal — the thing the novelty of this project rests on — is real and reproducible across all three seeds: representational overlap measured by linear CKA tracks catastrophic forgetting monotonically in the expected direction.

The remaining work is to establish a controlled activity axis, characterize the sparsity-forgetting and sparsity-overlap relationships with adequate resolution, and then run the formal mediation model with the full seed count. §11.8 documents the two subsequent steps that completed the activity-axis problem: the fixed-threshold sweep that exposed the limits of threshold-based control, and the k-WTA run that finally produced the controlled axis the downstream analysis requires.

### 11.8 Fixed-Threshold Failure and the k-WTA Result

#### 11.8.1 The Fixed-Threshold Sweep: A Negative Methodological Result

The corrected design described in §11.2 — treating the threshold as the intervention and activity as the measured outcome — was implemented as a 27-condition sweep: 3 seeds crossed with 9 fixed thresholds ($\theta \in \{1.5, 3, 5, 8, 16, 24, 32, 48, 64\}$), 10 epochs per task. The sweep was designed to produce a clean, monotonic activity axis spanning the full reachable range from near-zero to approximately 38%.

It did not. Observed over-window active fraction clustered bimodally around 0.33–0.58 across the middle of the threshold range and was non-monotonic in $\theta$: $\theta = 1.5$ produced approximately 35% activity, $\theta = 16$ produced approximately 57%, and $\theta = 48$ produced approximately 38%. No threshold value yielded a stable moderate-activity regime in the 5–30% range. At $\theta = 64$, all three seeds collapsed to a dead network: essentially zero spikes, training loss frozen at $\ln 2 \approx 0.693$, and final accuracy at approximately 51.6% — chance performance on binary classification tasks, consistent with the dead-network cliff described in §11.4.

This is not a tuning artifact. Because calibration had already been removed, the behavior reflects a fundamental property of the trained network: as weights grow during training, activity re-saturates into the 33–58% band regardless of the frozen threshold. The threshold controls the dead/alive boundary — whether the network fires at all — but not the activity level within the alive regime. A single frozen global threshold cannot serve as a sparsity dial for a trained SNN.

The old threshold mechanism is retained in the codebase as a config-selectable fallback, so the negative result remains reproducible.

#### 11.8.2 The k-WTA Switch

Sparsity control was moved to a k-winner-take-all mechanism. For each input sample and each hidden layer, a fixed winner set of $k$ neurons is selected once per forward pass, scored by summed membrane potential over the full $T = 25$ timestep window, and held for the entire window: non-winners have both their spikes and membrane potential multiplied by a detached 0/1 mask at every timestep. This is a per-sample, per-layer, whole-window winner set.

The distinction from per-timestep top-$k$ selection is important. Per-timestep selection would allow different neurons to win on different timesteps; the union of winners over the window would cover most of the layer, defeating the over-window activity metric. The whole-window winner set upper-bounds the over-window active fraction per layer at $k / \text{width}$, making activity a directly-set dial. Gradients still flow through kept neurons via the surrogate gradient mechanism; the mask is detached, so it does not block the backward pass through the winning neurons.

#### 11.8.3 The Controlled-Axis Result

The k-WTA sweep ran 18 conditions: 3 seeds crossed with 6 target fractions ($f \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.40\}$), 10 epochs per task. Measured mean observed activity tracked the target fraction almost exactly and monotonically across all three seeds:

| Target fraction | Observed activity |
|---|---|
| 0.01 | 0.012 |
| 0.05 | 0.050 |
| 0.10 | 0.101 |
| 0.20 | 0.198 |
| 0.30 | 0.280 |
| 0.40 | 0.327 |

The top fraction lands near 0.33 rather than 0.40 because k-WTA upper-bounds which neurons *may* fire; not all winners cross the firing threshold within the window. The resulting axis spans approximately 1–33%, clean and monotonic. No conditions were flagged dead.

Seed-averaged outcomes across the six target fractions are as follows:

| Target fraction | Final accuracy | Mean forgetting | CKA |
|---|---|---|---|
| 0.01 | 0.771 | 0.270 | 0.0132 |
| 0.05 | 0.828 | 0.199 | 0.0163 |
| 0.10 | 0.778 | 0.262 | 0.0136 |
| 0.20 | 0.815 | 0.220 | 0.0107 |
| 0.30 | 0.839 | 0.193 | 0.0093 |
| 0.40 | 0.877 | 0.147 | 0.0091 |

Cosine similarity between task-mean representations remained high throughout (approximately 0.73–0.99), consistent with the earlier finding that cosine similarity on raw mean vectors is insensitive to the representational structure the mediation hypothesis cares about. CKA is the meaningful overlap metric.

#### 11.8.4 Scientific Findings on the Controlled Axis

**H3 (inverted-U, moderate-sparsity optimum) is not supported.** Accuracy increases roughly monotonically with activity across the 1–33% range, and forgetting decreases roughly monotonically. There is no interior optimum. On Split-MNIST with this LIF configuration, denser activity is simply better within the tested range. This is a substantive negative result against the proposal's central framing — the "moderate sparsity near 20–40% is the sweet spot" prediction does not hold here.

**H2 (extreme sparsity harms accuracy) is weakly supported, as a soft gradient rather than a cliff.** Accuracy declines gently toward the sparse end (approximately 0.88 at 33% activity down to approximately 0.77 at 1%), and the network still learns at 1% activity. This is qualitatively different from the dead-network collapse seen in the fixed-threshold regime: the k-WTA mechanism prevents the hard cliff by construction, so the extreme-sparse regime is now characterized as a gradual accuracy penalty rather than a degenerate boundary condition.

**The mechanism signal — representational overlap tracking forgetting — is present and in the hypothesized direction.** CKA falls from approximately 0.016 at the sparse end to approximately 0.009 at the dense end, while forgetting also falls over the same range. Lower overlap co-occurs with less forgetting, consistent with the mediation hypothesis. However, this is activity-confounded: overlap and forgetting both co-vary with activity across the sweep, so this screening run cannot separate genuine mediation from mere co-variation with the common cause. Disentangling them is the job of the full study's formal mediation model, which estimates the indirect effect of sparsity on forgetting through overlap conditional on the direct effect.

**Net assessment.** The k-WTA switch is an engineering success: it delivers the controlled activity axis that is the prerequisite for everything downstream. The scientific picture is mixed and informative. The inverted-U did not materialize on Split-MNIST, which challenges the proposal's central framing and raises the value of two things: the formal mediation analysis (which can still support H4 even without an interior optimum, provided the indirect effect is nonzero) and possibly a harder benchmark such as Split-CIFAR, where representational interference may be severe enough that an interior optimum emerges. The negative H3 result sharpens the project rather than ending it; the novelty now leans more heavily on the mediation story than on the performance-optimum story.

### 11.9 The Mediation Test: No Evidence Beyond Activity Co-Variation

#### 11.9.1 Purpose and Scope

Phase C of the pilot ran a formal mediation analysis on the 18 k-WTA conditions (6 activity fractions $\times$ 3 seeds) to test H4 directly: does representational overlap (CKA) *mediate* the sparsity–forgetting relationship, or does it merely co-vary with activity alongside forgetting? This is an exploratory analysis. With n = 18 observations, three seeds, and a single benchmark, it is severely underpowered for confirmatory inference. The purpose is to screen whether the mediation signal is present at all before committing to the full study's seed count and benchmark scope. No p-values from this analysis are treated as confirmatory evidence.

The implementation used numpy-only OLS with a percentile bootstrap (no new dependency beyond the existing stack). All three variables — observed activity fraction ($X$), linear CKA between task representations ($M$), and mean forgetting ($Y$) — were standardized to zero mean and unit variance before estimation, so all path coefficients are in standard-deviation units and directly comparable in magnitude.

#### 11.9.2 Method

The mediation model follows the structure specified in Section 5.3, simplified to the pilot's single-mechanism, single-seed-level data. Four quantities are estimated:

- **Total effect** $c$: the regression coefficient of $Y$ on $X$ (activity predicting forgetting, ignoring overlap).
- **Path $a$**: the regression coefficient of $M$ on $X$ (activity predicting CKA overlap).
- **Path $b$**: the regression coefficient of $Y$ on $M$ conditional on $X$ (overlap predicting forgetting, with activity held constant).
- **Direct effect** $c'$: the regression coefficient of $Y$ on $X$ conditional on $M$.
- **Indirect effect** $a \times b$: the product of path $a$ and path $b$, estimated with a 95% percentile bootstrap CI (resampled over the 18 conditions with replacement, 10,000 iterations).

The proportion mediated is $(a \times b) / c$, reported for completeness but interpreted cautiously when the indirect effect is near zero or the wrong sign.

#### 11.9.3 Results

Standardized path estimates:

| Quantity | Estimate |
|---|---|
| Total effect $c$ (activity $\to$ forgetting) | $-0.483$ |
| Path $a$ (activity $\to$ CKA overlap) | $-0.805$ |
| Path $b$ (CKA overlap $\to$ forgetting $\mid$ activity) | $-0.159$ |
| Direct effect $c'$ (activity $\to$ forgetting $\mid$ overlap) | $-0.611$ |
| Indirect effect $a \times b$ | $+0.128$ |
| 95% bootstrap CI for $a \times b$ | $[-0.586,\ +0.994]$ |
| Proportion mediated $(a \times b) / c$ | $-0.265$ |

**Verdict: no evidence of mediation beyond activity co-variation.** The bootstrap CI for the indirect effect includes zero by a wide margin. The proportion mediated is negative and nonsensical ($-0.265$), which arises because the indirect effect and the total effect have opposite signs: the total effect of activity on forgetting is negative (more activity, less forgetting), but the indirect path through overlap is positive (more activity reduces overlap via path $a$, but lower overlap is associated with *more* forgetting via path $b$, not less). Path $b$ is weakly negative ($-0.159$) rather than the positive value the mediation hypothesis requires — the hypothesis predicts that lower overlap should predict lower forgetting conditional on activity, but the data show the opposite weak tendency once activity is held constant.

#### 11.9.4 Interpretation

The raw co-movement of CKA and forgetting visible in Section 11.8.4 is explained by both variables co-varying with activity: as activity increases, both CKA and forgetting fall together, producing the apparent correlation. Conditioning forgetting on activity leaves CKA with no additional, correctly-signed contribution. The overlap signal is not carrying information about forgetting beyond what activity already explains.

Combined with the H3 non-result from Section 11.8.4, the pilot has now produced two negative screening results on Split-MNIST: the inverted-U moderate-sparsity optimum did not appear, and the overlap-mediation mechanism did not survive conditioning on activity. Both H3 and H4 are unsupported at pilot scale on this benchmark.

This does not refute the underlying idea. The negative results are consistent with the pilot being underpowered and the benchmark being too easy. On Split-MNIST, the five tasks are binary digit-pair classifications sharing the same input space; representational interference is modest, the CKA range across the entire activity axis is minuscule (approximately 0.009–0.016), and the three variables (activity, CKA, forgetting) are near-collinear, making the b-path coefficient difficult to estimate reliably. The mediation model needs a setting where overlap varies substantially and independently of activity — which requires a harder benchmark and a wider, better-decoupled activity range.

#### 11.9.5 Caveats and Full-Study Implications

Several limitations of this analysis must be stated explicitly.

**Sample size.** n = 18 is far too small for reliable mediation estimation. The bootstrap CI spans 1.58 standard-deviation units, which reflects near-total uncertainty about the indirect effect. The analysis can rule out a very large mediation effect but cannot rule out a moderate one.

**Near-collinearity.** Activity, CKA, and forgetting are nearly collinear in this dataset: all three move together monotonically across the activity axis. Conditioning on activity in the b-path regression leaves very little residual variance in CKA to predict forgetting, making the b-path estimate noisy and unstable. A full study that decouples CKA from activity — for example by using a harder benchmark where different activity levels produce more varied overlap patterns — is required to estimate the b-path with any precision.

**CKA range.** The CKA values span only approximately 0.009–0.016 across the entire axis. This is a very narrow range for a mediator; small measurement noise can dominate the signal.

**Exploratory status.** This analysis was not pre-registered and was run after observing the k-WTA results. It is exploratory, not confirmatory. The negative result should be treated as a design signal, not as evidence against the mediation hypothesis.

**Full-study design implications.** The two negative screens sharpen rather than undermine the full study. The required changes are: (1) increase seeds to 8–10 to reduce estimation variance; (2) use a harder benchmark such as Split-CIFAR, where representational interference is more severe and the CKA range is likely wider; (3) design the activity sweep to decouple CKA from activity more effectively, for example by comparing conditions matched on activity but differing in sparsity mechanism (threshold vs. WTA vs. activity regularization), which the full study's multi-mechanism design already provides. A negative result on Split-MNIST with n = 18 is the expected output of a screening run; it does not license abandoning the mediation hypothesis, only redesigning the test.

### 11.10 Conv-SNN on CIFAR-10: A Fair Capacity-Pressure Test

#### 11.10.1 Motivation: The Flatten-MLP Was Too Weak to Create Real Pressure

The Split-MNIST results in §11.8 and §11.9 were obtained with a flat LIF-SNN whose input layer simply flattened the image into a 784-dimensional vector before feeding two 256-unit LIF hidden layers. On CIFAR-10, that architecture is genuinely weak: final average accuracy sat near 0.62 and was nearly flat across the entire activity axis. A model that cannot learn the task well cannot create real capacity pressure, and without capacity pressure the inverted-U predicted by H3 has no reason to appear. Testing H3 on a model that is already at ceiling-of-its-ability is an unfair test.

To give the hypotheses a fair hearing on a harder benchmark, a small spiking convolutional frontend was added. The architecture is three spiking convolutional layers (channels 3 to 16 to 32 to 64, 3x3 kernels, max-pool after each layer), feeding the same two 256-unit LIF hidden layers and per-task output heads used throughout the pilot. The k-WTA sparsity gate and the activity metric remain on the two fully connected hidden layers only, so the sparsity axis stays directly comparable across all models in the progression. The convolutional layers are not subject to k-WTA; they fire freely. This design choice is deliberate: the goal is to test whether the sparsity-forgetting relationship changes when the model is competent, not to study sparsity in convolutional layers.

#### 11.10.2 Result: The Conv-SNN Is Genuinely Competent on CIFAR

The Conv-SNN reaches a final average accuracy of approximately 0.71 across the activity range, compared with the flatten-MLP's flat 0.62. That is a real nine-point lift, not a marginal improvement, and it confirms that the convolutional frontend is doing meaningful feature extraction rather than merely adding parameters.

Seed-averaged outcomes by observed activity level are as follows:

| Observed activity | Final accuracy | Mean forgetting | CKA |
|---|---|---|---|
| ~1% | 0.609 | 0.249 | 0.0037 |
| ~5% | 0.687 | 0.254 | 0.0074 |
| ~9% | 0.705 | 0.255 | 0.0088 |
| ~16% | 0.711 | 0.248 | 0.0085 |
| ~18% | 0.706 | 0.256 | 0.0098 |
| ~20% | 0.713 | 0.226 | 0.0104 |

One feature of the activity axis is worth noting: observed activity tops out near 20% rather than the 33% seen on Split-MNIST. The convolutional frontend drives the fully connected layers less densely than direct pixel injection does, so the k-WTA gate operates on a somewhat sparser input distribution. The sparsity axis is still clean and monotonic; it simply does not extend as far toward the dense end.

#### 11.10.3 H2 (Extreme Sparsity Hurts Accuracy): Now Clearly Supported

On Split-MNIST, the accuracy penalty at 1% activity was modest and the network still learned reasonably well. On CIFAR with the Conv-SNN, the penalty is visible and meaningful: accuracy at approximately 1% activity is 0.609, while the plateau across the moderate-to-dense range sits near 0.71, a gap of roughly ten percentage points. This is the first result in the pilot that clearly supports H2. The decline is graded rather than catastrophic, consistent with the k-WTA mechanism preventing a hard dead-network cliff, but the penalty is large enough to be scientifically meaningful rather than a rounding artifact.

The flatten-MLP on CIFAR could not reveal this because its accuracy was already near 0.62 at all activity levels; there was no room for a sparsity penalty to show up against a flat baseline. The Conv-SNN, by being competent, creates the headroom that makes the penalty visible.

#### 11.10.4 H3 (Inverted-U Sweet-Spot): Still Not Supported

The accuracy curve rises from 0.609 at 1% activity to a plateau near 0.71 across the 9–20% range, then does not decline. This is a saturating curve, not an inverted-U. Dense activity is never worse than moderate activity; it is simply no better. The interior-peak requirement stated in §9.4 is not met: there is no fitted vertex strictly inside the tested range where accuracy peaks and then falls. H3 remains unsupported across all three models in the progression.

This is a consistent finding, not a benchmark-specific accident. On Split-MNIST with the flat LIF-SNN, the curve was monotonically rising. On CIFAR with the Conv-SNN, the curve rises and then saturates. Neither produces the inverted-U. The pattern across both architectures and both benchmarks is: more activity helps up to a point, then additional activity provides no further benefit. The "moderate sparsity is best" framing is not supported by the data.

#### 11.10.5 H4 (Overlap Mediates Forgetting): Still Not Supported — A Fourth Negative Screen

Forgetting is flat across the activity axis, ranging from approximately 0.23 to 0.26 with no discernible trend. CKA is small throughout (0.004 to 0.010) and if anything rises slightly with activity rather than falling. The raw co-movement that was visible on Split-MNIST is absent here.

A formal exploratory mediation analysis on the Conv-SNN CIFAR data (n = 18 conditions, same structure as §11.9) gives the following estimates:

- Total effect $c$ (activity to forgetting): $-0.003$
- Path $a$ (activity to CKA overlap): $+0.815$
- Path $b$ (CKA overlap to forgetting, conditional on activity): $+0.037$
- Indirect effect $a \times b$: $+0.030$
- 95% bootstrap CI for $a \times b$: $[-0.973,\ +0.729]$

Two features of this result are notable. First, path $a$ is positive: more activity associates with *more* representational overlap, the opposite of the direction the mediation hypothesis requires. On CIFAR, the convolutional frontend appears to produce representations that become more similar across tasks as the FC layers are allowed to be more active, rather than less similar. Second, the total effect is essentially zero ($-0.003$), meaning activity has no detectable relationship with forgetting on this benchmark at this scale. With both the total effect and the indirect effect near zero and the CI spanning nearly 1.7 standard-deviation units, there is no evidence of mediation. This is the fourth negative screen for H4 across the pilot: two benchmarks, two architectures, and two mediation analyses, all returning null results.

The null on CIFAR is not simply a repeat of the MNIST null. The mechanism is different: on MNIST, activity and forgetting co-varied but the mediation path did not survive conditioning; on CIFAR, activity and forgetting do not co-vary at all, and the a-path runs in the wrong direction. Both are negative for H4, but for distinct reasons that together paint a more complete picture of where the mediation hypothesis fails.

#### 11.10.6 Net Assessment: The Conv-SNN Completes the Three-Model Progression

The Conv-SNN result closes the pilot's empirical arc. The flatten-MLP on MNIST established the controlled activity axis and produced the first two negative screens (H3 and H4). The flatten-MLP on CIFAR showed that a weak model cannot create capacity pressure and therefore cannot test H2 or H3 fairly. The Conv-SNN on CIFAR provided the competent model the fair test required, and the results are clear: H2 is now supported (extreme sparsity carries a real accuracy penalty), H3 remains unsupported (the pattern is saturation, not an inverted-U), and H4 remains unsupported (a fourth negative screen, with the a-path running in the wrong direction on this benchmark).

The caveats from §11.9.5 apply here with equal force: n = 18, three seeds, exploratory and not pre-registered, not confirmatory. The Conv-SNN result is a design signal, not a settled finding. What it does establish is that the negative results for H3 and H4 are not artifacts of an underpowered model on an easy benchmark. They survive a competent model on a harder benchmark, which raises the bar for what the full study would need to show to overturn them.

---

## 12. Claim Boundaries

### 12.1 What This Project Does Not Claim

Several claims are tempting but not supported by the design, and stating them would misrepresent the contribution.

**"SNNs solve catastrophic forgetting."** This project tests whether spike sparsity *reduces* forgetting under controlled conditions. Even if H1 is supported, the reduction is partial and conditional on the sparsity level, the mechanism, and the benchmark. No claim of solving the problem is warranted.

**"Sparsity always improves continual learning."** H3 predicts an inverted-U: extreme sparsity hurts accuracy (H2), and the benefit is concentrated in the 20–40% range. Sparsity is not universally beneficial, and the project is designed to find the limits, not to advocate for sparsity as a general solution.

**"LIF results generalize to all SNNs."** The LIF model is the simplest standard SNN model. It lacks adaptive thresholds, refractory-period dynamics, dendritic computation, and the diversity of biological neuron types. Results from LIF networks may not transfer to adaptive LIF, Izhikevich, or other neuron models. This is stated as a limitation, not a caveat to be minimized.

**"Spike-count energy proxies prove hardware energy efficiency."** The energy proxy (spike count $\times$ synaptic operations) is a computational estimate that correlates with energy use on neuromorphic hardware under idealized assumptions. Real hardware energy depends on memory access patterns, data movement, chip architecture, and operating conditions. The proxy is useful for within-study comparisons but cannot be used to claim hardware energy efficiency without validation on actual neuromorphic hardware (Intel Loihi, IBM TrueNorth, or similar).

### 12.2 Biological Interpretation

Biological sparsity is used as motivation for the hypothesis, not as evidence of biological fidelity. The mechanisms that produce sparse activity in biological neural systems — inhibitory circuits, homeostatic plasticity, metabolic constraints — are complex and not modeled here. Threshold tuning, winner-take-all rules, and activity penalties are simplified computational controls. The paper should treat biological sparsity (Olshausen and Field, 1996; Buzsaki, 2006) as motivation for asking whether sparse activity reduces interference, not as evidence that the SNN model captures the relevant biology.

### 12.3 Acceptable Claims at Each Stage

**Acceptable pilot-stage claim** (if the correlation precondition holds):
> In a controlled Split-MNIST LIF-SNN setting, moderate spike sparsity is associated with lower representational overlap and lower forgetting.

**Acceptable full-study claim** (if the mediation analysis supports H4):
> Across multiple sparsity controls and continual learning methods, the relationship between spike sparsity and forgetting is non-linear, and reduced representational overlap between tasks mediates part of this relationship.

**Acceptable full-study claim** (if H4 fails but H1 holds):
> Moderate spike sparsity reduces forgetting in LIF-SNNs on Split-MNIST and Permuted-MNIST, but the reduction is not explained by reduced representational overlap; the mechanism appears to involve reduced plasticity or capacity rather than representational separation.

The third claim is a negative result for the mediation hypothesis but a positive result for the performance effect. It is still publishable and scientifically informative.

---

## 13. Limitations and Future Work

### 13.1 Current Limitations

**Simplified neuron model.** The LIF model lacks adaptive thresholds, refractory-period dynamics, and dendritic computation. Biological neurons show spike-frequency adaptation (firing rate decreases with sustained input), subthreshold resonance, and complex dendritic integration. These phenomena may be relevant to how biological sparsity interacts with memory consolidation. Later work should test adaptive LIF or Izhikevich models to determine whether the findings hold with richer neuron dynamics.

**Benchmark scope.** Split-MNIST and Permuted-MNIST are useful first benchmarks but relatively simple. MNIST images are 28$\times$28 grayscale, the tasks are binary or 10-class, and the dataset is well-curated with minimal label noise. Results on MNIST-scale benchmarks do not necessarily transfer to harder datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet) or to more realistic continual learning settings with gradual task boundaries, online data streams, or class imbalance. The difficulty of the continual learning *setting* is addressed within this study by evaluating both task-incremental and class-incremental settings; the remaining limitation is dataset complexity.

**Mechanism non-equivalence.** Threshold tuning, winner-take-all selection, and activity regularization produce different kinds of sparsity at the same active-neuron percentage. The per-mechanism analysis required by this project's design is the correct response to this limitation, but it means that results from one mechanism cannot be straightforwardly generalized to others. A unified theory of how different sparsity mechanisms affect representational overlap and forgetting remains an open problem.

**Biological simplification.** The sparsity controls are computational approximations and do not model the full biological basis of sparse coding. The project treats biological sparsity as motivation, not as a target for biological fidelity. This is the correct epistemic stance, but it means the project cannot make claims about biological neural systems.

**Energy proxy limitations.** The spike-count energy proxy is a computational estimate that does not account for memory access, data movement, or chip-specific characteristics. Hardware validation is required before any energy efficiency claim can be made.

### 13.2 Future Directions

**Adaptive sparsity.** Allow sparsity to change during training based on task difficulty or task identity. A network that automatically adjusts its sparsity level to minimize interference — without explicit task labels — would be a more powerful and more biologically plausible system.

**Recurrent architectures.** Extend the experiments to recurrent SNNs for temporal sequence tasks. Recurrent connections introduce additional dynamics (persistent activity, attractor states) that may interact with sparsity in ways not captured by feedforward networks.

**Neuromorphic hardware validation.** Test energy-efficiency claims on hardware such as Intel Loihi or IBM TrueNorth. This requires translating the trained SNN to a hardware-compatible format and measuring actual power consumption, which is a non-trivial engineering challenge but essential for any claim about neuromorphic efficiency.

**Biological experiments.** Collaborate with neuroscience labs to test whether biological sparsity correlates with reduced interference in relevant learning settings. This would require designing behavioral paradigms that parallel the continual learning benchmarks used here and measuring neural activity with sufficient resolution to quantify representational overlap.

**Richer neuron models.** Test whether the findings hold with adaptive LIF neurons (which show spike-frequency adaptation), Izhikevich neurons (which can exhibit bursting and resonance), or conductance-based models. If the sparsity-forgetting relationship is robust across neuron models, it is more likely to reflect a general principle rather than an artifact of the LIF model's simplicity.

**Online and blurry task boundaries.** The current design uses sharp, explicit task boundaries. Real continual learning often involves gradual distribution shifts without clear boundaries. Testing whether the sparsity mechanism operates in online or blurry-boundary settings would substantially increase the practical relevance of the findings.


---

## 14. Related Work in Context

### 14.1 Foundational Continual Learning

McCloskey and Cohen (1989) documented catastrophic interference in connectionist networks, establishing the problem that this project addresses. Kirkpatrick et al. (2017) introduced Elastic Weight Consolidation (EWC), which adds a quadratic penalty $\frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{1,i})^2$ to protect parameters important for earlier tasks, where $F_i$ is the diagonal Fisher information. Zenke, Poole, and Ganguli (2017) introduced Synaptic Intelligence (SI), which accumulates an online estimate of each parameter's importance during training. Lopez-Paz and Ranzato (2017) introduced Gradient Episodic Memory (GEM), which constrains gradient updates to not increase loss on stored examples from earlier tasks. Li and Hoiem (2017) introduced Learning without Forgetting (LwF), which uses the old model's predictions as soft targets when training on new tasks. Mallya and Lazebnik (2018) introduced PackNet, which prunes and freezes parameters after each task, allocating different subsets to different tasks.

Van de Ven and Tolias (2019) provided the three-scenario taxonomy (task-IL, domain-IL, class-IL) that is now the standard framing for continual learning research and that this project adopts throughout.

### 14.2 SNN Continual Learning with Sparsity

Shen, Ni, Xu, and Tang (2024, AAAI) is the closest prior work. They used trace-based K-Winner-Take-All and variable thresholds to create sparse selective activation for continual learning in SNNs, demonstrating that sparse activation can reduce forgetting. This paper occupies the "sparse activation reduces forgetting in SNNs" space and is the primary comparison point for novelty assessment.

Hammouamri, Masquelier, and Wilson (TMLR, OpenReview 15SoThZmtU) used firing-threshold modulation specifically to reduce forgetting in SNNs, directly occupying the threshold-control component of this project.

Meem, Nadid, and Mia (2026, arXiv:2602.12236) combined replay, learnable LIF neurons, and an adaptive spike scheduler for energy-aware continual learning in SNNs, occupying the spike-budget control space.

Additional recent work includes: active dendrites for efficient continual learning in time-to-first-spike SNNs (arXiv:2404.19419); TACOS, a task-agnostic SNN continual learning method using synaptic consolidation and neuromodulation (arXiv:2409.00021); gradient-free continual learning via inter-spike interval regularization (Roy et al., 2026, arXiv:2604.16496); SAFA-SNN for few-shot class-incremental learning (Zhang et al., 2025, arXiv:2510.03648); and CATFormer, combining continual learning with spiking transformers using dynamic-threshold LIF neurons (Nagabhushana et al., 2026, arXiv:2603.15184).

### 14.3 Representational Overlap and Forgetting

Ramasesh, Dyer, and Raghu (2020, ICLR) studied forgetting through hidden representations and task semantics, establishing the ANN-side link between representational overlap and forgetting. Doan et al. (2021) provided a theoretical analysis of catastrophic forgetting through the NTK overlap matrix, treating task overlap as central to forgetting. Abbasi et al. (2022) used k-winner sparse activations and heterogeneous dropout to reduce overlap between task representations in ANNs. Hu et al. (2024, ICML) used orthogonal sparse network partitioning to reduce interference and share useful knowledge.

Kornblith, Norouzi, Lee, and Hinton (2019, ICML) introduced CKA as a principled representation similarity measure, which this project uses as both a cross-task overlap metric and a representation drift metric.

### 14.4 Spike Sparsity and Energy Efficiency

Yan, Bai, and Wong (2024, arXiv:2409.08290) caution that SNN energy efficiency depends on time window, sparsity, memory access, and data movement — an important reference for avoiding simplistic energy claims. High-performance deep SNNs with 0.3 spikes per neuron (Nature Communications, 2024) demonstrate that high-performance SNNs can operate with very sparse spiking. Wei et al. (2025, arXiv:2505.10909) exploit hierarchical sparsity in SNN activations for hardware efficiency.

### 14.5 Biological Sparse Coding

Olshausen and Field (1996, Nature) showed that learning a sparse code for natural images produces basis functions resembling simple-cell receptive fields, providing the foundational biological motivation for sparse coding. Buzsaki (2006, Rhythms of the Brain) provides broad neuroscience background on neural rhythms and temporal neural activity. Lennie (2003, Current Biology) provides the classic biological energy-constraint reference, estimating that fewer than 1–4% of cortical neurons are active simultaneously.

---

## 15. Software and Reproducibility

### 15.1 Software Stack

The project uses the following software stack (from requirements.txt):

- **PyTorch 2.12.0** and **torchvision 0.27.0**: deep learning framework and dataset utilities
- **snntorch 0.9.4**: SNN simulation library implementing LIF neurons with surrogate gradient training
- **numpy 2.4.6**: numerical computing
- **scikit-learn 1.9.0**: linear probes, PCA, and statistical utilities
- **matplotlib 3.11.0**: visualization
- **pandas 3.0.3**: data management and tabular analysis
- **scipy 1.17.1**: statistical tests (t-tests, Wilcoxon, quadratic regression F-tests)

### 15.2 Reproducibility Requirements

All experiments are run with fixed random seeds that vary network initialization, data shuffling, and task order. Seeds are reported for every condition. The accuracy matrix $A \in \mathbb{R}^{T \times T}$ is saved after each task for every seed and condition. Hidden representations are saved for overlap analysis. Sparsity calibration tables (target activity, observed activity, threshold value) are reported for every condition.

The analysis uses observed active-neuron percentage throughout, never the calibration target. All statistical tests report corrected p-values with the correction family stated explicitly. Effect sizes (Cohen's $d$) and confidence intervals are reported alongside every pairwise comparison.

---

## 16. Summary of Design Decisions and Their Rationale

This section collects the key design decisions made in this project and the reasoning behind each, as an audit trail for future reference.

**Why task-incremental as the primary setting?** Task-IL isolates forgetting in the shared body, which is where the sparsity mechanism is hypothesized to operate. Class-IL forgetting is structurally unbounded (the output head is retrained for each new task), which makes it harder to attribute forgetting differences to the sparsity mechanism rather than to output-layer dynamics. Task-IL is the cleaner first test; class-IL is the harder validation.

**Why naive sequential training as the primary CL method?** Naive training exposes the maximum forgetting signal, making it the most sensitive test of whether sparsity provides any protection. Methods like EWC and replay already reduce forgetting through explicit mechanisms; testing sparsity on top of them would make it harder to detect the sparsity effect.

**Why three sparsity mechanisms?** Because threshold, WTA, and activity regularization produce different kinds of sparsity at the same active-neuron percentage. Using only one mechanism would leave open the question of whether the result is specific to that mechanism. Using all three, with per-mechanism analysis before pooling, provides a more complete picture.

**Why the mediation model rather than correlation?** Because correlation between sparsity, overlap, and forgetting does not establish that overlap is the pathway through which sparsity affects forgetting. The mediation model estimates the indirect effect conditional on confounds, which is the correct test of the mechanism claim.

**Why the interior-peak requirement for H3?** Because a significant quadratic coefficient can arise from a curve that merely flattens or bends at an extreme, not from a genuine inverted-U with an interior peak. The interior-peak requirement ensures that the non-linearity claim is substantive.

**Why 8–10 seeds for confirmatory conditions?** Because 3 seeds (the pilot minimum) are insufficient for confirmatory inference. With 8–10 seeds, the standard deviation across seeds is estimated with reasonable precision, and the paired t-tests have adequate power to detect the effect sizes predicted by H1 (30% forgetting reduction, which corresponds to a large Cohen's $d$ if the effect is real).

**Why fix the learning rate across sparsity levels?** Because a higher threshold reduces the effective gradient magnitude (fewer neurons receive gradient), which could reduce forgetting through a learning-rate effect rather than a sparsity effect. Fixing the learning rate and running a sensitivity analysis at 0.001, 0.0001, and 0.00001 controls for this confound.

**Why the count-matched dense-frozen baseline?** Because the most obvious alternative explanation for sparsity reducing forgetting is that fewer weights are updated during task B. The count-matched baseline directly tests this explanation by matching the number of unupdated weights while removing the sparse coding structure.

**Why the activation-dropout control?** Because random dropout at matched active% tests whether the effect requires a learned, input-dependent sparse code or merely fewer active units. If random dropout reproduces the forgetting reduction, the effect does not require structured sparse coding.

---

## References

1. McCloskey, M., and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.

2. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., and Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.

3. Zenke, F., Poole, B., and Ganguli, S. (2017). Continual learning through synaptic intelligence. *Proceedings of the 34th International Conference on Machine Learning*, 3987–3995.

4. Lopez-Paz, D., and Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems*, 30.

5. Li, Z., and Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12), 2935–2947.

6. Mallya, A., and Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 7765–7773.

7. van de Ven, G. M., and Tolias, A. S. (2019). Three scenarios for continual learning. *arXiv:1904.07734*.

8. Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). Similarity of neural network representations revisited. *Proceedings of the 36th International Conference on Machine Learning*, 3519–3529.

9. Ramasesh, V. V., Dyer, E., and Raghu, M. (2020). Anatomy of catastrophic forgetting: Hidden representations and task semantics. *International Conference on Learning Representations*.

10. Shen, J., Ni, W., Xu, Q., and Tang, H. (2024). Efficient spiking neural networks with sparse selective activation for continual learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(1), 611–619.

11. Hammouamri, I., Masquelier, T., and Wilson, D. G. (n.d.). Mitigating catastrophic forgetting in spiking neural networks through threshold modulation. *OpenReview*. https://openreview.net/forum?id=15SoThZmtU

12. Meem, A. T., Nadid, M. H., and Mia, M. Z. A. (2026). Energy-aware spike budgeting for continual learning in spiking neural networks for neuromorphic vision. *arXiv:2602.12236*.

13. Olshausen, B. A., and Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607–609.

14. Buzsaki, G. (2006). *Rhythms of the Brain*. Oxford University Press.

15. Lennie, P. (2003). The cost of cortical computation. *Current Biology*, 13(6), 493–497.

16. Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659–1671.

17. Kingma, D. P., and Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.

18. Doan, T., Bennani, M. A., Mazoure, B., Rabusseau, G., and Alquier, P. (2021). A theoretical analysis of catastrophic forgetting through the NTK overlap matrix. *Proceedings of Machine Learning Research*.

19. Abbasi, A., et al. (2022). Sparsity and heterogeneous dropout for continual learning. *Proceedings of Machine Learning Research*.

20. Hu, X., et al. (2024). Task-aware orthogonal sparse network for exploring shared knowledge in continual learning. *Proceedings of the 41st International Conference on Machine Learning*.

21. Yan, Z., Bai, Z., and Wong, W.-F. (2024). Reconsidering the energy efficiency of spiking neural networks. *arXiv:2409.08290*.

22. Roy, K., Kobayashi, Chakraborty, Talukder, and Alam. (2026). Gradient-free continual learning in spiking neural networks via inter-spike interval regularization. *arXiv:2604.16496*.

23. Farajtabar, M., Azizan, N., Mott, A., and Li, A. (2020). Orthogonal gradient descent for continual learning. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics*, 3762–3773.

24. Saha, G., Garg, I., and Roy, K. (2021). Gradient projection memory for continual learning. *International Conference on Learning Representations*.

25. Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., and Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54–71.

26. Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. (2017). SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. *Advances in Neural Information Processing Systems*, 30.

27. Wei, C., Duan, B., Guo, C., Zhang, J., Song, Q., Li, H., and Chen, Y. (2025). Phi: Leveraging pattern-based hierarchical sparsity for high-efficiency spiking neural networks. *arXiv:2505.10909*.

28. Nagabhushana, Agrawal, and Borthakur. (2026). CATFormer: When continual learning meets spiking transformers with dynamic thresholds. *arXiv:2603.15184*.

29. Softky, W. R., and Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs. *Journal of Neuroscience*, 13(1), 334–350.

30. Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.

---

*Note on citation status: Several 2025–2026 preprints in this reference list are marked as provisionally verified in RELATED_WORK_REFERENCES.md. Full bibliographic metadata (author lists, DOIs, venue) should be confirmed on the arXiv abstract pages before final submission. Entries 12, 22, 27, and 28 in particular require verification.*

