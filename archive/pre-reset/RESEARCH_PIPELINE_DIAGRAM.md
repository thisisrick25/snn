# Research pipeline and hypothesis diagram

This file gives an editable diagram view of the project. The first diagram shows the experiment pipeline. The second diagram shows the causal hypothesis. The third diagram shows what the pilot actually found and how the claim should be framed.

## 1. Research pipeline

```mermaid
flowchart TD
    A[Research question] --> B[Does spike sparsity reduce catastrophic forgetting in SNNs?]
    B --> C[Pilot setup]

    C --> C1[Dataset: Split-MNIST]
    C --> C2[Setting: task-incremental continual learning]
    C --> C3[Model: LIF-SNN]
    C --> C4[Training: naive sequential learning]
    C --> C5[Control: LIF firing threshold]

    C5 --> D[Threshold calibration]
    D --> D1[Nominal targets: 1%, 10%, 20%, 40%, 80% activity]
    D --> D2[Record achieved activity before and after training]

    D2 --> E[Sequential task training]
    E --> E1[Task 1: 0 vs 1]
    E --> E2[Task 2: 2 vs 3]
    E --> E3[Task 3: 4 vs 5]
    E --> E4[Task 4: 6 vs 7]
    E --> E5[Task 5: 8 vs 9]

    E --> F[Measure outcomes]
    F --> F1[Accuracy matrix A i,j]
    F --> F2[Final average accuracy]
    F --> F3[Mean forgetting and BWT]
    F --> F4[Spike rate and energy proxy]
    F --> F5[Representational overlap]

    F5 --> G[Mechanism analysis]
    G --> G1[Cosine similarity]
    G --> G2[PCA subspace overlap]
    G --> G3[Optional CKA]

    G --> H[Decision]
    H --> H1[Continue if sparsity affects forgetting and overlap]
    H --> H2[Revise if activity control or mechanism evidence is weak]
    H --> H3[Reframe if results contradict the original hypothesis]

    H1 --> I[Expansion only after pilot]
    I --> I1[Add MLP, ConvNet, Conv-SNN]
    I --> I2[Add Replay, EWC, SI, LwF, PackNet]
    I --> I3[Add Permuted-MNIST, Rotated-MNIST, CIFAR]
    I --> I4[Add WTA and activity regularization]
```

## 2. Hypothesis mechanism

```mermaid
flowchart LR
    A[Higher LIF firing threshold] --> B[Reduced spiking activity]
    B --> C[Fewer active hidden units]
    C --> D[Lower task-representation overlap]
    D --> E[Less interference during new-task learning]
    E --> F[Lower catastrophic forgetting]

    C --> G[Too few active units]
    G --> H[Lower representational capacity]
    H --> I[Lower task accuracy]

    F --> J[Expected tradeoff]
    I --> J
    J --> K[Too dense: interference]
    J --> L[Moderate sparsity: best retention]
    J --> M[Too sparse: underfitting]
```

## 3. Pilot result interpretation

```mermaid
flowchart TD
    A[Pilot result] --> B[Best performance at target 0.10]
    B --> B1[Final accuracy: 0.977 ± 0.001]
    B --> B2[Mean forgetting: 0.026 ± 0.002]

    A --> C[Worst performance at target 0.80]
    C --> C1[Final accuracy: 0.735 ± 0.040]
    C --> C2[Mean forgetting: 0.319 ± 0.050]

    A --> D[Mechanism evidence]
    D --> D1[PCA overlap vs forgetting: r = -0.873]
    D --> D2[Cosine overlap vs forgetting: r = +0.756]
    D --> D3[PCA supports mechanism more than cosine]

    A --> E[Important caveat]
    E --> E1[Nominal target activity did not equal achieved activity]
    E --> E2[Post-training activity drifted after threshold calibration]
    E --> E3[Manipulated variable is better described as threshold strength]

    E --> F[Reframed claim]
    D --> F
    B --> F
    C --> F
    F --> G[Increasing the LIF firing threshold reduced forgetting on Split-MNIST under naive sequential learning]
    G --> H[The reduction co-varied with PCA-subspace overlap between task representations]
```

## 4. Codebase flow

```mermaid
flowchart TD
    A[Local environment] --> A1[requirements.txt]
    A --> A2[.venv with torch, torchvision, snntorch, numpy, sklearn, matplotlib, pandas]

    A2 --> B[src/run_pilot.py]
    B --> C[src/data.py]
    C --> C1[Download and load MNIST]
    C --> C2[Build Split-MNIST tasks: 0v1, 2v3, 4v5, 6v7, 8v9]
    C --> C3[Create train/test loaders]
    C --> C4[Create calibration batch]

    B --> D[src/model.py]
    D --> D1[LIFNet: 784 -> 256 LIF -> 256 LIF -> task head]
    D --> D2[Returns logits, spike counts, and h2_mean representations]

    B --> E[src/sparsity.py]
    C4 --> E
    D --> E
    E --> E1[Calibrate firing threshold to target activity]
    E --> E2[Record achieved calibration activity]

    E2 --> F[src/train.py]
    C3 --> F
    D --> F
    F --> F1[Train tasks sequentially with naive learning]
    F --> F2[Evaluate all seen tasks after each task]
    F --> F3[Collect hidden representations after final task]

    F2 --> G[src/metrics.py]
    G --> G1[Accuracy matrix]
    G --> G2[Final average accuracy]
    G --> G3[Mean forgetting and BWT]
    G --> G4[Spike rate, active percentage, energy proxy]

    F3 --> H[src/overlap.py]
    H --> H1[Cosine overlap]
    H --> H2[PCA subspace overlap]
    H --> H3[Linear CKA]

    G --> I[results/metrics.csv]
    H --> I
    F2 --> J[results/runs/*.json]
    F3 --> J

    I --> K[src/plots.py]
    J --> K
    K --> K1[fig_forgetting_vs_activity.png]
    K --> K2[fig_accuracy_vs_activity.png]
    K --> K3[fig_retention_curves.png]
    K --> K4[fig_overlap_vs_activity.png]
    K --> K5[fig_overlap_vs_forgetting.png]
```

This diagram shows the implementation path. `run_pilot.py` is the orchestrator. It calls the data loader, model, threshold calibration, sequential training loop, metrics module, and overlap module. It writes the numeric outputs to `results/metrics.csv` and the per-run accuracy matrices to `results/runs/*.json`. `plots.py` then reads those files and produces the five pilot figures.

## 5. Caption for paper or slides

The study tests whether spike sparsity reduces catastrophic forgetting by lowering overlap between task representations. The pilot uses Split-MNIST, a task-incremental LIF-SNN, naive sequential learning, and threshold-controlled spiking activity. After each task, the model is evaluated on all previously seen tasks to build an accuracy matrix and compute forgetting. Hidden-layer spike representations are then compared across tasks using cosine similarity and PCA subspace overlap. The first pilot supports continuing the project, but with a narrower claim: threshold strength, rather than precisely controlled achieved activity, is the cleanest manipulated variable in the current implementation.

## 6. Notes for the next version

- Fix activity control before expanding the experiment grid. The nominal activity target did not map cleanly to post-training achieved activity.
- Treat PCA subspace overlap as the stronger mechanism signal for now. Cosine similarity gave a contradictory trend.
- Keep the claim bounded to Split-MNIST, naive sequential learning, and LIF-SNNs until the expanded experiments are complete.
- Do not claim that SNNs solve catastrophic forgetting or that spike-count energy proxies prove hardware energy efficiency.
