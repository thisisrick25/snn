# Research progress report

This report tracks the current state of the research on spike sparsity and catastrophic forgetting in continual learning spiking neural networks.

## 1. Current research direction

- Main problem:
  - Neural networks often forget old tasks after learning new ones.
  - This is catastrophic forgetting in continual learning.
  - This project studies the problem in spiking neural networks (SNNs).
- Current research question:
  - Does changing spike sparsity in a LIF-SNN affect catastrophic forgetting during continual learning?
- Working mechanism:
  - Higher LIF firing threshold changes spiking behavior.
  - Changed spiking behavior changes which hidden neurons are active.
  - Changed hidden activity changes task-representation overlap.
  - Lower overlap may reduce interference between tasks.
  - Lower interference should reduce forgetting.
- Current claim boundary:
  - This is not a claim that SNNs solve catastrophic forgetting.
  - The current evidence supports a narrower threshold-strength pilot result on Split-MNIST.

## 2. Research planning completed

- `RESEARCH_IDEA_REFINED.md`
  - Refined the original idea into clearer research questions, testable hypotheses, mechanism tests, limitations, and safe claim boundaries.
  - Clarified that simple MLP and LIF models are controlled first-stage models, not state-of-the-art evidence.
- `RELATED_WORK_REFERENCES.md`
  - Collected related papers on SNN continual learning, threshold modulation, spike budgeting, sparse selective activation, representation overlap, catastrophic forgetting, and sparse coding.
- `EXPERIMENT_PROTOCOL.md`
  - Defined the pilot before implementation.
  - Locked the first experiment to Split-MNIST, LIF-SNN, naive sequential learning, threshold control, five nominal activity targets, and three seeds.
- `RESEARCH_PIPELINE_DIAGRAM.md`
  - Added diagrams for the research pipeline, hypothesis mechanism, pilot interpretation, and codebase flow.
- `TECHNICAL_OVERVIEW.md`
  - Explains the research, codebase, experiment, findings, caveats, and next steps in plain technical language.

## 3. Codebase work completed

- Environment:
  - Created `.venv/` with Python 3.12.
  - Installed `torch`, `torchvision`, `snntorch`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`, and `scipy`.
  - Added `requirements.txt` and updated `.gitignore`.
- `src/data.py`
  - Downloads MNIST.
  - Builds Split-MNIST tasks: `0v1`, `2v3`, `4v5`, `6v7`, `8v9`.
  - Creates train/test loaders and calibration batch.
- `src/model.py`
  - Implements `LIFNet` using `snntorch`.
  - Architecture: `784 -> 256 LIF -> 256 LIF -> task-specific 2-class head`.
  - Returns logits, hidden-layer spike counts, and `h2_mean` representations.
- `src/sparsity.py`
  - Calibrates LIF firing threshold to target active-neuron percentage.
  - Records achieved activity because nominal targets did not match post-training activity cleanly.
- `src/train.py`
  - Runs naive sequential training.
  - Evaluates all seen tasks after each task.
  - Builds the accuracy matrix and collects hidden representations.
- `src/metrics.py`
  - Computes final average accuracy, forgetting, backward transfer, spike statistics, and energy proxy.
- `src/overlap.py`
  - Computes cosine overlap, PCA subspace overlap, and linear CKA.
- `src/run_pilot.py`
  - Runs all pilot configurations.
  - Writes `results/metrics.csv` and `results/runs/*.json`.
- `src/plots.py`
  - Creates five pilot figures: forgetting vs activity, accuracy vs activity, retention curves, overlap vs activity, and overlap vs forgetting.

## 4. Pilot experiment completed

- Pilot size:
  - 5 nominal threshold/activity targets.
  - 3 random seeds.
  - 15 total runs.
- Pilot setup:
  - Dataset: Split-MNIST.
  - Setting: task-incremental continual learning.
  - Model: feedforward LIF-SNN.
  - Learning method: naive sequential learning.
  - Control variable: LIF firing threshold.
  - Nominal activity targets: `1%`, `10%`, `20%`, `40%`, `80%`.
  - Training: 10 epochs per task, Adam, learning rate `0.001`, batch size `128`.
- Pilot outputs:
  - `results/metrics.csv`
  - `results/runs/*.json`
  - `fig_forgetting_vs_activity.png`
  - `fig_accuracy_vs_activity.png`
  - `fig_retention_curves.png`
  - `fig_overlap_vs_activity.png`
  - `fig_overlap_vs_forgetting.png`

## 5. Pilot results

| Nominal target | Trained activity | Final accuracy | Mean forgetting | Cosine overlap | PCA overlap |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0.654 +/- 0.029 | 0.942 +/- 0.007 | 0.068 +/- 0.009 | 0.976 | 0.701 |
| 0.10 | 0.443 +/- 0.037 | 0.977 +/- 0.001 | 0.026 +/- 0.002 | 0.945 | 0.676 |
| 0.20 | 0.367 +/- 0.012 | 0.952 +/- 0.023 | 0.056 +/- 0.029 | 0.943 | 0.647 |
| 0.40 | 0.357 +/- 0.030 | 0.876 +/- 0.016 | 0.149 +/- 0.020 | 0.980 | 0.594 |
| 0.80 | 0.462 +/- 0.009 | 0.735 +/- 0.040 | 0.319 +/- 0.050 | 0.993 | 0.526 |

- Best condition:
  - Target `0.10`.
  - Final accuracy: `0.977 +/- 0.001`.
  - Mean forgetting: `0.026 +/- 0.002`.
  - This condition had the best retention and highest final accuracy.
- Worst condition:
  - Target `0.80`.
  - Final accuracy: `0.735 +/- 0.040`.
  - Mean forgetting: `0.319 +/- 0.050`.
  - This condition had the most forgetting and lowest final accuracy.

## 6. Correlation results

- Trained activity vs mean forgetting: `r = -0.068`.
- Trained activity vs final accuracy: `r = +0.066`.
- Trained activity vs cosine overlap: `r = +0.326`.
- Trained activity vs PCA overlap: `r = +0.439`.
- Cosine overlap vs mean forgetting: `r = +0.756`.
- PCA overlap vs mean forgetting: `r = -0.873`.
- PCA overlap vs trained activity: `r = +0.439`.

## 7. Main findings

- The project should continue.
  - The pilot produced meaningful differences across threshold conditions.
  - The result is strong enough to justify another experiment.
- The strongest performance came from the `0.10` target condition.
  - It had the highest final accuracy and the lowest mean forgetting.
  - This supports a threshold-strength version of the original idea.
- PCA overlap gave the strongest mechanism signal.
  - PCA overlap vs mean forgetting was `r = -0.873`.
  - This suggests subspace structure may matter more than simple task-mean similarity.
- Cosine overlap contradicted the PCA signal.
  - Cosine overlap vs mean forgetting was `r = +0.756`.
  - The next experiment should investigate why cosine and PCA disagree.
- The full spike-sparsity hypothesis is not proven yet.
  - Threshold calibration did not precisely control post-training achieved activity.
  - The manipulated variable is better described as LIF threshold strength, not exact spike sparsity.

## 8. Major caveats

- Nominal activity target did not equal achieved activity after training.
  - The threshold was calibrated before training.
  - Activity drifted after training.
  - This means post-training spike sparsity was not precisely controlled.
- The `0.80` target was not a clean dense baseline.
  - The model should not be described as reaching 80% active neurons.
- Cosine and PCA overlap disagreed.
  - They measure different properties of representation overlap.
  - Both should stay in the analysis for now.
- The pilot is intentionally limited.
  - It uses Split-MNIST, a feedforward LIF-SNN, and naive sequential learning only.
  - It does not yet include ANN baselines, Replay, EWC, SI, LwF, WTA, or activity regularization.

## 9. Current safe claim

- Claim paragraph:
  - This pilot does not yet prove that spike sparsity causally reduces catastrophic forgetting. Instead, it shows that LIF threshold strength can substantially change forgetting behavior in a Split-MNIST task-incremental setting. The best condition, nominal target 0.10, achieved the highest final accuracy (0.977 ± 0.001) and lowest mean forgetting (0.026 ± 0.002), while the nominal 0.80 condition performed worst. Representation analysis further showed that PCA-subspace overlap correlated strongly with forgetting, although cosine overlap did not agree. These findings support continuing the study, but the next experiment must improve activity control before making a stronger sparsity-based claim.
- Safe claim:
  - Increasing the LIF firing threshold, which suppresses dense spiking behavior, reduced catastrophic forgetting on Split-MNIST under naive sequential training. This reduction co-varied with PCA-subspace overlap between task representations.
- Keep this claim narrow:
  - Split-MNIST only.
  - Feedforward LIF-SNN only.
  - Naive sequential learning only.
  - Threshold strength, not perfectly controlled spike sparsity.

## 10. Claims to avoid

- Do not claim that SNNs solve catastrophic forgetting.
- Do not claim that sparsity always improves continual learning.
- Do not claim that LIF results generalize to all SNN models.
- Do not claim that the spike-count energy proxy proves hardware energy efficiency.
- Do not claim that achieved activity was precisely controlled.

## 11. Next steps

- Fix activity control before expanding.
  - The main weakness is that nominal target activity did not match post-training achieved activity.
  - Possible fixes: recalibrate during training, recalibrate after each task, add activity regularization, use winner-take-all control, or use homeostatic threshold adaptation.
- Repeat the pilot after fixing activity control.
  - Keep the same simple setup first: Split-MNIST, LIF-SNN, naive sequential learning.
  - This isolates whether cleaner activity control changes the result.
- Treat PCA overlap as the primary mechanism signal for now.
  - PCA overlap had the strongest relationship with forgetting.
  - Cosine overlap should remain in the analysis because it contradicted PCA.
- Add WTA and activity regularization after the threshold result is understood.
  - WTA can force a fixed number of active neurons.
  - Activity regularization can penalize spike rate during training.
- Add baselines after the mechanism is clearer.
  - Add MLP, ConvNet, and possibly Conv-SNN.
  - Add Replay, EWC, SI, and LwF.
- Update the paper after the next experiment.
  - The current pilot is useful, but the paper claim should wait until activity control is cleaner.

## 12. Current status

- Status: Continue, but reframe.
- Meaning:
  - Continue the research because the pilot found a useful signal.
  - Reframe the current result as a threshold-strength pilot, not as a final proof that precise spike sparsity causally reduces forgetting.
