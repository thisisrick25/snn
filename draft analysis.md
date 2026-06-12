# Draft Analysis: Investigating the Relationship Between Spike Sparsity and Catastrophic Forgetting

## Verdict
**Promising research direction with a clear central question, but the current document is an *idea sketch* rather than a paper draft.** It requires significant structural and content development before submission.

---

## Strengths
1. **Clear Central Question**: The primary question, "Does increasing spike sparsity reduce catastrophic forgetting?" is well-scoped and testable.
2. **Sound Biological Motivation**: The contrast between ANN catastrophic forgetting and human sparse neural activity provides a strong narrative hook.
3. **Defensible Novelty Claim**: The sparsity → representation drift → forgetting causal chain is underexplored in the literature.
4. **Well-Chosen Evaluation Metrics**: Accuracy, forgetting score, spike rate, sparsity index, and energy proxy cover the key dimensions.

## Critical Issues
1. **No Actual Paper Structure**: The document lacks Abstract, Introduction, Method, Experiments, Results, and Conclusion sections. It is currently a brainstorm/outline.
2. **Insufficient Contribution Depth**: Only 2 arxiv references are cited. A thorough Related Work section is needed to contrast with prior SNN continual learning work, sparsity studies in ANNs, and standard CL methods in non-spiking networks.
3. **Method Section is a Skeleton**: Missing details on architecture, datasets, and precise quantification of forgetting.
4. **Vague Hypotheses**: Claims like "We expect Too Dense → Optimal → Too Sparse" are hand-wavy and need to be operationalized with specific sparsity levels and metrics.
5. **No Experimental Evidence**: This is still a proposal with no results, tables, or figures.
6. **Ambiguity on Novelty**: Need to clarify how sparsity will be linked to the *mechanism* of forgetting (e.g., via Fisher information, synaptic overlap, or hidden-state drift).
7. **Underdeveloped Energy Evaluation**: The "Energy Proxy" needs validation against standard neuromorphic literature.
8. **Unclear Continual Learning Setup**: Missing details on whether this is task-incremental or class-incremental, task balancing, and replay buffer sizing.

## Actionable Recommendations
1. **Add Structure**: Convert this into a full paper with Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion.
2. **Strengthen Related Work**: Cite at least 10-15 papers covering SNN continual learning, sparsity in ANNs, and CL methods in non-spiking networks.
3. **Operationalize Hypotheses**: Define specific sparsity regimes (e.g., 1%, 5%, 10%, 20%, 50%, 80%) with numerical predictions.
4. **Add Experiments**: Run experiments and include figures: (a) Accuracy vs. Sparsity, (b) Forgetting vs. Sparsity, (c) Energy vs. Accuracy tradeoff.
5. **Mechanism Analysis**: If possible, analyze *why* sparsity helps (e.g., measure overlap in weight updates or hidden activations across tasks).
6. **Clean Content**: Remove the "Idea A/B/C" notes and crossed-out text.
