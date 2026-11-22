# Notes on ACE

*“Transformers Converge to Invariant Algorithmic Cores,” arXiv v3. These controls use your [released code at commit 527948b](https://github.com/joshseth/cores/tree/527948bb3573199d123d331cb2adc1e54647b317).*

The two-site IOI experiment is what got me excited about ACE. You fit a map on ordinary sentence pairs, edit the repeated-name state in a held-out sentence, and predict how the final-token state changes. That prediction can fail. Across six models, your map explains 61–87% of the measured change **in the selected final-token coordinates** and beats a fixed signed response. I’d lead with that result. [Paper, §6 and Table A11](https://arxiv.org/abs/2602.22600v3)

## At a glance

| Topic | Finding |
|---|---|
| [IOI](#what-do-the-gradients-add-on-ioi) | Giving PCA the same role-swap pairs closes much of the gap to ACE. Gradients add a further gain on removal in all five models, largest in LLaMA and Gemma. |
| [Modular addition at the readout](#modular-addition-at-the-readout) | The answer shift generalizes across held-out classes near grokking. PCA, class means and the readout recover the same frequencies as ACE. |
| [One layer earlier](#one-layer-earlier) | Removal needs few ACE directions just after grokking; keeping needs many more. After long decay, a compact clock appears. |
| [Weight decay](#weight-decay-and-the-optimizer-restart) | Late broadening follows decay under a matched optimizer restart. |
| [Markov](#markov) | A random lookup table for the current token recovers the transition spectrum. |
| [Grokking delay](#grokking-delay-and-coverage) | Delay depends strongly on training coverage. p = 17 crosses .99 at 75% coverage. |
| [Agreement](#subject-verb-agreement) | The difference of means scores better on mixed-attractor prompts. ACE flips better on opposite-number prompts. |
| [Experiments](#experiments-id-be-most-interested-in) | Equal search on IOI; a frozen map on new edits; invariance in the system-drift sense. |

## What do the gradients add on IOI?

- Your ACE extraction uses role-swap pairs. The published PCA baseline uses unpaired activations. I gave PCA the same pairs by taking the uncentered SVD of their activation differences, matching ACE’s treatment of those differences.
- The comparison below uses your selected layers and ranks. The entries are held-out name-preference AUC after removing both subspaces. Chance is .5. [Control script](controls/ioi_ladder.py) · [Archived GPU results](results/ioi/)

| Model | S2 + END rank | ACE | Pair-difference SVD | Unpaired PCA | Gradients only |
|---|---:|---:|---:|---:|---:|
| GPT-2 Small | 14 + 2 | .608 | .653 | .747 | .850 |
| GPT-2 Medium | 12 + 37 | .529 | .571 | .929 | .925 |
| GPT-2 Large | 24 + 38 | .600 | .676 | .925 | .958 |
| LLaMA-3.1-8B | 32 + 35 | .645 | .768 | 1.000 | .984 |
| Gemma-2-9B | 64 + 79 | .686 | .834 | .969 | .955 |

**Keep and flip**

- At these ranks, pair-difference SVD alone keeps held-out AUC at or above .998 and brings it to .067 or below when either site is flipped.
- At smaller ranks the ordering can matter. In Gemma, flipping the repeated-name site at rank four gives .37 with ACE and .83 with pair-difference SVD. [Gemma rank curve](results/ioi/ioi_ladder_gemma-2-9b.json)

**Remove**

- ACE brings joint removal closer to chance in all five models.
- In Gemma, going from unpaired to paired PCA gains .135 AUC toward chance. Adding gradient information gains another .148.
- Only GPT-2 Medium reaches the paper’s .55 cutoff on these held-out prompts at the published ranks.
- Selecting layers and ranks independently for each method would tell us how much of the advantage survives an equal search budget.

**Transfer**

In a separate three-GPT-2 check with a smaller extraction name pool, I compared the carried map with a target-only fit given exactly the same calibration pairs. Entries are the share of the measured swap response explained.

| Direction | Carried map | Target-only fit |
|---|---:|---:|
| Medium → Small | .835 | .744 |
| Small → Large | .234 | .614 |

- The target-only fit won in four of six directions.
- Rotating the source map hurt prediction in all six.
- I’d like to see the matched-calibration comparison across families. [Transfer scores](results/transfer/transfer_intervals.log)

## What the toy tasks tell me

### Modular addition at the readout

**The answer shift generalizes near grokking.** At epoch 800 in the three CPU replicas, I held out each contiguous four-class block in turn. Entries are median held-out R² for the true answer shift.

| Basis | Replica 1 | Replica 2 | Replica 3 |
|---|---:|---:|---:|
| ACE | .978 | .890 | .939 |
| Activation PCA | .979 | .891 | .933 |
| Scrambled class order | negative | negative | negative |

That’s good evidence for learned arithmetic geometry near grokking. [Holdout results](results/toy/toy_checks.log)

**Other bases recover the same frequencies.**

- As §4 acknowledges, seeds can use different Fourier frequencies.
- At epoch 800, ACE, readout SVD, class means, PCA and the full state recovered the same occupied frequency bins within each seed in the paper’s held-out fit: 5, 7 and 8 bins respectively across seeds.
- My question is what this identifies about the computation beyond the geometry of the answers. [Basis comparison](results/toy/modadd_operator_bases.json)

**Two facts help interpret the late fits.**

- *Readout row space.* Projecting onto the **centered** linear readout’s row space preserves all logit differences at any checkpoint. Removing it leaves only fixed bias differences.
- *Full centroid span.* If 53 class centroids are affinely independent, their centered span has dimension 52. On that full span, *any* cyclic ordering induces an exact linear map with all 26 nonzero frequency pairs. This applies to an unregularized fit using every transition.
- In the high-rank terminal replicas, shuffled cycles often recover most or all of those bins. The held-out ridge fit in the ACE core at the same checkpoints gives 16–18 bins. [Checkpoint results](results/toy/modadd_controls_wd1.json)

### One layer earlier

**Epoch 800.** Removing the first 15–20 ACE directions disrupted performance. How many directions also suffice when kept? I tested both criteria at the **same prefix**, with both token positions flattened.

| Ordering | Replica 1 | Replica 2 | Replica 3 |
|---|---:|---:|---:|
| ACE | 95 | 105 | 127 |
| Gradients only | 23 | 37 | 75 |

- Editing just the last position gave the same ordering.
- The flattened-state intervention is an extra test I introduced. [Joint-rank check](results/toy/joint.log)

**Epoch 20,000, with continued decay.**

- All three models reached the removal threshold at ACE rank 14.
- At that rank, after matching centroid scales across layers, ACE recovered 5–7 frequencies with held-out R² of .94–.98.
- The class-mean basis recovered the same counts and similar fits.
- Ridge strength matters here because centroid scale changes substantially across layers. [Layer-one results](results/toy/modadd_layer1.json) · [Ridge check](results/toy/ridge.log)

### Weight decay and the optimizer restart

The published decay-off branch also restarts AdamW at epoch 900. Both missing arms have now been run, and late broadening follows decay. Entries are frequencies above 10% of peak in the logits at epoch 20,000, for the three replicas.

| | Optimizer retained | Optimizer reset at epoch 900 |
|---|---:|---:|
| Decay on | 21, 18, 21 | 16, 19, 12 |
| Decay off | 7, 7, 7 | 7, 7, 7 |

- The reset slows the broadening.
- All three forked arms also re-seed the random stream at epoch 900, and the original decay-on arm does not. [Restart results](results/toy/modadd_controls_restart.json) · [Decay on, optimizer reset](controls/modadd_wd1_reset_branch.py) · [Decay off, optimizer retained](controls/modadd_wd0_keepopt_branch.py)

### Markov

- **A lookup table recovers the spectrum.** A random lookup table for the current token recovers the transition spectrum with normalized error .004, compared with .007 for trained ACE. The observed sequence supplies the transitions. This limits what spectral recovery alone establishes about a learned algorithm. [Markov controls](results/toy/markov_controls.json)
- **A positive result inside the trained transformer.** Your saved centered-readout arm at rank three gives mean keep/remove accuracy .748/.249 and spectral error .009. I’d include it in Fig. A1 and qualify Appendix B.1’s “only ACE and PCA” line. [Author’s saved ladder, Readout-3 rows](https://github.com/joshseth/cores/blob/527948bb3573199d123d331cb2adc1e54647b317/supplement/markov_null_ladder_12seed/results.json)

### Grokking delay and coverage

- **Coverage matters.** The modulus curve fits well at 50% training coverage. At fixed p = 53, the three-seed mean delay was about 4,200 steps at 40% coverage and 450 at 75%. [Coverage sweep](results/toy/grok_share.jsonl)
- **Below the fitted cutoff.** At p = 17 and 75% coverage, the three seeds first reached .99 test accuracy at steps 2,555, 3,209 and 3,029 under the paper’s first-crossing rule. Accuracy later dipped in every run. This limits the interpretation of the fitted cutoff as an absolute architectural floor. The fit at 50% coverage survives. [Training records, native trainer’s `paper_grok` field](results/toy/grok17.jsonl)

### Subject-verb agreement

From your saved mixed-attractor runs, the difference of means scores better than ACE on keep, remove and flip in all six models under the paper’s directional scoring convention. LLaMA is the largest gap:

| LLaMA-3.1-8B, mixed attractors | Keep | Remove | Flip |
|---|---:|---:|---:|
| Difference of means | .952 | .608 | .152 |
| ACE | .778 | .683 | .395 |

- On the original opposite-number prompts, ACE flips more strongly in all three saved GPT-2 comparisons.
- I’d show both protocols together. [Mixed-attractor table](https://github.com/joshseth/cores/blob/527948bb3573199d123d331cb2adc1e54647b317/supplement/outputs/21_sva_controls_table.md) · [Original-protocol results](https://github.com/joshseth/cores/tree/527948bb3573199d123d331cb2adc1e54647b317/experiments/sva)

## Experiments I’d be most interested in

1. **Equal search on IOI.** Give ACE and pair-difference SVD the same candidate layers, rank pairs and search budget. Freeze each choice and score on fresh names and frames, with paired uncertainty intervals across frames.
2. **Freeze the map, change the edit.** Predict the downstream state and name-margin changes after partial swaps and writes orthogonal to the mean role difference. Compare with a fixed signed response and a rank-one map with input-dependent amplitude.
3. **Test invariance in the sense of the earlier system-drift work.** Use a sequence task where two states can have the same immediate answer and different future answers. Fit one cross-model state alignment, freeze it, transplant states, and predict several outputs under different suffixes. That would give the shared-realization hypothesis a direct test.

## Provenance

Take this for what it is — careful AI-assisted replication work by an independent reader.

- Claude helped run the initial experiments.
- Astra read the paper before inspecting the controls, audited the code, reran the toy analyses, and ran the p = 17 and small-pool transfer checks.
- The IOI table comes from five archived full-pool GPU runs. The GPT-2 transfer check used 29 extraction names.
- Qwen has no completed full-pool control here, and the agreement figures come from your saved files.

[Scripts, recorded outputs and reproduction commands](README.md)
