# Notes on ACE

*"Transformers Converge to Invariant Algorithmic Cores", arXiv 2602.22600v3, code at commit 527948b.*

- Short version: ACE is a useful tool. The paper's story (convergence, invariance) is its weakest part.
- How this was made: controls run with Claude; blind review and audit of the controls by Astra. No outside check. Every number has a script that calls your functions.

## Reproduces

- Markov: all 12 seeds match your saved ladder.
- Modular addition: retrained at paper settings (CPU, 3 models). Grokking, core sizes, mode counts follow your figures.
- IOI: `golden.py` gives your GPT-2 Small numbers to 3 decimals. Rebuilt cores match your saved scores in GPT-2 S/M/L, LLaMA-3.1-8B, Gemma-2-9B.

## Where ACE earns its place

**IOI against a pairing-aware baseline**

- Your PCA baseline is unpaired. ACE gets the role-swap pairs.
- Fairer baseline: PCA of the pair differences, same ranks.

| Model | ACE | Pair-difference PCA | Unpaired PCA (yours) | Gradients only |
|---|---:|---:|---:|---:|
| GPT-2 Small | .608 | .653 | .747 | .850 |
| GPT-2 Medium | .529 | .571 | .929 | .925 |
| GPT-2 Large | .600 | .676 | .925 | .958 |
| LLaMA-3.1-8B | .645 | .768 | 1.000 | .984 |
| Gemma-2-9B | .686 | .834 | .969 | .955 |

*Removal AUC, your held-out frames and new names, your layers and ranks. Lower is better.*

- Pairing does most of the work. Gradients add a real gain, larger in the big models.
- Neither half works alone in LLaMA or Gemma.
- ACE flips with fewer dimensions. Gemma, rank 4: flip AUC .37 (ACE), .83 (pair-difference PCA).

**Two-site map predicts an unseen intervention**

- Best experiment in the paper (§6, Table A11): 61–87% of the END response explained, six models, held-out sentences and names.
- Your saved results already show it beating a fixed signed direction.
- Astra, three GPT-2 models: rotating the source side of a transferred map destroys the prediction.

**Low-rank arithmetic near grokking**

- Epoch 800, each block of answer classes held out in turn: median held-out R² .98, .89, .94.
- Scrambled class order: negative.

**One layer before the readout**

- After long weight decay: a compact clock at layer 1. Necessity at 14 dims in all three models, 5–7 frequencies, held-out R² .94–.98. PCA misses it.
- Just after grokking: ACE reaches necessity at 15–20 dims; class means 22–27; PCA over 100.

## Where the story doesn't hold

**Markov**

- The optimal predictor needs only the current token.
- Random 4×64 lookup table, no transformer: spectral error .004. Trained ACE: .007.
- `markov_null_ladder_12seed/results.json` has a centred-readout arm: keep .748, remove .249, error .009. Not in the figure. Appendix B.1 says only ACE and PCA pass.

**Modular addition, last layer**

- Linear head: keeping the readout row space reproduces the logits; removing it leaves constants. True before training.
- Readout SVD, class means, PCA and the unprojected state give the same held-out operator as the ACE core.
- 26 modes is forced past about 49 dims. Random relabelings of the classes also give 26.
- Your held-out fit at the final epoch: 16–18. Logits: 18–21.

**"Invariant", "converge"**

- Three seeds: 5, 7, 8 frequencies. Different sets at layer 1.
- What is shared is the rotation, which is the task.

**Agreement** (your saved files; not rerun)

- Mixed-attractor prompts: difference of means beats ACE on keep, remove and flip in all six models. LLaMA: .952/.608/.152 against .778/.683/.395.
- Opposite-number prompts: ACE flips better.

**Grokking law**

- Fits well at 50% coverage.
- p = 53: delay about 4,200 steps at 40% coverage, about 450 at 75%. No term for coverage.
- p = 17, 75% coverage: three seeds first reach .99 test accuracy (not sustained). Below the fitted floor.

**Open**

- The no-decay branch restarts the optimizer at epoch 900; the decay arm does not. Late broadening could be either. Two missing arms are training.

## A better story

- Possible title: **"Balanced Truncation for Transformers: Small Causal Subspaces That Predict Interventions"**.
- Lead with the IOI sentence pair, the swap, the prediction.
- Pair-difference PCA beside ACE on the same page.
- Toy tasks as calibration: one real low-rank shift, one diagnostic that succeeds for a simple reason (Markov).

## Experiments worth running

1. **Equal search.** My comparison used layers and ranks chosen for ACE. Same layer and rank search for every ordering, scored on fresh names.
2. **Frozen map, new edits.** Partial swaps; writes off the mean role difference. Rank-one map and fixed response as baselines.
3. **Transfer against a target-only fit.** GPT-2 test: carried map helped one way (.84 against .74), hurt the other (.23 against .61).
4. **Invariance, as in the 2017 paper.** Sequence task where two hidden states give the same immediate answer and different futures. One alignment, transplanted states, predictions several steps ahead. A label code fails; a shared realisation passes.

## Limits

- 3 seeds for modular addition. CPU for toy tasks. One rented GPU for language models.
- Qwen2.5-32B not run. Agreement not rerun.
- Layer-1 "state" is both positions flattened; the paper never uses it.
- Your operator fit's ridge term is not normalised to centroid size. I rescaled at layer 1; mode counts there move a little with that choice.
- The p = 17 test and the GPT-2 transfer check are Astra's runs.

The control scripts, saved results, and reproduction commands are on GitHub in [bubskee/ace-controls](https://github.com/bubskee/ace-controls).
