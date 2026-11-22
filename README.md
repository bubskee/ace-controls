# ACE controls

Independent experiments accompanying a reading of Joshua S. Schiffman’s *Transformers Converge to Invariant Algorithmic Cores*. Start with the two-site indirect object identification (IOI) experiment. ACE uses paired prompts and gradients. How much does using the gradients add once a baseline sees exactly the same pairs? [Paper](https://arxiv.org/abs/2602.22600v3) · [Author’s code at the tested commit](https://github.com/joshseth/cores/tree/527948bb3573199d123d331cb2adc1e54647b317)

## Inspect the results

This command reads the archived measurements using only the Python standard library.

```sh
python checks/show_ioi.py
```

| Model | S2 + END rank | ACE | Paired-difference PCA | Unpaired PCA | Gradients only |
|---|---:|---:|---:|---:|---:|
| GPT-2 Small | 14 + 2 | .608 | .653 | .747 | .850 |
| GPT-2 Medium | 12 + 37 | .529 | .571 | .929 | .925 |
| GPT-2 Large | 24 + 38 | .600 | .676 | .925 | .958 |
| LLaMA-3.1-8B | 32 + 35 | .645 | .768 | 1.000 | .984 |
| Gemma-2-9B | 64 + 79 | .686 | .834 | .969 | .955 |

The entries are held-out two-name margin AUC after removing both subspaces. Chance is .5. The pair-aware baseline preserves and flips the answer nearly as well as ACE in these runs. ACE consistently brings removal closer to chance at the layers and ranks chosen for ACE. An independent layer and rank search for each method remains open. The five full-name-pool GPU result files include the keep, remove and single-site flip measurements. Qwen has no completed control run here. [Measurements](results/ioi/) · [Scoring script](controls/ioi_ladder.py)

## Reproduce a comparison

From the root of this repository, set up the author’s code in a Python environment. The upstream project declares its dependencies in its pyproject. The local CPU checks used Python 3.12, PyTorch 2.14.1, NumPy 2.5.4 and SciPy 1.18.1. GPU archives come from a separate environment.

```sh
git clone https://github.com/joshseth/cores.git cores
git -C cores checkout --detach 527948bb3573199d123d331cb2adc1e54647b317
python -m pip install -e ./cores
mkdir -p runs
```

**Rescore transfer without downloading a model.** This uses the included GPT-2 reduced coordinates and measured responses to a partner-state swap. It compares a transferred source operator, a direct target-only fit on the same calibration pairs, the target’s own operator, and ten rotations of the source operator. Both calibration folds hold out entire sentence frames. The source helps in both directions into GPT-2 Small. Direct target calibration scores higher in the four directions into Medium or Large. This separate experiment used a 29-name extraction pool. [Coordinates and recorded scores](results/transfer/)

```sh
python checks/transfer_check.py --score-only
```

**Re-extract the full-pool GPT-2 Small IOI comparison.** The upstream loader uses the GPT-2 checkpoint and NLTK names corpus. The archived GPT-2 runs have 1,057 extraction names. Check the output’s `extraction_names` and `standins` fields before comparing values. The GPT-2 GPU archives sweep equal site ranks through 64, and the current script also evaluates 96 and 128. Its random basis draw can therefore differ.

```sh
python controls/ioi_ladder.py cores gpt2 gpt2 6:9 14:2 cuda runs/ioi_gpt2.json
```

## Toy experiments

| Question | Entry point |
|---|---|
| Does the Markov spectrum also appear in a random codebook for the current token? | [Markov controls](controls/markov_controls.py) |
| Which modular-addition shift frequencies survive changing the basis, class order or holdout block? | [Class-order and readout controls](controls/modadd_controls.py) · [Alternative bases](controls/modadd_operator_bases.py) · [All contiguous holdouts](checks/toy_checks.py) |
| How does moving one layer earlier change the result? | [Layer-one Jacobians and ablations](controls/modadd_layer1.py) · [Joint keep/remove rank at one or both positions](checks/layer1_joint.py) · [Ridge sensitivity](controls/modadd_layer1_ridge_check.py) |
| How does training coverage affect grokking delay? | [Training sweep](controls/grok_share.py) · [Refit on archived runs](controls/grok_share_fit.py) · [Three p = 17 runs at 75% coverage](results/toy/grok17.jsonl) |

```sh
python controls/grok_share_fit.py cores results/toy/grok_share.jsonl
python controls/markov_controls.py cores runs/markov.json

# Generate the two original arithmetic checkpoint branches before analyzing them.
(cd cores && python ../controls/run_cpu.py modadd.py --only train)
MODADD_ARMS=wd1 python controls/modadd_controls.py cores runs/modadd.json 800 20000
python checks/toy_checks.py
python checks/layer1_joint.py
```

The archived arithmetic measurements use three CPU-trained replicas. The two additional training scripts vary the optimizer restart at the epoch-900 fork. They are included for inspection; the results directory covers the original decay-on and decay-off branches. [Decay on, optimizer reset](controls/modadd_wd1_reset_branch.py) · [Decay off, optimizer retained](controls/modadd_wd0_keepopt_branch.py)

**Provenance.** These are independent, AI-assisted controls. The toy analyses reuse the upstream implementations and were rerun locally. The full-pool IOI table is read from archived GPU outputs. The small-pool transfer test was run locally on three GPT-2 sizes. The subject–verb agreement comparisons discussed separately come from the author’s saved files. In the archived grokking rows use `paper_mem` and `paper_grok` for the original trainer’s one-based steps. The auxiliary recorder was one step early for both crossings, leaving their difference intact. The public script corrects the counter and restores the evaluation hook between worker jobs. Selected toy JSON files retain Python’s NaN encoding for undefined fits.

The upstream MIT notice is included for adapted code. [Upstream license](LICENSE.upstream)
