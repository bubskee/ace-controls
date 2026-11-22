"""Optimizer-restart control for the weight-decay comparison.

The repository's two arms differ in more than weight decay. The decay arm trains straight through from epoch 0.
The no-decay arm is a branch from the epoch-900 checkpoint that also starts a fresh optimizer (reset_optimizer=True)
and re-seeds the random stream (seed_all before the call). This script runs the missing arm: the same branch call
as phase 2 of run_modadd.run_train @ 527948b (see modadd_wd0_branch.py), with weight decay left at 1.

  decay on,  no restart   ckpts/wd1         (repository phase 1)
  decay off, restart      ckpts/wd0         (repository phase 2)
  decay on,  restart      ckpts/wd1_reset   (this script)

Usage (from the repository root):  python ../controls/modadd_wd1_reset_branch.py
"""
import functools
import sys
from pathlib import Path

import torch

torch.backends.mps.is_available = functools.lru_cache(lambda: False)  # CPU; see run_cpu.py
sys.path.insert(0, str(Path.cwd() / "src"))

import cores.run_modadd as R  # noqa: E402
from cores.modular_addition_experiment import E2Args, train  # noqa: E402
from cores.utils import seed_all  # noqa: E402

OUT_DIR = R.BASE / "ckpts" / "wd1_reset"
common = dict(lr=R.LR, batch_size=R.BATCH_SIZE, k_models=R.K_MODELS, seed=R.SEED)
branch_ckpt = R.WD1_DIR / f"epoch_{R.BRANCH_EPOCH:06d}.pt"
assert branch_ckpt.exists(), branch_ckpt
seed_all(R.SEED)
train(OUT_DIR, E2Args(**common, epochs=R.TOTAL_EPOCHS, weight_decay=R.WD, save_every=R.DENSE_SAVE,
                      resume_from=str(branch_ckpt), save_every_late=R.SPARSE_SAVE, late_after=R.LATE_AFTER,
                      reset_optimizer=True), vocab_size=R.VOCAB_SIZE)
