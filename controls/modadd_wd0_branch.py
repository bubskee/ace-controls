"""Run phase 2 of the repository's run_modadd.run_train (the branch without weight decay) on its own.

The repository's script trains the weight-decay arm to epoch 20,000 and only then starts the branch from
the epoch-900 checkpoint. This runs the branch alongside, with the same call and arguments as phase 2 of
run_train (copied below from src/cores/run_modadd.py @ 527948b), so both arms finish sooner.

Usage (from the repository root):  python ../controls/modadd_wd0_branch.py
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

common = dict(lr=R.LR, batch_size=R.BATCH_SIZE, k_models=R.K_MODELS, seed=R.SEED)
branch_ckpt = R.WD1_DIR / f"epoch_{R.BRANCH_EPOCH:06d}.pt"
assert branch_ckpt.exists(), branch_ckpt
seed_all(R.SEED)
train(R.WD0_DIR, E2Args(**common, epochs=R.TOTAL_EPOCHS, weight_decay=0.0, save_every=R.DENSE_SAVE,
                        resume_from=str(branch_ckpt), save_every_late=R.SPARSE_SAVE, late_after=R.LATE_AFTER,
                        reset_optimizer=True), vocab_size=R.VOCAB_SIZE)
