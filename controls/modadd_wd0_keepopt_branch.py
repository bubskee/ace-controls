"""Second optimizer-restart control for the weight-decay comparison: decay off, optimizer state kept.

Completes the two-by-two started by modadd_wd1_reset_branch.py. The repository's train() supports this directly:
without reset_optimizer it loads the saved optimizer state and then sets the weight decay to the requested value.
Like the other branches it starts from the epoch-900 checkpoint and re-seeds the random stream.

  decay on,  no restart   ckpts/wd1          (repository phase 1; never re-seeded)
  decay off, restart      ckpts/wd0          (repository phase 2)
  decay on,  restart      ckpts/wd1_reset    (modadd_wd1_reset_branch.py)
  decay off, no restart   ckpts/wd0_keepopt  (this script)

Usage (from the repository root):  python ../controls/modadd_wd0_keepopt_branch.py
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

OUT_DIR = R.BASE / "ckpts" / "wd0_keepopt"
common = dict(lr=R.LR, batch_size=R.BATCH_SIZE, k_models=R.K_MODELS, seed=R.SEED)
branch_ckpt = R.WD1_DIR / f"epoch_{R.BRANCH_EPOCH:06d}.pt"
assert branch_ckpt.exists(), branch_ckpt
seed_all(R.SEED)
train(OUT_DIR, E2Args(**common, epochs=R.TOTAL_EPOCHS, weight_decay=0.0, save_every=R.DENSE_SAVE,
                      resume_from=str(branch_ckpt), save_every_late=R.SPARSE_SAVE, late_after=R.LATE_AFTER,
                      reset_optimizer=False), vocab_size=R.VOCAB_SIZE)
