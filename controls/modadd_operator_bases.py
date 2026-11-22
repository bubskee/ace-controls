"""The paper's shift-operator fit, run in bases other than the ACE core.

Question: does the "mechanism" read off the core (eigenvalues of the fitted shift operator on the unit
circle, counted by frequency bin) depend on ACE, or does any basis of the same rank give it?

For every checkpoint analysed by modadd_controls.py, at the paper's core rank r (r_nec from that run):
  ACE            first r ACE directions (the paper's core)
  Readout SVD    top right singular vectors of the centred lm_head weight
  Class means    top right singular vectors of the centred class-mean activations
  Activation PCA top principal directions of the last-token activations
  Full state     no projection (identity basis, 128 dims)
  Random         a random orthonormal basis of rank r (5 draws)
Readout SVD and Class means have at most 52 directions, so they use min(r, 52).

Centroids (token_core_reps), the operator fit (fit_operator_svd) and the mode count (mode_bin_counts)
are the repository's functions, called as in modadd_controls.py: the Fig. 3 settings (no holdout,
no ridge) and the Fig. 2C settings (block-edge holdout, ridge 1).

Usage:  python modadd_operator_bases.py /path/to/cores controls_wd1.json [controls_wd0.json ...] out.json
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.mps.is_available = functools.lru_cache(lambda: False)  # CPU; see run_cpu.py

ROOT = Path(sys.argv[1]).resolve()
SAVED = [Path(p).resolve() for p in sys.argv[2:-1]]
OUT = Path(sys.argv[-1]).resolve()
sys.path.insert(0, str(ROOT / "src"))

from cores.extraction import ace  # noqa: E402
from cores.infer_mechanism import fit_operator_svd, token_core_reps  # noqa: E402
from cores.models import build_modadd_dataloader, collect_last_token_h2  # noqa: E402
from cores.modular_addition_analysis import load_model, mode_bin_counts  # noqa: E402

P, D = 53, 128
EPS = np.sin(np.pi / P)
WD1 = ROOT / "experiments/modular_addition/ckpts/wd1"
WD0 = ROOT / "experiments/modular_addition/ckpts/wd0"
device = torch.device("cpu")
test_loader = build_modadd_dataloader(train=False, batch_size=512)
y_test = torch.cat([y[:, -1] for _, y in test_loader])
gen = torch.Generator().manual_seed(53)


def fit(reps, fig2c):
    kw = dict(holdout_style="block_edges", ridge=1.0) if fig2c else dict(holdout_style="none", ridge=0.0)
    _, r2, r2_test, k, evals = fit_operator_svd(reps, k_max=52, svd_tol=1e-3, holdout_frac=0.075, holdout_seed=0, **kw)
    present, _ = mode_bin_counts(evals, P, EPS)
    return dict(modes=int(present.sum()), bins=[int(i) for i in np.flatnonzero(present)], k=int(k),
                r2=float(r2), r2_test=float(r2_test))


@torch.no_grad()
def analyse(row):
    arm, ep, m_ix, r = row["arm"], row["epoch"], row["model"], row["r_nec"]
    model = load_model(ep, m_ix, device, *((WD1,) if arm == "wd1" else (WD0, WD1)), vocab_size=P, d_model=D)
    H = collect_last_token_h2(model, test_loader, device, max_seqs=3000)
    Q_full, _ = ace(H.to(device=device, dtype=torch.float32), lambda: (lambda h: model.lm_head(h)),   # as in ace_ablate
                    sample_size=3000, min_rank=1, max_rank=D, return_full=True)
    Hd = H.double()
    W = model.lm_head.weight.detach().double()
    M = torch.stack([Hd[y_test == c].mean(0) for c in range(P)]) - Hd.mean(0, keepdim=True)
    top = lambda A, n: torch.linalg.svd(A, full_matrices=False)[2][:n].T.float().contiguous()
    r52 = min(r, 52)
    bases = {"ACE": Q_full[:, :r].contiguous(),
             "Readout SVD": top(W - W.mean(0, keepdim=True), r52),
             "Class means": top(M, r52),
             "Activation PCA": top(Hd - Hd.mean(0, keepdim=True), r),
             "Full state": torch.eye(D)}
    for i in range(5):
        bases[f"Random {i}"] = torch.linalg.qr(torch.randn(D, r, generator=gen))[0].contiguous()
    out = dict(arm=arm, epoch=ep, model=m_ix, r=int(r),
               logit_bins=[k + 1 for k, a in enumerate(row["amplitudes"]["alpha"]) if a >= 0.1 * max(row["amplitudes"]["alpha"])])
    for name, B in bases.items():
        reps = token_core_reps(model, B, test_loader, device, P=P)
        out[name] = dict(fig3=fit(reps, False), fig2c=fit(reps, True))
    assert out["ACE"]["fig3"]["modes"] == row["fit"]["modes"], (out["ACE"]["fig3"], row["fit"])   # same core as the saved run
    return out


rows = []
for path in SAVED:
    for row in json.loads(path.read_text())["rows"]:
        r = analyse(row)
        rows.append(r)
        rnd3 = [r[f"Random {i}"]["fig3"]["modes"] for i in range(5)]
        rnd2 = [r[f"Random {i}"]["fig2c"]["modes"] for i in range(5)]
        cell = lambda n: f"{r[n]['fig3']['modes']:2d}/{r[n]['fig2c']['modes']:2d}"
        same = all(r[n]["fig2c"]["bins"] == r["ACE"]["fig2c"]["bins"] for n in ("Readout SVD", "Class means", "Activation PCA", "Full state"))
        print(f"{r['arm']} ep={r['epoch']:5d} m{r['model']} r={r['r']:2d} | modes fig3/fig2c: ACE {cell('ACE')} readout {cell('Readout SVD')} "
              f"means {cell('Class means')} pca {cell('Activation PCA')} full {cell('Full state')} "
              f"random {np.mean(rnd3):4.1f}/{np.mean(rnd2):4.1f} | fig2c bins identical to ACE: {same} | "
              f"held-out R2 ACE {r['ACE']['fig2c']['r2_test']:.2f} readout {r['Readout SVD']['fig2c']['r2_test']:.2f} "
              f"means {r['Class means']['fig2c']['r2_test']:.2f} | logit freqs {len(r['logit_bins'])}", flush=True)
    OUT.write_text(json.dumps(dict(commit="527948b", rows=rows), indent=1))
print("wrote", OUT)
