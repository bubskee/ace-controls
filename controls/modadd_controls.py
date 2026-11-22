"""Modular-addition controls run through the paper's own code (github.com/joshseth/cores @ 527948b).

Checkpoints come from the repository's `modadd.py --only train` (paper settings: p = 53, three
two-layer transformers, weight decay 1 for 20,000 epochs, and a branch without weight decay from
epoch 900). Core extraction, ablation, class centroids and the shift-operator fit are the
repository's functions (ace_ablate, ablate, token_core_reps, fit_operator_svd, mode_bin_counts),
called with the arguments analyze_epoch uses.

Per checkpoint this adds:
  mode count controls   the same operator fit with the +1 shift replaced by random 53-cycles, with
                        Gaussian centroids of the same shape, and in the full 52-dimensional ACE span
  readout baseline      keep / remove for the row space of lm_head (centred, 52 dims), and the prefix
                        crossings (paper's definitions) in readout-SVD, activation-PCA and class-mean order
  between-class share   share of the variance of the core coordinates explained by the answer class
  amplitude profile     alpha_k = coefficient of cos(2 pi k (a + b - c) / p) in the logits over all p^2
                        inputs, against the theory's psi_k = 1 - cos(2 pi k / p) and against flat
  class-mean geometry   equal norms and equal angles of the class means (neural-collapse measures)

Usage:  python modadd_controls.py /path/to/cores out.json [epoch ...]
"""
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.mps.is_available = functools.lru_cache(lambda: False)  # CPU; see run_cpu.py

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
EPOCHS = [int(e) for e in sys.argv[3:]] or [0, 300, 500, 700, 800, 900, 1000, 1500, 2000, 5000, 10000, 15000, 20000]
sys.path.insert(0, str(ROOT / "src"))

from cores.ablations import ablate  # noqa: E402
from cores.extraction import _scan_nec, _scan_suf, ace_ablate  # noqa: E402
from cores.infer_mechanism import fit_operator_svd, token_core_reps  # noqa: E402
from cores.models import build_modadd_dataloader, collect_last_token_h2  # noqa: E402
from cores.modular_addition_analysis import load_model, mode_bin_counts, ring_metrics  # noqa: E402

P, D = 53, 128
EPS = np.sin(np.pi / P)
BRANCH = 900
WD1 = ROOT / "experiments/modular_addition/ckpts/wd1"
WD0 = ROOT / "experiments/modular_addition/ckpts/wd0"
# arm -> checkpoint directories, searched in order; the last two are the optimizer-restart controls
# (modadd_wd1_reset_branch.py, modadd_wd0_keepopt_branch.py), branches from epoch 900 like wd0
ARMS = {"wd1": (WD1,), "wd0": (WD0, WD1), "wd1_reset": (WD1.parent / "wd1_reset", WD1), "wd0_keepopt": (WD1.parent / "wd0_keepopt", WD1)}
device = torch.device("cpu")
test_loader = build_modadd_dataloader(train=False, batch_size=512)
y_test = torch.cat([y[:, -1] for _, y in test_loader]).numpy()
grid = torch.tensor([[a, b] for a in range(P) for b in range(P)], dtype=torch.long)
psi = 1 - np.cos(2 * np.pi * np.arange(1, P // 2 + 1) / P)
rng = np.random.default_rng(53)


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def fit(reps, *, paper_fig2c=False):
    """Operator fit as analyze_epoch calls it (Fig. 3), or as the clock panels call it (Fig. 2C)."""
    kw = dict(holdout_style="block_edges", ridge=1.0) if paper_fig2c else dict(holdout_style="none", ridge=0.0)
    _, r2, r2_test, k, evals = fit_operator_svd(reps, k_max=52, svd_tol=1e-3, holdout_frac=0.075, holdout_seed=0, **kw)
    present, _ = mode_bin_counts(evals, P, EPS)
    return dict(modes=int(present.sum()), k=int(k), r2=float(r2), r2_test=float(r2_test),
                frac_near=float(ring_metrics(evals, eps=EPS)[0]))


def crossings(model, Q, base):
    Q = Q.contiguous()
    suf = _scan_suf(model, Q, base, test_loader, device, start=1)
    nec = _scan_nec(model, Q, 2.0 / P, test_loader, device, start=1)
    full = Q.shape[1]
    keep_all = ablate(model, model.layer2, Q, "keep", test_loader, device, last_token=True)
    rem_all = ablate(model, model.layer2, Q, "remove", test_loader, device, last_token=True)
    # the scans return the last rank when the threshold is never crossed; flag that case
    return dict(suf=int(suf), nec=int(nec), dims=int(full), keep_all=float(keep_all), remove_all=float(rem_all),
                suf_reached=bool(keep_all >= base - 1e-3 or suf < full), nec_reached=bool(rem_all <= 2.0 / P or nec < full))


@torch.no_grad()
def analyse(arm, ep, m_ix):
    dirs = ARMS[arm]
    model = load_model(ep, m_ix, device, *dirs, vocab_size=P, d_model=D)
    H = collect_last_token_h2(model, test_loader, device, max_seqs=3000)
    Q_full, r_energy, r_nec_strict, r_suf_strict, base = ace_ablate(   # under no_grad, as in analyze_epoch
        model, H, device, test_loader, vocab_size=P, d_model=D, sample_size=3000)
    r_nec, r_suf = max(r_energy, r_nec_strict), max(r_energy, r_suf_strict)
    U = Q_full[:, :r_nec].contiguous()
    out = dict(arm=arm, epoch=ep, model=m_ix, acc=float(base), r_energy=int(r_energy), r_nec=int(r_nec), r_suf=int(r_suf),
               r_nec_strict=int(r_nec_strict), r_suf_strict=int(r_suf_strict),
               keep=float(ablate(model, model.layer2, U, "keep", test_loader, device, last_token=True)),
               remove=float(ablate(model, model.layer2, U, "remove", test_loader, device, last_token=True)))

    # --- operator fit, as in the paper, and its controls -------------------------------------------
    reps = token_core_reps(model, U, test_loader, device, P=P)
    out["fit"] = fit(reps)
    out["fit_fig2c"] = fit(reps, paper_fig2c=True)
    perm = [fit(reps[rng.permutation(P)]) for _ in range(20)]
    perm2c = [fit(reps[rng.permutation(P)], paper_fig2c=True) for _ in range(20)]
    out["random_cycle"] = dict(modes_mean=float(np.mean([f["modes"] for f in perm])),
                               modes_min=int(min(f["modes"] for f in perm)), modes_max=int(max(f["modes"] for f in perm)),
                               r2=float(np.mean([f["r2"] for f in perm])),
                               r2_test_fig2c=float(np.nanmean([f["r2_test"] for f in perm2c])))
    gauss = [fit(rng.standard_normal(reps.shape)) for _ in range(20)]
    out["gaussian_centroids"] = dict(modes_mean=float(np.mean([f["modes"] for f in gauss])),
                                     r2=float(np.mean([f["r2"] for f in gauss])))
    reps52 = token_core_reps(model, Q_full[:, :52].contiguous(), test_loader, device, P=P)
    out["fit_ace52"] = fit(reps52)
    out["fit_ace52_fig2c"] = fit(reps52, paper_fig2c=True)
    p52 = [fit(reps52[rng.permutation(P)]) for _ in range(20)]
    out["random_cycle_ace52"] = dict(modes_mean=float(np.mean([f["modes"] for f in p52])),
                                     r2=float(np.mean([f["r2"] for f in p52])))
    sv = np.linalg.svd(reps - reps.mean(0), compute_uv=False)
    out["centroid_sv"] = dict(top=float(sv[0]), n_above_1e3=int((sv >= 1e-3 * sv[0]).sum()), n=int(len(sv)))

    # --- readout baseline and alternative orderings -------------------------------------------------
    W = model.lm_head.weight.detach().double()
    Vr = torch.linalg.svd(W - W.mean(0, keepdim=True), full_matrices=False)[2][:52].T.float()
    Hd = H.double()
    Hc = Hd - Hd.mean(0, keepdim=True)
    Vp = torch.linalg.svd(Hc, full_matrices=False)[2][:53].T.float()
    yt = torch.from_numpy(y_test)
    M = torch.stack([Hd[yt == c].mean(0) for c in range(P)]) - Hd.mean(0, keepdim=True)
    Vm = torch.linalg.svd(M, full_matrices=False)[2][:52].T.float()
    out["orderings"] = {"ACE": crossings(model, Q_full[:, :53], base), "Readout SVD": crossings(model, Vr, base),
                        "Activation PCA": crossings(model, Vp, base), "Class means": crossings(model, Vm, base)}

    # --- between-class share ------------------------------------------------------------------------
    def share(Z):
        Z = Z - Z.mean(0, keepdim=True)
        between = sum((yt == c).sum() * (Z[yt == c].mean(0) ** 2).sum() for c in range(P))
        return float(between / (Z ** 2).sum())
    out["between_class_share"] = dict(core=share(Hd @ U.double()), full_state=share(Hd),
                                      outside_core=share(Hd - (Hd @ U.double()) @ U.double().T))

    # --- amplitude profile of the logits over all p^2 inputs ---------------------------------------
    L = model(grid)[:, -1].double().numpy().reshape(P, P, P)
    L = L - L.mean(2, keepdims=True)
    F = np.fft.fftn(L)
    K = P // 2
    alpha = np.array([2 * F[k, k, (-k) % P].real / P ** 3 for k in range(1, K + 1)])
    power = sum(2 * abs(F[k, k, (-k) % P]) ** 2 for k in range(1, K + 1)) / np.sum(np.abs(F) ** 2)
    out["amplitudes"] = dict(alpha=alpha.tolist(), clock_share_of_logit_variance=float(power),
                             cos_with_psi=cos(alpha, psi), cos_with_flat=cos(alpha, np.ones(K)),
                             min_over_max=float(alpha.min() / alpha.max()), alpha1_over_alphaK=float(alpha[0] / alpha[-1]),
                             predicted_alpha1_over_alphaK=float(psi[0] / psi[-1]),
                             n_above_10pct_of_max=int((alpha >= 0.1 * alpha.max()).sum()))

    # --- class-mean geometry (neural-collapse measures) --------------------------------------------
    Mn = M.numpy()
    norms = np.linalg.norm(Mn, axis=1)
    C = (Mn / norms[:, None]) @ (Mn / norms[:, None]).T
    off = C[~np.eye(P, dtype=bool)]
    out["class_means"] = dict(norm_cv=float(norms.std() / norms.mean()), cos_mean=float(off.mean()), cos_sd=float(off.std()),
                              simplex_cos=-1.0 / (P - 1))
    return out


rows = []
for arm in os.environ.get("MODADD_ARMS", "wd1,wd0").split(","):   # MODADD_ARMS=wd1 analyses one arm
    for ep in EPOCHS:
        if arm != "wd1" and ep <= BRANCH:
            continue
        for m_ix in range(3):
            r = analyse(arm, ep, m_ix)
            rows.append(r)
            a, o = r["amplitudes"], r["orderings"]
            print(f"{arm} ep={ep:5d} m{m_ix} acc={r['acc']:.3f} r_en={r['r_energy']:2d} r_suf={r['r_suf']:3d} r_nec={r['r_nec']:3d} "
                  f"keep={r['keep']:.3f} rem={r['remove']:.3f} | modes={r['fit']['modes']:2d} k={r['fit']['k']:2d} "
                  f"rand-cycle={r['random_cycle']['modes_mean']:4.1f} gauss={r['gaussian_centroids']['modes_mean']:4.1f} "
                  f"ace52={r['fit_ace52']['modes']:2d} rand52={r['random_cycle_ace52']['modes_mean']:4.1f} | "
                  f"readout keep={o['Readout SVD']['keep_all']:.3f} rem={o['Readout SVD']['remove_all']:.3f} | "
                  f"share core={r['between_class_share']['core']:.3f} | cos psi={a['cos_with_psi']:.2f} flat={a['cos_with_flat']:.2f}",
                  flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(commit="527948b", epochs=EPOCHS, rows=rows), indent=1))
print("wrote", OUT)
