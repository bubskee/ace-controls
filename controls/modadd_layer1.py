"""Modular addition away from the readout: the paper's analysis one layer earlier.

The paper extracts its modular-addition core from the output of the last transformer layer, which feeds a linear
readout. There the readout's row space is sufficient and necessary by algebra. This script repeats the analysis on
the output of layer 1, where the map to the logits is nonlinear (layer 2, then the readout), so no such identity
applies.

State: the full layer-1 output, all positions, flattened (T x 128 dims). Tail: layer 2 and the readout, logits at
the last token. ACE is the repository's _balanced_svd applied to the covariance of the state and the mean J^T J of
the tail, built as the repository's ace() builds them (the Jacobians are taken by backpropagation, one logit at a
time; the routine is checked against the linear readout, where mean J^T J must equal W^T W).

Orderings compared, all in the same flattened space:
  ACE            balanced directions
  Gradient only  top eigenvectors of mean J^T J
  PCA            principal directions of the state
  Class means    principal directions of the state's means by answer class (at most 52)
  Random         random orthonormal directions
For each: the smallest prefix that is sufficient (accuracy within 1e-3 of baseline when only the prefix is kept) and
necessary (accuracy at or below 2/53 when it is removed), the paper's two criteria; and the paper's shift-operator
fit on the class centroids in the prefix of ACE's necessary rank (held-out settings of Fig. 2C).

Usage:  python modadd_layer1.py /path/to/cores out.json
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.mps.is_available = functools.lru_cache(lambda: False)  # CPU; see run_cpu.py

ROOT, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / "src"))

from cores.extraction import _balanced_svd  # noqa: E402
from cores.infer_mechanism import fit_operator_svd  # noqa: E402
from cores.models import build_modadd_dataloader  # noqa: E402
from cores.modular_addition_analysis import load_model, mode_bin_counts  # noqa: E402

P, D = 53, 128
EPS = np.sin(np.pi / P)
WD1 = ROOT / "experiments/modular_addition/ckpts/wd1"
WD0 = ROOT / "experiments/modular_addition/ckpts/wd0"
CHECKPOINTS = [("wd1", 0), ("wd1", 800), ("wd1", 2000), ("wd1", 20000), ("wd0", 20000)]
device = torch.device("cpu")
loader = build_modadd_dataloader(train=False, batch_size=512)
X = torch.cat([x for x, _ in loader])
Y = torch.cat([y[:, -1] for _, y in loader])
N, T = X.shape
MAXR = min(T * D, 200)
rng = torch.Generator().manual_seed(53)
causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)


def mean_jtj(tail, h):
    """Mean over inputs of J^T J, J the Jacobian of the logits with respect to each input's own state."""
    h = h.detach().clone().requires_grad_(True)
    out = tail(h)
    obs = torch.zeros(h[0].numel(), h[0].numel(), dtype=torch.float64)
    for v in range(out.shape[1]):
        g, = torch.autograd.grad(out[:, v].sum(), h, retain_graph=True)
        g = g.reshape(len(h), -1).double()
        obs += g.T @ g
    return obs / len(h)


def top(A, n):
    return torch.linalg.svd(A, full_matrices=False)[2][:n].T.contiguous()


class FlatHook:
    """Keep or remove a subspace of the flattened layer-1 output (no centring, as in the repository's hook)."""

    def __init__(self, U, mode):
        self.U, self.mode = U, mode

    def __call__(self, module, inp, out):
        f = out.reshape(out.shape[0], -1)
        proj = (f @ self.U) @ self.U.T
        return (proj if self.mode == "keep" else f - proj).reshape(out.shape)


@torch.no_grad()
def accuracy(model, U=None, mode=None):
    handle = model.layer1.register_forward_hook(FlatHook(U, mode)) if U is not None else None
    try:
        return float((model(X)[:, -1].argmax(-1) == Y).float().mean())
    finally:
        if handle:
            handle.remove()


def crossings(model, Q, base):
    keep = [accuracy(model, Q[:, :r].contiguous(), "keep") for r in range(1, Q.shape[1] + 1)]
    rem = [accuracy(model, Q[:, :r].contiguous(), "remove") for r in range(1, Q.shape[1] + 1)]
    suf = next((r for r, a in enumerate(keep, 1) if a >= base - 1e-3), None)
    nec = next((r for r, a in enumerate(rem, 1) if a <= 2.0 / P), None)
    return dict(suf=suf, nec=nec, dims=int(Q.shape[1]), keep_at_16=keep[min(15, len(keep) - 1)], remove_at_16=rem[min(15, len(rem) - 1)],
                keep_all=keep[-1], remove_all=rem[-1])


def centroids(Z):
    return np.stack([Z[Y.numpy() == c].mean(0) for c in range(P)])


def top_sv(reps):
    return float(np.linalg.svd(reps - reps.mean(0), compute_uv=False)[0])


def operator(Z, scale):
    """The paper's held-out fit. Its ridge term (1.0 * I added to X^T X) is not normalised, so it depends on the
    size of the centroids; layer-1 centroids are 30-60 times smaller than layer-2 ones under weight decay and the
    raw fit is then dominated by the penalty. `scale` multiplies the centroids so that the full layer-1 state's
    centroids have the same top singular value as the layer-2 centroids of the same checkpoint; one factor per
    checkpoint, shared by every basis. Both fits are reported."""
    out = {}
    for name, c in (("raw", 1.0), ("scaled", scale)):
        _, r2, r2_test, k, evals = fit_operator_svd(centroids(Z) * c, k_max=52, svd_tol=1e-3, holdout_frac=0.075, holdout_seed=0, holdout_style="block_edges", ridge=1.0)
        present, _ = mode_bin_counts(evals, P, EPS)
        out[name] = dict(modes=int(present.sum()), bins=[int(i) for i in np.flatnonzero(present)], k=int(k), r2_test=float(r2_test))
    return out


def analyse(arm, ep, m_ix):
    model = load_model(ep, m_ix, device, *((WD1,) if arm == "wd1" else (WD0, WD1)), vocab_size=P, d_model=D)
    model.eval()
    with torch.no_grad():
        _, h1, h2 = model(X, return_states=True)
    base = accuracy(model)

    # check of the Jacobian routine where the answer is known: the linear readout
    W = model.lm_head.weight.detach().double()
    err = (mean_jtj(lambda h: model.lm_head(h), h2[:, -1]) - W.T @ W).abs().max() / (W.T @ W).abs().max()
    assert err < 1e-5, err

    H = h1.reshape(N, -1).double()
    Hc = H - H.mean(0, keepdim=True)
    cov = (Hc.T @ Hc) / (N - 1)
    obs = mean_jtj(lambda h: model.lm_head(model.layer2(h, src_mask=causal))[:, -1], h1)
    Q_ace, _ = _balanced_svd(cov, obs, max_rank=T * D)
    M = torch.stack([H[Y == c].mean(0) for c in range(P)]) - H.mean(0, keepdim=True)
    orders = {"ACE": Q_ace[:, :MAXR].float().contiguous(),
              "Gradient only": torch.linalg.eigh(obs)[1].flip(1)[:, :MAXR].float().contiguous(),
              "PCA": top(Hc, MAXR).float(),
              "Class means": top(M, 52).float(),
              "Random": torch.linalg.qr(torch.randn(T * D, MAXR, generator=rng))[0].contiguous()}
    out = dict(arm=arm, epoch=ep, model=m_ix, acc=base, state_dims=T * D, orderings={k: crossings(model, Q, base) for k, Q in orders.items()})
    r = out["orderings"]["ACE"]["nec"] or out["orderings"]["ACE"]["suf"] or 16
    out["operator_rank"] = r
    scale = top_sv(centroids(h2[:, -1].double().numpy())) / top_sv(centroids(H.numpy()))
    out["centroid_scale"] = scale
    out["operator"] = {k: operator((H.float() @ Q[:, :min(r, Q.shape[1])]).numpy(), scale) for k, Q in orders.items()}
    out["operator"]["Full state"] = operator(H.numpy(), scale)
    out["operator"]["Layer 2 (paper's site), full state"] = operator(h2[:, -1].double().numpy(), 1.0)
    return out


rows = []
for arm, ep in CHECKPOINTS:
    for m_ix in range(3):
        r = analyse(arm, ep, m_ix)
        rows.append(r)
        o, f = r["orderings"], r["operator"]
        print(f"{arm} ep={ep:5d} m{m_ix} acc={r['acc']:.3f} | suf/nec " + " ".join(f"{k} {v['suf']}/{v['nec']}" for k, v in o.items())
              + f" | held-out modes at rank {r['operator_rank']}, centroids x{r['centroid_scale']:.0f}: "
              + " ".join(f"{k} {v['scaled']['modes']} ({v['scaled']['r2_test']:.2f})" for k, v in f.items()), flush=True)
        OUT.write_text(json.dumps(dict(commit="527948b", rows=rows), indent=1))
print("wrote", OUT)
