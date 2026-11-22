"""Markov-chain controls run through the paper's own code (github.com/joshseth/cores @ 527948b).

Everything model-related is imported from the repository: OneLayerTransformer, train_markov_lm,
ace, ablate, fit_core_dynamics, and the 12-seed ladder's own helpers (readout_basis, pca_basis,
matched_spectrum_error, cca_coordinates, project, load_or_train). Seeds, data, epochs, learning
rate, rank and Jacobian sample count are the ladder's.

Arms added here, all rank 3:
  trained models      Token-3: span of the centred mean activation per current token
  untrained models    ACE-3 (the ladder's own untrained models, seed + 9000), PCA-3, Token-3:
                      operator spectrum, which the ladder computes only for trained models
  frozen-body probe   the same untrained transformer with only lm_head trained by the paper's
                      training loop: accuracy, ACE-3 keep / remove, spectrum, CCA to trained ACE-3
  random embedding    no transformer: h_t = E[x_t] with E a random 4 x 64 matrix; spectrum
For every arm: share of the subspace's variance explained by the current token alone.

Usage:  python markov_controls.py /path/to/cores  out.json
"""
import functools
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# torch 2.14 MPS cannot run dropout inside attention in training mode; use CPU throughout.
torch.backends.mps.is_available = functools.lru_cache(lambda: False)  # torch._dynamo expects __wrapped__

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "supplement"))

from cores.ablations import ablate  # noqa: E402
from cores.extraction import ace  # noqa: E402
from cores.infer_mechanism import fit_core_dynamics  # noqa: E402
from cores.models import (MarkovDataset, OneLayerTransformer, collect_markov_activations,  # noqa: E402
                          eval_accuracy, train_markov_lm)
from cores.utils import get_device, seed_all  # noqa: E402
import markov_null_ladder_12seed as L  # noqa: E402
from markov_model_cache import load_or_train  # noqa: E402

SEEDS, D, RANK, T = L.SEEDS, L.D_MODEL, L.RANK, L.T
device = get_device()
print("device", device, flush=True)

train = MarkovDataset(T=T, seed=0, num_samples=3000, seq_len=32)
test = MarkovDataset(T=T, seed=999, num_samples=3000, seq_len=32)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False)
truth = np.linalg.eigvals(T)
truth = truth[np.argsort(-np.abs(truth))][1:]
tokens = test.x.reshape(-1).numpy()


def ace3(model, hidden):
    q, w = ace(hidden.to(device, torch.float32), lambda m=model: (lambda h: m.lm_head(h)),
               sample_size=128, energy_threshold=0.999, max_rank=D, return_full=True)
    return q[:, :RANK].detach().cpu()


def token_basis(hidden):
    h = hidden.to(torch.float64)
    means = torch.stack([h[torch.from_numpy(tokens == k)].mean(0) for k in range(4)]) - h.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(means, full_matrices=False)
    return vh[:RANK].T.to(torch.float32)


def token_share(hidden, basis):
    """Share of the variance of the subspace coordinates explained by the current token."""
    z = L.project(hidden, basis)
    between = sum((tokens == k).sum() * (z[tokens == k].mean(0) ** 2).sum() for k in range(4))
    return float(between / (z ** 2).sum())


def operator(seq, basis):
    _, ev, r2 = fit_core_dynamics(seq, basis.cpu())
    ev = np.asarray(ev)
    ev = ev[np.argsort(-np.abs(ev))]
    return dict(r2=float(r2), spectral_distance=float(L.matched_spectrum_error(ev, truth)),
                evals=[[float(z.real), float(z.imag)] for z in ev])


def causal(model, basis):
    b = basis.to(device)
    return dict(keep=float(ablate(model, model.layer, b, "keep", test_loader, device, last_token=False)),
                remove=float(ablate(model, model.layer, b, "remove", test_loader, device, last_token=False)))


def collect(model):
    model.eval()
    _, seq, _ = collect_markov_activations(model, test_loader, device)
    return seq, seq.reshape(-1, D).detach().cpu()


def pair_cca(coords_a, coords_b=None):
    if coords_b is None:
        vals = [L.cca_coordinates(coords_a[i], coords_a[j]) for i, j in combinations(range(len(coords_a)), 2)]
    else:
        vals = [L.cca_coordinates(a, b) for a in coords_a for b in coords_b]
    return np.asarray(vals).mean(0).tolist()


rows, coords = [], {}


def record(group, name, seed, model, seq, hidden, basis, with_causal):
    r = dict(group=group, subspace=name, seed=seed, token_share=token_share(hidden, basis), **operator(seq, basis))
    if with_causal:
        r.update(base=float(eval_accuracy(model, test_loader, device)), **causal(model, basis))
    rows.append(r)
    coords.setdefault(f"{group} {name}", []).append(L.project(hidden, basis))


onehot = np.eye(4)[tokens][:, :3]
onehot = onehot - onehot.mean(0)
rng = np.random.default_rng(404)

for i, seed in enumerate(SEEDS):
    # trained: the ladder's arms, plus Token-3
    model, cached = load_or_train(seed, train_loader, device, d_model=D, epochs=40, lr=1e-4)
    seq, hidden = collect(model)
    arms = {"ACE-3": ace3(model, hidden), "PCA-3": L.pca_basis(hidden), "Readout-3": L.readout_basis(model),
            "Token-3": token_basis(hidden), "Random-3": L.random_basis(rng)}
    for name, basis in arms.items():
        record("trained", name, seed, model, seq, hidden, basis, True)
    with torch.no_grad():
        logits = torch.cat([model(x.to(device)).cpu() for x, _ in test_loader]).reshape(-1, 4).numpy().astype(np.float64)
    logits = logits - logits.mean(0)
    coords.setdefault("trained logits", []).append(logits @ np.linalg.svd(logits, full_matrices=False)[2][:3].T)

    # untrained: the ladder's own construction
    seed_all(seed + 9000)
    um = OneLayerTransformer(4, D).to(device)
    um.eval()
    useq, uhidden = collect(um)
    for name, basis in {"ACE-3": ace3(um, uhidden), "PCA-3": L.pca_basis(uhidden),
                        "Token-3": token_basis(uhidden)}.items():
        record("untrained", name, seed, um, useq, uhidden, basis, False)

    # frozen-body probe: same initialisation, only the readout is trained, by the paper's loop
    for lr in (1e-4, 1e-3):
        seed_all(seed + 9000)
        pm = OneLayerTransformer(4, D).to(device)
        for n, p in pm.named_parameters():
            p.requires_grad_(n.startswith("lm_head"))
        for _ in train_markov_lm(pm, train_loader, device, epochs=40, lr=lr):
            pass
        pseq, phidden = collect(pm)
        for name, basis in {"ACE-3": ace3(pm, phidden), "Readout-3": L.readout_basis(pm)}.items():
            record(f"probe lr={lr:g}", name, seed, pm, pseq, phidden, basis, True)

    # random embedding, no transformer
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(4, D, generator=g)
    eseq = E[test.x]
    ehidden = eseq.reshape(-1, D)
    record("random embedding", "PCA-3", seed, None, eseq, ehidden, L.pca_basis(ehidden), False)
    print(f"seed {seed} done ({'cached' if cached else 'trained'})", flush=True)

summary = {}
for key in sorted({(r["group"], r["subspace"]) for r in rows}):
    sel = [r for r in rows if (r["group"], r["subspace"]) == key]
    s = {m: float(np.mean([r[m] for r in sel])) for m in ("base", "keep", "remove", "r2", "spectral_distance", "token_share")
         if m in sel[0]}
    s["spectral_distance_max"] = float(np.max([r["spectral_distance"] for r in sel]))
    summary[" / ".join(key)] = s

cca = {name: pair_cca(c) for name, c in coords.items()}
cca["one-hot current token (identical across seeds)"] = [1.0, 1.0, 1.0]
cca_to_trained = {name: pair_cca(c, coords["trained ACE-3"]) for name, c in coords.items() if name != "trained ACE-3"}
cca_to_token = {name: np.mean([L.cca_coordinates(z, onehot) for z in c], 0).tolist() for name, c in coords.items()}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dict(commit="527948b", seeds=SEEDS, summary=summary, cross_seed_cca=cca,
                               cca_to_trained_ACE3=cca_to_trained, cca_to_current_token=cca_to_token, rows=rows), indent=1))
print("\narm                                   base   keep  remove   r2    spec_err (max)   token_share")
for k, s in summary.items():
    print(f"{k:36s} {s.get('base', float('nan')):6.3f} {s.get('keep', float('nan')):6.3f} {s.get('remove', float('nan')):6.3f} "
          f"{s['r2']:6.3f}  {s['spectral_distance']:.4f} ({s['spectral_distance_max']:.4f})   {s['token_share']:.4f}")
print("\ncross-seed CCA (mean over pairs)")
for k, v in cca.items():
    print(f"  {k:48s}", np.round(v, 3))
print("\nCCA to trained ACE-3 of other and same seeds (mean)")
for k, v in cca_to_trained.items():
    print(f"  {k:48s}", np.round(v, 3))
print("\nCCA of each arm's coordinates with the one-hot current token")
for k, v in cca_to_token.items():
    print(f"  {k:48s}", np.round(v, 3))
