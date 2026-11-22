"""Control ladder for the paper's two-site IOI cores (github.com/joshseth/cores @ 527948b), GPT-2 models.

The paper's core at each site (S2, END) is built by role_factors from two second moments over role-swap pairs:
the pair differences X = x+ - x- of the states, and the gradients G of the answer margin. Its "write" directions
are the balanced combination of the two. The paper's baselines are PCA of the unpaired states, a random subspace,
the mean pair difference (rank 1) and the unembedding difference (rank 1).

This script rebuilds the core at the paper's layers and ranks with the repository's functions (same pairs, seeds
and settings as golden.py's final cores) and adds, at the same ranks and from the same pairs:

  pair-difference PCA   top right singular vectors of X alone (uncentred, as the core uses it): the pairing
                        without the gradients
  gradient only         top right singular vectors of G alone: the gradients without the pairing

with the paper's own baselines (unpaired PCA, random, mean difference) alongside. Every subspace is scored with
the repository's evaluate() on its three prompt sets: the training frames, the held-out frames with new names, and
the prompts of Wang et al. A rank curve on the held-out set compares the three orderings at equal rank.

scikit-learn and nltk are not installed here; controls/shims holds stand-ins (an exact roc_auc_score, and a names
corpus reader). Without NLTK_NAMES_DIR the extraction name pool is the repository's own list only, far smaller
than the paper's; the held-out and Wang prompt sets do not depend on it.

Usage:  python ioi_ladder.py /path/to/cores MODEL TAG S2:END_layers S2:END_ranks DEVICE out.json [dtype attn_impl batch grad_batch]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401  (loaded before the sklearn stand-in is importable, so transformers does not try to use it)

ROOT = Path(sys.argv[1]).resolve()
MODEL, TAG = sys.argv[2], sys.argv[3]
OUT = Path(sys.argv[7]).resolve()
try:                                       # the real packages if installed (the pod), else the stand-ins (this laptop)
    import nltk  # noqa: F401
    import sklearn.metrics  # noqa: F401
    STANDINS = False
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "shims"))
    STANDINS = True
sys.path.insert(0, str(ROOT / "src"))

from cores.ioi.common import (COL, SEED, SITES, build_role_pairs, capture, ent, evaluate, load_model, orth,  # noqa: E402
                              role_factors, seed_all, setup_data, stratified_pairs)
from cores.ioi.controls import pca_dirs  # noqa: E402
from cores.ioi.wang import NAMES, frames  # noqa: E402

lay = dict(zip(SITES, (int(x) for x in sys.argv[4].split(":"))))
rank = dict(zip(SITES, (int(x) for x in sys.argv[5].split(":"))))
dev = torch.device(sys.argv[6])
DTYPE = sys.argv[8] if len(sys.argv) > 8 else "float32"                   # the paper's per-model settings (notebooks/ioi.ipynb)
ATTN = (sys.argv[9] if len(sys.argv) > 9 else "none")
BS, GRAD_BS = (int(sys.argv[10]), int(sys.argv[11])) if len(sys.argv) > 11 else (64, 32)
N_PAIRS = 12000
RANKS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
t0 = time.time()

seed_all(SEED)
tok, model = load_model(MODEL, DTYPE, None if ATTN == "none" else ATTN, dev, sink=1)
S = setup_data(tok)
train = stratified_pairs(tok, 3, S["seed"] + 301, S["fr_ext"], S["ext_names"])
heldout = stratified_pairs(tok, 19, S["seed"] + 201, S["fr_test"], S["test_names"])
wang = stratified_pairs(tok, 1, 601, frames(np.random.default_rng(601), 34), NAMES)
SETS = {"train": train, "held-out": heldout, "wang": wang}
print(f"{MODEL} | S2 L{lay['S2']} rank {rank['S2']} + END L{lay['END']} rank {rank['END']} | extraction names {len(S['ext_names'])} "
      f"| train {len(train[0]['y'])}, held-out {len(heldout[0]['y'])}, wang {len(wang[0]['y'])} prompts", flush=True)

# ---- the core's ingredients, as golden.py builds its final cores
pairs, _ = build_role_pairs(tok, N_PAIRS, np.random.default_rng(S["seed"] + 205), S["fr_ext"], ent(S["ext_names"]), p_noun=0.0, p_pronoun=0.0)
H, G = capture(model, pairs, sorted(set(lay.values())), N_PAIRS, GRAD_BS, dev)
top = lambda A: np.linalg.svd(np.linalg.qr(np.asarray(A, np.float64) / np.sqrt(len(A)), mode="r"))[2].T.astype(np.float32)   # right singular vectors, uncentred
order, mu, overlap = {}, {}, {}
rng = np.random.default_rng(S["seed"] + 701)
for t in SITES:
    h, g = H[lay[t]][t], G[lay[t]][t]
    fac = role_factors(h, g, dev)
    Hc, Gc = np.concatenate(h), np.concatenate(g)
    X = Hc[0::2] - Hc[1::2]
    d = Hc[0::2].mean(0) - Hc[1::2].mean(0)
    kmax = max(max(RANKS), rank[t])
    order[t] = {"ACE": fac["Tn"][:, :kmax],                                   # orth() of a prefix is the paper's core
                "pair-difference PCA": top(X)[:, :kmax],
                "gradient only": top(Gc)[:, :kmax],
                "PCA, unpaired (paper)": pca_dirs(Hc, kmax, dev),
                "random (paper)": rng.standard_normal((Hc.shape[1], kmax)).astype(np.float32),
                "mean difference, rank 1 (paper)": (d / np.linalg.norm(d))[:, None].astype(np.float32)}
    mu[t] = fac["mu"]
    Qa = orth(order[t]["ACE"][:, :rank[t]])
    overlap[t] = {b: float(np.linalg.norm(Qa.T @ orth(order[t][b][:, :rank[t]])) ** 2 / rank[t]) for b in ("pair-difference PCA", "gradient only", "PCA, unpaired (paper)")}
    del Hc, Gc, X
del H, G
print(f"cores from {N_PAIRS} pairs ({time.time() - t0:.0f}s) | share of the ACE core inside each same-rank subspace: {overlap}", flush=True)


def basis(t, b, r):
    A = order[t][b]
    return orth(A[:, :min(r, A.shape[1])])


def auc(data, specs):
    d, pool = data
    return evaluate(model, d, pool, [(lay[t], COL[t], basis(t, b, r), mu[t], mode) for (t, b, r, mode) in specs], BS, dev)["auc"]


R = {"commit": "527948b", "standins": STANDINS, "model": MODEL, "layers": [lay["S2"], lay["END"]], "ranks": [rank["S2"], rank["END"]], "extraction_names": len(S["ext_names"]),
     "overlap_with_ACE": overlap, "clean": {k: auc(v, []) for k, v in SETS.items()}, "rows": {}, "curve": {}}
print("clean " + ", ".join(f"{k} {v:.3f}" for k, v in R["clean"].items()), flush=True)
for b in order["S2"]:
    R["rows"][b] = {}
    for name, data in SETS.items():
        v = {"keep": auc(data, [(t, b, rank[t], "keep") for t in SITES]), "remove": auc(data, [(t, b, rank[t], "remove") for t in SITES]),
             "flip_S2": auc(data, [("S2", b, rank["S2"], "flip")]), "flip_END": auc(data, [("END", b, rank["END"], "flip")])}
        R["rows"][b][name] = v
        print(f"  {b:<32} {name:<8} keep {v['keep']:.3f}, remove {v['remove']:.3f}, flip S2 {v['flip_S2']:.3f}, END {v['flip_END']:.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(R, indent=1))

print("\nheld-out, same rank at both sites: keep / remove / flip at S2", flush=True)
for b in ("ACE", "pair-difference PCA", "gradient only"):
    R["curve"][b] = {}
    for r in RANKS:
        v = {"keep": auc(heldout, [(t, b, r, "keep") for t in SITES]), "remove": auc(heldout, [(t, b, r, "remove") for t in SITES]),
             "flip_S2": auc(heldout, [("S2", b, r, "flip")])}
        R["curve"][b][r] = v
    print(f"  {b:<20} " + "; ".join(f"{r}: {v['keep']:.2f} {v['remove']:.2f} {v['flip_S2']:.2f}" for r, v in R["curve"][b].items()), flush=True)
    OUT.write_text(json.dumps(R, indent=1))
print(f"wrote {OUT} ({time.time() - t0:.0f}s)", flush=True)
