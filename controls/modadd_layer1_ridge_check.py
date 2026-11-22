"""Is the layer-1 operator result in modadd_layer1.py a scale effect of the paper's ridge penalty?

fit_operator_svd adds ridge * I to X^T X without normalising X, so its effect depends on the size of the class
centroids. This prints, for layer-1 and layer-2 centroids of the same checkpoints, the centroid singular values and
the held-out fit with the paper's ridge (1.0), with no ridge, and with the centroids rescaled to unit top singular
value times 100 (ridge then negligible), each in the full state.

Usage:  python modadd_layer1_ridge_check.py /path/to/cores
"""
import functools, sys
from pathlib import Path
import numpy as np, torch
torch.backends.mps.is_available = functools.lru_cache(lambda: False)
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT / "src"))
from cores.infer_mechanism import fit_operator_svd
from cores.models import build_modadd_dataloader
from cores.modular_addition_analysis import load_model, mode_bin_counts
P, D = 53, 128; EPS = np.sin(np.pi / P)
WD1, WD0 = ROOT / "experiments/modular_addition/ckpts/wd1", ROOT / "experiments/modular_addition/ckpts/wd0"
loader = build_modadd_dataloader(train=False, batch_size=512)
X = torch.cat([x for x, _ in loader]); Y = torch.cat([y[:, -1] for _, y in loader]).numpy()

def fit(reps, ridge):
    _, r2, r2t, k, ev = fit_operator_svd(reps, k_max=52, svd_tol=1e-3, holdout_frac=0.075, holdout_seed=0, holdout_style="block_edges", ridge=ridge)
    return f"{int(mode_bin_counts(ev, P, EPS)[0].sum()):2d} modes (k={k}, held-out R2 {r2t:5.2f})"

for arm, ep in [("wd1", 800), ("wd1", 2000), ("wd1", 20000), ("wd0", 20000)]:
    for m in range(3):
        model = load_model(ep, m, torch.device("cpu"), *((WD1,) if arm == "wd1" else (WD0, WD1)), vocab_size=P, d_model=D); model.eval()
        with torch.no_grad(): _, h1, h2 = model(X, return_states=True)
        for name, H in (("layer 1, all positions", h1.reshape(len(X), -1)), ("layer 1, last position", h1[:, -1]), ("layer 2, last position", h2[:, -1])):
            H = H.double().numpy(); reps = np.stack([H[Y == c].mean(0) for c in range(P)])
            s = np.linalg.svd(reps - reps.mean(0), compute_uv=False)
            between = ((reps - H.mean(0)) ** 2).sum() * (len(H) / P) / ((H - H.mean(0)) ** 2).sum()
            print(f"{arm} ep={ep:5d} m{m} {name:23s} top sv {s[0]:8.3f} sv[10] {s[10]:7.3f} between-class share {between:.3f} | ridge 1: {fit(reps, 1.0)} | ridge 0: {fit(reps, 0.0)} | rescaled, ridge 1: {fit(reps * 100 / s[0], 1.0)}", flush=True)
