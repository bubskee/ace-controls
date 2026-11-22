"""Fit the paper's delay law and alternatives to the share sweep written by grok_share.py.

The law and its fit (curve_fit call, starting point, bounds) are the repository's own
(grok_sweep_two_layer.exact_ode_p and run_p_sweep). Usage: python grok_share_fit.py /path/to/cores out.jsonl
"""
import collections
import json
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, sys.argv[1])
import grok_sweep_two_layer as gs  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[2])]
r2 = lambda y, f: float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2))


def table(thr):
    cell = collections.defaultdict(list)
    for r in rows:
        c, m = r["cross"][thr], r["mem"]
        cell[(r["frac"], r["p"])].append(c - m if c is not None and m is not None and c > m else None)
    return cell


def law_fit(ps, tau):
    (A, pc), _ = curve_fit(gs.exact_ode_p, ps, tau, p0=[10000, 10], bounds=(0, [np.inf, np.min(ps) - 0.1]), maxfev=20000)
    return A, pc, r2(tau, gs.exact_ode_p(ps, A, pc))


for thr in ("0.99", "0.9", "0.5"):
    cell = table(thr)
    fracs = sorted({f for f, _ in cell}); ps_all = sorted({p for _, p in cell})
    print(f"\n=== delay = first step with held-out accuracy >= {thr}, minus first step with train accuracy >= 0.99 (mean of 3 seeds; '-' = a seed never reached it)")
    print("share   " + "".join(f"p={p:<7d}" for p in ps_all) + "| law: Omega  p_crit  R2   | power law: exponent  R2(lin)")
    for f in fracs:
        means = {p: (np.mean(cell[(f, p)]) if all(v is not None for v in cell[(f, p)]) else None) for p in ps_all}
        ok = [p for p in ps_all if means[p] is not None]
        line = f"{f:.2f}    " + "".join(f"{means[p]:<9.0f}" if means[p] is not None else f"{'-':<9s}" for p in ps_all)
        if len(ok) >= 3:
            ps, tau = np.array(ok, float), np.array([means[p] for p in ok])
            A, pc, rl = law_fit(ps, tau)
            b, lc = np.polyfit(np.log(ps), np.log(tau), 1)
            line += f"| {A:10.0f}  {pc:5.1f}  {rl:.3f} | {b:8.2f}  {r2(tau, np.exp(lc) * ps ** b):.3f}"
        print(line)
    # pooled: one (Omega, p_crit) for all shares, as the law states, against log-linear in p and share
    P_, F_, T_ = [], [], []
    for (f, p), v in cell.items():
        if all(x is not None for x in v):
            P_.append(p); F_.append(f); T_.append(np.mean(v))
    P_, F_, T_ = np.array(P_, float), np.array(F_), np.array(T_)
    A, pc, rl = law_fit(P_, T_)
    X = np.c_[np.ones(len(P_)), np.log(P_), np.log(F_)]
    beta, *_ = np.linalg.lstsq(X, np.log(T_), rcond=None)
    bp, *_ = np.linalg.lstsq(X[:, :2], np.log(T_), rcond=None)
    print(f"pooled over shares ({len(T_)} cells): the law with one Omega and p_crit: R2 = {rl:.3f} (linear), "
          f"{r2(np.log(T_), np.log(gs.exact_ode_p(P_, A, pc))):.3f} (log)")
    print(f"   log delay = {beta[0]:.2f} {beta[1]:+.2f} log p {beta[2]:+.2f} log share: R2 = {r2(np.log(T_), X @ beta):.3f} (log);"
          f"  p alone: R2 = {r2(np.log(T_), X[:, :2] @ bp):.3f}")
    if thr == "0.99":
        for p in ps_all:
            v = [np.mean(cell[(f, p)]) for f in fracs if all(x is not None for x in cell[(f, p)])]
            if len(v) == len(fracs):
                print(f"   p={p}: delay ratio share {fracs[0]} / share {fracs[-1]} = {v[0] / v[-1]:.1f}")
print("\nnever reached 0.99 within the cap:", sorted({(r['p'], r['frac'], r['seed']) for r in rows if r['cross']['0.99'] is None}))
print("stopped by wall-clock guard:", sorted({(r['p'], r['frac'], r['seed']) for r in rows if r['stopped_by_time_budget']}))
