"""Grokking delay against training share, with the paper's own training function.

Calls grok_sweep_two_layer.train_run (github.com/joshseth/cores @ 527948b) with the modulus sweep's
settings (two-layer transformer, d_model 128, d_ff 512, lr 1e-3, weight decay 1, full batch, evaluation
every step, thresholds 0.99, cap 30,000 steps, seeds 42-44) and changes one argument the paper holds at
0.5: `frac`, the share of the p^2 pairs used for training.

The paper's law, tau = -Omega log(1 - p_crit / p), has an optimizer constant Omega and an "architectural"
constant p_crit; neither depends on the training share. The share = 0.5 column repeats six modulus settings with three initialization seeds.

The repository's per-step evaluation is wrapped to record the accuracy trajectory, so the delay can also
be read at other accuracy thresholds; training itself is untouched.

Usage:  python grok_share.py /path/to/cores out.jsonl [workers]
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

PS = [31, 43, 53, 61, 79, 101]
FRACS = [0.4, 0.5, 0.6, 0.75]
SEEDS = [42, 43, 44]
BUDGET_S = 45 * 60          # wall-clock guard per run; a run stopped by it is recorded as such


class OutOfTime(Exception):
    pass


def one(task):
    root, p, frac, seed = task
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, root)
    import grok_sweep_two_layer as gs

    traj, t0, inner = [], time.time(), gs.eval_acc_and_margin_stats

    def recording(model, X, Y):
        out = inner(model, X, Y)
        traj.append(out[0])
        if len(traj) % 200 == 0 and time.time() - t0 > BUDGET_S:
            raise OutOfTime
        return out

    gs.eval_acc_and_margin_stats = recording
    n_train = int(p * p * frac)
    stopped = False
    try:
        out = gs.train_run(p=p, device=torch.device("cpu"), d_model=128, n_head=4, d_ff=512, lr=1e-3, omega=1.0,
                           frac=frac, batch_size=n_train + 1, max_epochs=30000, eval_every_steps=1,
                           grok_thresh=0.99, mem_thresh=0.99, seed=seed, compile_mode="off")
    except OutOfTime:
        out, stopped = dict(p=p, omega=1.0, n_train=n_train), True
    finally:
        gs.eval_acc_and_margin_stats = inner
    tr, te = traj[0::2], traj[1::2][:len(traj[0::2])]

    def first(xs, thr):
        for step, v in enumerate(xs, start=1):
            if v >= thr:
                return step
        return None

    mem = first(tr, 0.99)
    cross = {str(t): first(te, t) for t in (0.5, 0.9, 0.95, 0.99)}
    return dict(p=p, frac=frac, seed=seed, n_train=out.get("n_train"), mem=mem, cross=cross,
                paper_mem=out.get("tau_mem_step"), paper_grok=out.get("tau_grok_step"), paper_drift=out.get("tau_drift_step"),
                steps_run=len(te), final_test=te[-1] if te else None, max_test=max(te) if te else None,
                stopped_by_time_budget=stopped, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    root, out = str(Path(sys.argv[1]).resolve()), Path(sys.argv[2]).resolve()
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["p"], r["frac"], r["seed"]))
    tasks = [(root, p, f, s) for p in PS for f in FRACS for s in SEEDS if (p, f, s) not in done]
    tasks.sort(key=lambda t: t[1] * t[2] * t[2] * (3 if t[2] <= 0.45 else 1))   # cheap runs first
    tasks.sort(key=lambda t: t[1] * t[1])
    print(len(tasks), "runs", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool, out.open("a") as fh:
        for r in pool.imap_unordered(one, tasks):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"p={r['p']:3d} frac={r['frac']:.2f} seed={r['seed']} mem={r['mem']} cross={r['cross']} "
                  f"paper_drift={r['paper_drift']} steps={r['steps_run']} final={r['final_test']:.3f} ({r['seconds']}s)"
                  + (" STOPPED" if r["stopped_by_time_budget"] else ""), flush=True)
