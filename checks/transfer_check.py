"""Small-pool within-GPT-2 replication and target-only calibration control. Run from the repository root."""
import argparse
import hashlib
import itertools
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path[:0] = ["cores/src", "controls/shims"]
from cores.ioi import common as c
from cores.ioi.reduced_model import ridge, share

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--score-only", action="store_true", help="Score previously extracted coordinates")
parser.add_argument("--data-dir", type=Path, help="Directory with the coordinate files")
parser.add_argument("--device", default="mps", help="Device for fresh extraction, e.g. mps, cpu, cuda")
args = parser.parse_args()
dest = args.data_dir or Path("results/transfer" if args.score_only else "runs/transfer")
models = {"gpt2": (6, 9, 14, 2), "gpt2-medium": (12, 20, 12, 37), "gpt2-large": (17, 27, 24, 38)}
device = torch.device(args.device)

@torch.no_grad()
def coords(model, data, layers, bases, swap=False):
    chunks = []
    hook = None
    if swap:
        hook = c.get_blocks(model)[layers[0]-1].register_forward_hook(c.SwapHook(bases[0], data, 2, device))
    try:
        for start in range(0, len(data["y"]), 16):
            sl = slice(start,start+16)
            hs = model(input_ids=data["ids"][sl].to(device), attention_mask=data["att"][sl].to(device),
                       output_hidden_states=True, use_cache=False).hidden_states
            parts = []
            for j,(l,q) in enumerate(zip(layers,bases)):
                pos = torch.as_tensor(data["pos"][sl,2+j],device=device)
                x = hs[l][torch.arange(len(pos),device=device),pos]
                parts.append(x.cpu().numpy() @ q)
            chunks.append(np.hstack(parts))
    finally:
        if hook: hook.remove()
    return np.vstack(chunks).astype(float)

def extract(tag):
    ls, le, rs, re = models[tag]
    tok, model = c.load_model("openai-community/"+tag, "float32", None, device, sink=True)
    setup = c.setup_data(tok)
    def pairs(n,seed):
        return c.build_role_pairs(tok,n,np.random.default_rng(seed),setup["fr_ext"],c.ent(setup["ext_names"]),p_noun=0,p_pronoun=0)[0]
    layers = [ls,le]
    H,G = c.capture(model,pairs(512,205),layers,512,8,device)
    bases = []
    for l,t,r in [(ls,"S2",rs),(le,"END",re)]:
        fac = c.role_factors(H[l][t],G[l][t],device)
        bases.append(c.core_basis(fac,r,"orth_write")[0])
    fit = coords(model,pairs(512,405),layers,bases)
    df = fit[1::2]-fit[::2]
    A = np.linalg.lstsq(df[:,:rs],df[:,rs:],rcond=None)[0]
    test,_ = c.stratified_pairs(tok,12,201,setup["fr_test"],setup["test_names"])
    frame = np.repeat(np.arange(len(setup["fr_test"])),12)
    h = coords(model,test,layers,bases)
    swapped = coords(model,test,layers,bases,True)
    d = h[1::2]-h[::2]
    digest = hashlib.sha256(np.asarray(test["ids"]).tobytes()).hexdigest()
    np.savez(dest/f"transfer_{tag}.npz", S=d[:,:rs], E=d[:,rs:], response=(swapped-h)[::2,rs:], A=A, frame=frame, digest=digest)
    print(tag, "own", share((swapped-h)[::2,rs:], d[:,:rs]@A),flush=True)

def score():
    data = {m:np.load(dest/f"transfer_{m}.npz") for m in models}
    assert len({str(d["digest"]) for d in data.values()}) == 1
    for a,b in itertools.permutations(data,2):
        x,y = data[a],data[b]
        halves = np.array_split(np.random.default_rng(0).permutation(np.unique(y['frame'])),2)
        pred = np.zeros_like(y['E']); direct = pred.copy(); rotated = np.zeros((10,*pred.shape))
        rng = np.random.default_rng(10)
        middle = [np.linalg.qr(rng.normal(size=(x['A'].shape[0],)*2))[0] @ x['A'] for _ in rotated]
        for h in halves:
            tr = np.isin(y['frame'],h)
            start = y['S'][~tr] @ ridge(y['S'][tr],x['S'][tr])
            end = ridge(x['E'][tr],y['E'][tr])
            pred[~tr] = start @ x['A'] @ end
            direct[~tr] = y['S'][~tr] @ ridge(y['S'][tr],y['E'][tr])
            for i,m in enumerate(middle): rotated[i,~tr] = start @ m @ end
        print(a,'to',b,'carried, direct, own, rotated mean:',*[round(share(y['response'],v),4) for v in [pred,direct,y['S']@y['A']]], round(np.mean([share(y['response'],v) for v in rotated]),4),flush=True)

        # Conditional intervals over held-out frames, treating the fitted fold models as fixed.
        frame = y['frame']
        energy = lambda err: np.bincount(frame,weights=np.square(err).sum(1))
        diff = energy(y['response']-direct) - energy(y['response']-pred)
        den = energy(y['response'])
        ids = np.random.default_rng(346).integers(0,len(den),size=(4000,len(den)))
        print('carried minus direct, frame bootstrap 95%:',np.round(np.quantile(diff[ids].sum(1)/den[ids].sum(1),[.025,.975]),3))

if __name__=='__main__':
    if not args.score_only:
        dest.mkdir(parents=True, exist_ok=True)
        for tag in models:
            extract(tag)
            if device.type == "mps":
                torch.mps.empty_cache()
    score()
