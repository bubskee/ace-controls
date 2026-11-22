"""Find the first prefix satisfying both criteria. Compare two-position and last-position interventions."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'cores/src')
from cores.models import build_modadd_dataloader
from cores.modular_addition_analysis import load_model
from cores.extraction import _balanced_svd
device=torch.device('cpu')
data=list(build_modadd_dataloader(train=False,batch_size=512))
X=torch.cat([a for a,b in data]); Y=torch.cat([b[:,-1] for a,b in data])
mask=torch.triu(torch.full((2,2),float('-inf')),diagonal=1)
for seed in range(3):
    m=load_model(800,seed,device,Path('cores/experiments/modular_addition/ckpts/wd1'))
    with torch.no_grad():
        logits,h1,_=m(X,return_states=True)
    base=(logits[:,-1].argmax(-1)==Y).float().mean().item()
    h=h1.detach().requires_grad_(True)
    tail=lambda v:m.lm_head(m.layer2(v,src_mask=mask))[:,-1]
    logits=tail(h)
    obs=torch.zeros(256,256,dtype=torch.float64)
    for j in range(53):
        g=torch.autograd.grad(logits[:,j].sum(),h,retain_graph=True)[0].reshape(-1,256).double()
        obs+=g.T@g/len(g)
    for last in (False,True):
        z=h1[:,-1] if last else h1.flatten(1)
        zc=(z-z.mean(0)).double()
        cov=zc.T@zc/(len(z)-1)
        sens=obs[-128:,-128:] if last else obs
        ace=_balanced_svd(cov,sens)[0]
        for name,q in [('ACE',ace),('gradient',torch.linalg.eigh(sens)[1].flip(1))]:
            q=q.float()
            with torch.no_grad():
                for r in range(1,q.shape[1]+1):
                    kept=(z@q[:,:r])@q[:,:r].T
                    def score(a):
                        state=torch.stack([h1[:,0],a],1) if last else a.reshape_as(h1)
                        return float((tail(state).argmax(-1)==Y).float().mean())
                    if score(kept)>=base-1e-3 and score(z-kept)<=2/53:
                        print(seed,'last' if last else 'both',name,r,score(kept),score(z-kept),flush=True)
                        break
                else: print(seed,'last' if last else 'both',name,'no joint crossing',flush=True)
