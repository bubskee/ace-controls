"""All contiguous holdouts, energy-matched ridge, and a nonlinear Jacobian check on CPU."""
import sys
from pathlib import Path
import json
import numpy as np
import torch
sys.path.insert(0,"cores/src")
from cores.extraction import ace
from cores.infer_mechanism import fit_operator_svd
from cores.models import build_modadd_dataloader
from cores.modular_addition_analysis import load_model

def sweep():
    rows=json.loads(Path("results/toy/modadd_controls_wd1.json").read_text())["rows"]
    loader=build_modadd_dataloader(train=False,batch_size=512)
    X=torch.cat([x for x,y in loader])
    Y=torch.cat([y[:,-1] for x,y in loader])
    rng=np.random.default_rng(512)
    for row in rows:
        if row["epoch"] not in (800, 20000):
            continue
        r=row["r_nec"]; ep=row["epoch"]; m=row["model"]
        model=load_model(ep,m,torch.device("cpu"),Path("cores/experiments/modular_addition/ckpts/wd1"))
        with torch.no_grad():
            _,h1,h2=model(X,return_states=True)
        H=h2[:,-1]
        q,_=ace(H,lambda:lambda h:model.lm_head(h),return_full=True)
        random=np.linalg.qr(rng.normal(size=(128,r)))[0]
        M=np.stack([H[Y==c].double().mean(0).numpy() for c in range(53)])
        orig=M@q[:,:r].detach().double().numpy()
        proj=M@random
        pca=np.linalg.svd(H.double().numpy()-H.double().mean(0).numpy(),full_matrices=False)[2][:r].T
        scores={}
        for label,rep in [('ACE',orig),('PCA',M@pca),('random',proj),('scrambled ACE',orig[rng.permutation(53)])]:
            vals=[]; adjusted=[]
            for shift in range(53):
                shifted=np.roll(rep,shift,axis=0)
                ref=np.roll(orig,shift,axis=0)
                # The author's fixed holdout in the rolled array. Match training-design Frobenius norms.
                start=np.random.RandomState(0).randint(53); train=np.ones(53,bool)
                train[(start+np.arange(4))%53]=False
                train &= np.roll(train,-1)
                norm=lambda z:np.linalg.norm(z[train]-z[train].mean(0))
                for out,scale in [(vals,1),(adjusted,norm(ref)/norm(shifted))]:
                    out.append(fit_operator_svd(shifted*scale,k_max=52,svd_tol=1e-3,ridge=1,holdout_style="block_edges")[2])
            scores[label]=[round(float(np.median(vals)),3),round(float(np.min(vals)),3),round(float(np.median(adjusted)),3)]
        print(ep,m,'[median, worst, matched-scale median]',scores,flush=True)
        if ep==800 and m==0:
            model=model.double()
            h=h1[:3].double().detach().requires_grad_()
            mask=torch.triu(torch.full((2,2),float("-inf"),dtype=h.dtype),1)
            fn=lambda u:model.lm_head(model.layer2(u,src_mask=mask))[:,-1,17]
            g=torch.autograd.grad(fn(h).sum(),h)[0]
            v=torch.tensor(rng.normal(size=h.shape),dtype=h.dtype)
            for step in (1e-5,1e-7,1e-8):
                finite=(fn(h+step*v)-fn(h-step*v))/(2*step)
                deriv=(g*v).sum((1,2))
                print('nonlinear directional derivative, step, analytic, finite',step,deriv.detach().tolist(),finite.detach().tolist())
if __name__=='__main__': sweep()
