"""Frozen CIFAR-100 export evaluation, including train-only refit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

from compression import expand_tail, groups_for
from data import load_data
from metal_mean import MetalMean
from models import Classifier
from train import evaluate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "support"))
from activation_support import CALIBRATION, fit_models, representations
from probe import deploy_layers

FOCUS32 = (64,32,64,32)
FOCUS13 = (128,64,128,64)


class FrozenLayer(nn.Module):
    def __init__(self, values):
        super().__init__()
        w,b,p = values
        self.register_buffer("w", w.contiguous())
        self.register_buffer("b", b.contiguous())
        self.register_buffer("p", p.contiguous())

    def forward(self, x):
        if x.device.type == "mps":
            return MetalMean.apply(x, self.w, self.p, self.b)
        u = x[:,None,:]*self.w[None,:,None] + self.b[None,:,None]
        return (torch.where(u > 0, 2*u, .5*u) * self.p).sum(-1)


class Frozen(nn.Module):
    def __init__(self, layers, head, input_sizes=(3072,160,160,160)):
        super().__init__()
        expanded = []
        for layer,n in zip(layers,input_sizes):
            w,b,p,index,tail = layer
            if index is not None:
                uniform = tail[:,None] / (n-index.shape[1])
                p = uniform.expand(-1,n).clone().scatter_(1,index,p)
            expanded.append(FrozenLayer((w,b,p)))
        self.hidden = nn.Sequential(*expanded)
        self.register_buffer("weight",head[0].contiguous())
        self.register_buffer("bias",head[1].contiguous())

    def forward(self,x):
        return torch.nn.functional.linear(self.hidden(x),self.weight,self.bias)


class MeanBeforeBiReLU(nn.Module):
    """Keep the identical tile masses and explicit routes, but collapse each tail to a mean."""
    def __init__(self, layer, k):
        super().__init__()
        w,b,p,_,_=layer
        groups = groups_for("tiles")
        top,ids = p.topk(k,dim=-1)
        selected_group = groups[ids]
        mass = p.double().new_zeros((len(p),48)).scatter_add_(1,groups.expand_as(p),p.double())
        mass = mass.scatter_add_(1,selected_group,-top.double()).clamp_min(0)
        counts = p.new_full((len(p),48),64.).scatter_add_(1,selected_group,-torch.ones_like(top))
        mass = mass.masked_fill(counts==0,0.).float()
        for key, value in (("w",w),("b",b),("top",top),("ids",ids),
                           ("selected_group",selected_group),("mass",mass),("count",counts.clamp_min(1))):
            self.register_buffer(key,value)

    def forward(self,x):
        tiles=x.reshape(len(x),3,4,8,4,8).sum(dim=(3,5)).flatten(1)
        selected=x[:,self.ids]
        groups=self.selected_group.expand(len(x),-1,-1)
        remaining=tiles[:,None,:].expand(-1,len(self.w),-1).scatter_add(2,groups,-selected)
        mean=remaining/self.count
        f_explicit=selected*self.w[:,None]+self.b[:,None]
        f_tail=mean*self.w[:,None]+self.b[:,None]
        return (torch.where(f_explicit>0,2*f_explicit,.5*f_explicit)*self.top).sum(-1) + (
            torch.where(f_tail>0,2*f_tail,.5*f_tail)*self.mass).sum(-1)


def stored_fp32(layers, head):
    return 4 * sum(v.numel() for v in head) + 4 * sum(w.numel()+b.numel()+p.numel()
                                                      for w,b,p,_,_ in layers)


def sparse_bytes(layers, head):
    return 4*sum(v.numel() for v in head) + sum(
        4*(w.numel()+b.numel()+tail.numel()) + 4*p.numel() for w,b,p,index,tail in layers
    )


def tiled_bytes(layers, head, k, groups):
    return stored_fp32(layers,head) - 4*layers[0][2].numel() + len(layers[0][0])*(6*k+4*groups)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--device",default="mps")
    parser.add_argument("--seeds",nargs="+",type=int,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    train,val,test = load_data(args.data,args.out,"cpu")[0]
    rng=torch.Generator().manual_seed(1223091)
    # Class-balanced train only: 15 per class plus 36 from unused train rows.
    labels=train[1]
    parts=[torch.where(labels==i)[0] for i in range(100)]
    parts=[p[torch.randperm(len(p),generator=rng)] for p in parts]
    base=torch.cat([p[:15] for p in parts])
    rest=torch.cat([p[15:] for p in parts])
    extra=rest[torch.randperm(len(rest),generator=rng)[:CALIBRATION-len(base)]]
    sample=torch.cat((base,extra))
    cal=train[0][sample]
    assert len(cal)==CALIBRATION
    sets={key:(x.to(args.device),y.to(args.device)) for key,(x,y) in [("validation",val),("test",test)]}
    rows=[]
    def score(seed,kind,model,storage):
        model.to(args.device).eval()
        row={"seed":seed,"method":kind,"packed_bytes_estimate":storage,
             **{split:evaluate(model,batch) for split,batch in sets.items()}}
        rows.append(row)
        print(json.dumps(row),flush=True)
    for seed in args.seeds:
        name=f"logit_pre_r8_s{seed}.pt"
        state=torch.load(HERE/"teacher"/"checkpoints"/name,weights_only=True,map_location="cpu")["best"]["state"]
        head=state["head.weight"],state["head.bias"]
        teacher=deploy_layers(state,None,"full")
        score(seed,"teacher",Frozen(teacher,head),stored_fp32(teacher,head))
        plans=[("global64_first",64,groups_for("global")),("tiles64_first",64,groups_for("tiles")),
               ("tiles96_first",96,groups_for("tiles"))]
        plans.extend((f"random64_first_s{i}",64,groups_for("random_tiles",7781+i)) for i in range(3))
        for kind,k,groups in plans:
            w,b,p,_,_=teacher[0]
            q=expand_tail(p,k,groups)
            layers=[(w,b,q,None,None),*teacher[1:]]
            score(seed,kind,Frozen(layers,head),tiled_bytes(teacher,head,k,int(groups.max())+1))
        proxy=Frozen(teacher,head)
        proxy.hidden[0]=MeanBeforeBiReLU(teacher[0],64)
        score(seed,"tiles64_first_mean_proxy",proxy,tiled_bytes(teacher,head,64,48))
        target=representations(cal,teacher)
        for config,label in ((FOCUS13,"focus13"),(FOCUS32,"focus32")):
            uniform=deploy_layers(state,config,"uniform_tail")
            score(seed,label+"_uniform",Frozen(uniform,head),sparse_bytes(uniform,head))
            modes=("topk_ridge",) if label=="focus13" else ("topk_ridge","topk_refit_ridge")
            for mode in modes:
                fitted,new_head=fit_models(state,cal,teacher,target,config,mode)
                score(seed,label+"_"+mode,Frozen(fitted,new_head),sparse_bytes(fitted,new_head))
        for kind in ("full_dense","factor_dense"):
            dense_state=torch.load(HERE/"controls"/"checkpoints"/f"{kind}_r8_s{seed}.pt",
                                   weights_only=True,map_location="cpu")["best"]["state"]
            dense=Classifier(kind,8,classes=100)
            dense.load_state_dict(dense_state,strict=True)
            score(seed,kind,dense,4*sum(v.numel() for v in dense.parameters()))
            if kind=="full_dense":
                for bits in (2,4,8):
                    dense=Classifier(kind,8,classes=100)
                    dense.load_state_dict(dense_state,strict=True)
                    total=0
                    for layer in [*dense.hidden,dense.head]:
                        max_code=2**(bits-1)-1
                        scale=layer.weight.abs().amax(-1,keepdim=True).clamp_min(1e-8)/max_code
                        layer.weight.copy_((layer.weight/scale).round().clamp(-max_code,max_code)*scale)
                        total+=math.ceil(layer.weight.numel()*bits/8)+4*(len(scale)+layer.bias.numel())
                    score(seed,f"dense_linear{bits}",dense,total)
    digest=hashlib.sha256(sample.numpy().tobytes()).hexdigest()
    (args.out/"results.json").write_text(json.dumps(dict(calibration_index_hash=digest,rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
