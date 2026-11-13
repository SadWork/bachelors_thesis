"""Stronger quantization controls with row codebooks and the same train-only ridge fit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from probe import (CompiledDense, CompiledRoute, dense_variant, library, load_model, load_data,
                   np_float, quantize_probability, validation)

WORKSPACE = Path(__file__).resolve().parents[3]


@torch.no_grad()
def codebook(p, bits):
    bins = 2**bits
    t = torch.linspace(0., 1., bins, dtype=p.dtype)
    maximum = p.amax(-1, keepdim=True)
    candidates = [maximum*t, maximum*torch.exp(12*(t-1))]
    reconstructions = []
    p = p.contiguous()
    for centroids in candidates:
        for _ in range(28):
            bounds = ((centroids[:,1:] + centroids[:,:-1])*.5).contiguous()
            code = torch.searchsorted(bounds, p)
            sums = p.new_zeros(centroids.shape).scatter_add_(1, code, p)
            counts = p.new_zeros(centroids.shape).scatter_add_(1, code, torch.ones_like(p))
            centroids = torch.where(counts > 0, sums/counts.clamp_min(1), centroids).sort(-1).values
        bounds = ((centroids[:,1:] + centroids[:,:-1])*.5).contiguous()
        q = centroids.gather(1, torch.searchsorted(bounds, p))
        q /= q.sum(-1, keepdim=True)
        reconstructions.append(q)
    a,b = reconstructions
    return torch.where((p-a).square().sum(-1,keepdim=True) <= (p-b).square().sum(-1,keepdim=True),a,b)


@torch.no_grad()
def routing_hidden(x, model, matrices):
    for layer,p in zip(model.hidden,matrices):
        rows = []
        for xx in x.split(24):
            u = xx[:,None,:]*layer.weight[None,:,None] + layer.bias[None,:,None]
            rows.append((torch.where(u > 0,2*u,.5*u)*p).sum(-1))
        x = torch.cat(rows)
    return x


def ridge(hidden, logits, prior):
    h, z = hidden.double(), logits.double()
    mu, center = h.mean(0), z.mean(0)
    h, z = h-mu, z-center
    gram = h.T @ h
    penalty = 1e-6 + .01*gram.diag().mean()
    initial = prior.T.double()
    correction = torch.linalg.solve(gram + penalty*torch.eye(gram.size(0),dtype=gram.dtype),
                                    h.T@(z-h@initial))
    w = initial+correction
    return w.T.float(), (center-mu@w).float()


def apply_head(compiled, weight, bias):
    compiled.arrays[-2:] = [np_float(weight),np_float(bias)]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    archive=WORKSPACE/"work"/"cifar10_shared_scalar"/"data"
    ref=WORKSPACE/"outputs"/"compact_routing_paper_check"
    dll=library(WORKSPACE/"outputs"/"compact_routing_article"/"systems"/"compiled_reference.dylib")
    data,metadata=load_data(archive,args.out,"cpu")
    x,_=data[0]
    train_ids=np.load(args.out/"split.npz")["train"]
    sample=np.random.default_rng(93187).choice(len(x),1536,replace=False)
    source_indices=train_ids[sample]
    cal=x[torch.from_numpy(sample)]
    sets={key:(np_float(a),b.numpy()) for key,(a,b) in zip(("validation","test"),data[1:])}
    rows=[]
    def score(seed,name,model,stored):
        row=dict(seed=seed,method=name,packed_bytes_estimate=stored,
                 **{key:validation(model,*values) for key,values in sets.items()})
        rows.append(row)
        print(json.dumps(row),flush=True)
    for seed in (0,1,2):
        pre=load_model(ref,"logit_pre",seed)
        exact=CompiledRoute(pre,(3072,160,160,160),metadata,dll)
        logits=torch.from_numpy(exact(np_float(cal)))
        for bits in (2,3,4):
            for mode in ("log","codebook"):
                matrices=[codebook(p,bits) if mode=="codebook" else quantize_probability(p,bits,mode)
                          for p in exact.expanded]
                compiled=CompiledRoute(pre,(3072,160,160,160),metadata,dll)
                compiled.arrays[1]=np_float(torch.cat([p.flatten() for p in matrices]))
                hidden=routing_hidden(cal,pre,matrices)
                new_head=ridge(hidden,logits,pre.head.weight)
                header=4*sum(len(p) for p in matrices)*(2**bits if mode=="codebook" else 1)
                stored=sum(math.ceil(p.numel()*bits/8) for p in matrices)+header
                stored+=sum(a.nbytes for a in compiled.arrays[4:])
                score(seed,f"route_{mode}{bits}",compiled,stored)
                apply_head(compiled,*new_head)
                score(seed,f"route_{mode}{bits}_ridge",compiled,stored)
        dense=load_model(ref,"full_dense",seed)
        dense_teacher=dense(cal)
        for bits in (2,4):
            compiled,stored=dense_variant(dense,bits,dll)
            weights,biases=map(torch.from_numpy,compiled.arrays[:2])
            h=cal
            start=0
            for i in range(4):
                n=h.shape[1]
                w=weights[start:start+160*n].view(160,n)
                u=h @ w.T + biases[i*160:(i+1)*160]
                h=torch.where(u>0,2*u,.5*u)
                start+=160*n
            prior=torch.from_numpy(compiled.arrays[-2])
            apply_head(compiled,*ridge(h,dense_teacher,prior))
            score(seed,f"dense_linear{bits}_ridge",compiled,stored+10*160*(4-bits/8))
    payload={"calibration_index_hash":hashlib.sha256(source_indices.tobytes()).hexdigest(),"rows":rows}
    (args.out/"results.json").write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":
    main()
