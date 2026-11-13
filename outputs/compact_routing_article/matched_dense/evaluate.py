"""Evaluate calibrated INT8 factorized Dense at the compact export's byte budget."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/"independent"))
from data import load_data as load_cifar100
from models import Classifier, birelu
from train import evaluate

loader=importlib.util.spec_from_file_location("cifar10_loader",ROOT/"outputs"/"compact_routing_paper_check"/"data.py")
cifar10=importlib.util.module_from_spec(loader)
loader.loader.exec_module(cifar10)


def hidden(model, x):
    for layer in model.hidden:
        x=birelu(layer(x))
    return x


@torch.no_grad()
def quantize(weight, bits=8):
    limit=2**(bits-1)-1
    scale=weight.abs().amax(-1,keepdim=True).clamp_min(1e-8)/limit
    return (weight/scale).round().clamp(-limit,limit)*scale


def ridge(h,z,prior):
    h,z=h.double(),z.double()
    mean,target=h.mean(0),z.mean(0)
    h,z=h-mean,z-target
    gram=h.T@h
    penalty=1e-6+.01*gram.diag().mean()
    initial=prior.T.double()
    w=initial+torch.linalg.solve(gram+penalty*torch.eye(gram.shape[0],dtype=gram.dtype),
                                 h.T@(z-h@initial))
    return w.T.float(),(target-mean@w).float()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--dataset",required=True,choices=("cifar10","cifar100"))
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--device",default="mps")
    args=p.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    is10=args.dataset=="cifar10"
    classes,rank=(10,31) if is10 else (100,41)
    loader=cifar10.load_data if is10 else load_cifar100
    data,meta=loader(args.data,args.out,"cpu")
    x,y=data[0]
    ids=np.load(args.out/"split.npz")["train"]
    if is10:
        chosen=torch.from_numpy(np.random.default_rng(93187).choice(len(x),1536,replace=False))
    else:
        rng=torch.Generator().manual_seed(1223091)
        parts=[torch.where(y==i)[0] for i in range(100)]
        parts=[v[torch.randperm(len(v),generator=rng)] for v in parts]
        base=torch.cat([v[:15] for v in parts])
        rest=torch.cat([v[15:] for v in parts])
        chosen=torch.cat((base,rest[torch.randperm(len(rest),generator=rng)[:1536-len(base)]]))
    cal=x[chosen]
    source_ids=chosen.numpy() if not is10 else ids[chosen.numpy()]
    val,test=[(a.to(args.device),b.to(args.device)) for a,b in data[1:]]
    rows=[]
    for seed in (0,1,2):
        file=HERE/args.dataset/"checkpoints"/f"factor_dense_r{rank}_s{seed}.pt"
        item=torch.load(file,map_location="cpu",weights_only=True)
        original=Classifier("factor_dense",rank,classes=classes).eval()
        original.load_state_dict(item["best"]["state"],strict=True)
        teacher=original(cal)
        dense=copy.deepcopy(original)
        layers=[linear for layer in dense.hidden for linear in (layer.down,layer.up)] + [dense.head]
        for layer in layers:
            layer.weight.copy_(quantize(layer.weight))
        packed=sum(layer.weight.numel()+4*layer.weight.shape[0]+(4*layer.bias.numel() if layer.bias is not None else 0)
                   for layer in layers)
        h=hidden(dense,cal)
        w,b=ridge(h,teacher,dense.head.weight)
        ridge_model=copy.deepcopy(dense)
        ridge_model.head.weight.copy_(quantize(w))
        ridge_model.head.bias.copy_(b)
        for name,model,bytes_ in (
            ("factor_fp32",original,4*sum(p.numel() for p in original.parameters())),
            ("factor_int8",dense,packed),("factor_int8_ridge",ridge_model,packed)):
            model.to(args.device)
            row=dict(dataset=args.dataset,seed=seed,rank=rank,method=name,best_epoch=item["best"]["epoch"],
                     packed_bytes_estimate=bytes_,validation=evaluate(model,val),test=evaluate(model,test))
            rows.append(row)
            print(json.dumps(row),flush=True)
    hash_=hashlib.sha256(source_ids.tobytes()).hexdigest()
    (args.out/"results.json").write_text(json.dumps(dict(split_sha256=meta["split_sha256"],
                                                        calibration_index_hash=hash_,rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
