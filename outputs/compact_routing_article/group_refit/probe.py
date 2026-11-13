"""Fit a positive mixture of tile distributions and explicit routes on train only."""
import json
import math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/"support"))
from activation_support import (CALIBRATION, fit_layer, features, representations, ridge_head,
                                validation_data, load_teacher, deploy_layers, original_test_set, evaluate)

KS=(32,32,64,32)
TILES=(8,16)
SEEDS=(0,1,2)


def spatial_groups(tile):
    side=32//tile
    return (torch.arange(3)[:,None,None]*side**2
            + (torch.arange(32)[None,:,None]//tile)*side
            + torch.arange(32)[None,None,:]//tile).flatten()


def simplex(x):
    sorted_values=x.sort(-1,descending=True).values
    ranks=torch.arange(1,len(x[0])+1,dtype=x.dtype)
    offsets=(sorted_values.cumsum(-1)-1.)/ranks
    active=(sorted_values>offsets).sum(-1,keepdim=True).clamp_min(1)-1
    return (x-offsets.gather(-1,active)).clamp_min(0)


def fit_first(cal,target,layer,k,tile):
    w,b,p,_,_=layer
    group=spatial_groups(tile)
    groups=int(group.max())+1
    n=p.shape[-1]
    top,index=p.topk(k,dim=-1)
    membership=F.one_hot(group,groups).float().contiguous()
    group_size=membership.sum(0)
    group_head=group[index]
    removed_mass=p @ membership
    removed_mass.scatter_add_(1,group_head,-top)
    omitted=group_size.expand(len(p),-1).clone().scatter_add_(1,group_head,-torch.ones_like(top))
    uniform=(removed_mass/omitted.clamp_min(1)).clamp_min(0).masked_fill_(omitted==0,0.)
    prior=torch.cat((uniform*group_size, top-uniform.gather(1,group_head)),dim=-1).clamp_min(0)
    prior/=prior.sum(-1,keepdim=True)
    results=[]
    for rows in torch.arange(len(w)).split(8):
        tiles=[]
        for batch in cal.split(128):
            tiles.append((features(batch,w[rows],b[rows]) @ membership) / group_size)
        shared=torch.cat(tiles,dim=1)
        explicit=features(cal,w[rows],b[rows],index[rows])
        design=torch.cat((shared,explicit),dim=-1)
        center=design.mean(-1,keepdim=True)
        design-=center
        y=target[:,rows].T.unsqueeze(-1)-center
        gram=design.transpose(1,2)@design/len(cal)
        cross=(design.transpose(1,2)@y).squeeze(-1)/len(cal)
        penalty=1e-8+1e-4*gram.diagonal(dim1=1,dim2=2).mean(-1)
        gram.diagonal(dim1=1,dim2=2).add_(penalty[:,None])
        cross+=penalty[:,None]*prior[rows]
        step=gram.abs().sum(-1).amax(-1,keepdim=True).clamp_min(1e-12).reciprocal()
        coefficient=accelerated=prior[rows]
        momentum=1.
        for _ in range(160):
            gradient=(gram @ accelerated.unsqueeze(-1)).squeeze(-1)-cross
            estimate=simplex(accelerated-step*gradient)
            next_momentum=(1+math.sqrt(1+4*momentum**2))/2
            accelerated=estimate+(momentum-1)/next_momentum*(estimate-coefficient)
            coefficient,momentum=estimate,next_momentum
        results.append(coefficient)
    coefficient=torch.cat(results).half().float()
    scale=coefficient.sum(-1,keepdim=True).reciprocal()
    decoded=coefficient*scale
    alpha,beta=decoded.split((groups,k),dim=-1)
    q=(alpha/group_size)[:,group].contiguous().scatter_add_(1,index,beta)
    torch.testing.assert_close(q.sum(-1),torch.ones(len(q)),atol=5e-6,rtol=5e-6)
    return (w,b,q,None,None), {"tile":tile,"coefficients":coefficient.half(),
                              "row_scale":scale,"indices":index,"groups":groups}


def fit_model(state,cal,tile):
    teacher=deploy_layers(state,None,"full")
    target=representations(cal,teacher)
    first,export=fit_first(cal,target[0],teacher[0],KS[0],tile)
    layers=[first]
    hidden=representations(cal,layers)[-1]
    for idx,k in enumerate(KS[1:],1):
        layer=fit_layer(hidden,target[idx],teacher[idx],k,"topk_refit_ridge")
        layers.append(layer)
        hidden=representations(hidden,[layer])[-1]
    logits=F.linear(target[-1],state["head.weight"],state["head.bias"])
    head=ridge_head(hidden,logits,state["head.weight"])
    bytes_estimate=137000 - 160*32*4 + 160*export["groups"]*2
    return layers,head,export,bytes_estimate


def write(name,x):
    (HERE/name).write_text(json.dumps(x,indent=2)+"\n")


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    x,y,cal,cal_hash=validation_data("cifar10",calibration_size=CALIBRATION)
    write("protocol.json",{"ks":KS,"tiles":TILES,"seeds":SEEDS,"calibration_index_hash":cal_hash,
                           "selection":"Smallest packed size within 1pp of teacher mean validation accuracy"})
    rows=[]
    for seed in SEEDS:
        state=load_teacher(seed)
        exact_head=state["head.weight"],state["head.bias"]
        teacher=deploy_layers(state,None,"full")
        base=evaluate((x,y),teacher,exact_head)[0]
        for tile in TILES:
            layers,head,export,storage=fit_model(state,cal,tile)
            metric=evaluate((x,y),layers,head)[0]
            row={"seed":seed,"tile":tile,"validation":metric,"teacher_validation":base,"packed_bytes_estimate":storage}
            rows.append(row)
            print(json.dumps(row),flush=True)
            torch.save({"layers":layers,"head":head,"first_export":export,"packed_bytes_estimate":storage},
                       HERE/f"tile{tile}_s{seed}.pt")
    write("validation.json",rows)
    feasible=[]
    for tile in TILES:
        subset=[row for row in rows if row["tile"]==tile]
        delta=sum(row["validation"]["accuracy_pct"]-row["teacher_validation"]["accuracy_pct"]
                  for row in subset)/len(subset)
        if delta >= -1:
            feasible.append((subset[0]["packed_bytes_estimate"],tile))
    if not feasible:
        print("NO_TEST_CONFIGURATION",flush=True)
        return
    _,chosen=min(feasible)
    write("frozen.json",{"tile":chosen,"ks":KS,"calibration_index_hash":cal_hash,"retuning":False})
    test=original_test_set()
    measured=[]
    for seed in SEEDS:
        state=load_teacher(seed)
        teacher=deploy_layers(state,None,"full")
        base=evaluate(test,teacher,(state["head.weight"],state["head.bias"]))[0]
        candidate=torch.load(HERE/f"tile{chosen}_s{seed}.pt",map_location="cpu",weights_only=True)
        score=evaluate(test,candidate["layers"],candidate["head"])[0]
        row={"seed":seed,"tile":chosen,"test":score,"teacher_test":base,
             "packed_bytes_estimate":candidate["packed_bytes_estimate"]}
        measured.append(row)
        print(json.dumps(row),flush=True)
    write("test.json",measured)


if __name__=="__main__":
    main()
