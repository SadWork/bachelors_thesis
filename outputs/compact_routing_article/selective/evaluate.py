"""Exploratory teacher-agreement fallback with frozen thresholds and a disjoint evaluation split."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"systems"))
from benchmark import (CompiledRoute, RefitRoute, benchmark, intensity_data, library, load_data, load_model,
                       make_library, metrics, np_float, predictions)

ROOT=HERE.parents[2]
THRESHOLDS=np.linspace(0,8,81)
CALIBRATION=2000
RISK=.01
ALPHA=.05


def margin(z):
    two=np.partition(z,-2,axis=-1)[...,-2:]
    return two.max(-1)-two.min(-1)


def null_binomial_cdf(n,p):
    prob=(1-p)**n
    masses=[prob]
    for k in range(n):
        prob*=((n-k)/(k+1))*p/(1-p)
        masses.append(prob)
    return np.cumsum(masses)


class Hybrid:
    def __init__(self, fast, exact, threshold):
        self.fast,self.exact,self.threshold=fast,exact,threshold
    def __call__(self,x):
        value=self.fast(x)
        return value if margin(value)[0]>=self.threshold else self.exact(x)


def main():
    torch.set_num_threads(1)
    data,metadata=load_data(ROOT/"work/cifar10_shared_scalar/data",HERE,"cpu")
    x,target=data[2]
    x=np_float(x)
    rng=np.random.default_rng(237873)
    rows=rng.permutation(len(x))
    cal,test=rows[:CALIBRATION],rows[CALIBRATION:]
    protocol=dict(calibration_size=len(cal),eval_size=len(test),seed=237873,risk=RISK,alpha=ALPHA,
                  thresholds=THRESHOLDS.tolist(),cal_sha256=hashlib.sha256(cal.tobytes()).hexdigest(),
                  selection="Largest calibration coverage passing lower-tail binomial test at alpha/(3*81)",
                  note="Exploratory split of the existing public test; risk refers to extra disagreement with teacher.")
    (HERE/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    exact_lib=library(HERE.parent/"systems/compiled_reference.dylib")
    refit_lib=make_library()
    _,_,level,norm=intensity_data(metadata)
    results=[]
    cutoff=null_binomial_cdf(len(cal),RISK)
    for seed in (0,1,2):
        model=load_model(ROOT/"outputs/compact_routing_paper_check","logit_pre",seed)
        exact=CompiledRoute(model,(3072,160,160,160),metadata,exact_lib)
        ref=torch.load(HERE.parent/"support"/f"focus32_topk_refit_ridge_seed{seed}.pt",
                       weights_only=True,map_location="cpu")
        fast=RefitRoute(ref,refit_lib,level,norm,half=True,hidden_buckets=True)
        fast.prepare(128,False)
        base=predictions(exact,x,128)
        approx=predictions(fast,x,128)
        m=margin(approx)
        wrong=base.argmax(-1)!=approx.argmax(-1)
        accepted=(m[cal,None]>=THRESHOLDS)
        errors=(accepted & wrong[cal,None]).sum(0)
        allowed=np.flatnonzero(cutoff[errors]<=ALPHA/(3*len(THRESHOLDS)))
        threshold=float(THRESHOLDS[allowed[0]]) if len(allowed) else float("inf")
        use_fast=m[test]>=threshold
        hybrid=np.where(use_fast[:,None],approx[test],base[test])
        comparison=dict(seed=seed,threshold=threshold if np.isfinite(threshold) else None,
                        coverage=float(use_fast.mean()),disagreement=float((wrong[test]&use_fast).mean()),
                        full_approx_disagreement=float(wrong[test].mean()),
                        teacher=metrics(base[test],target[test]),approximate=metrics(approx[test],target[test]),
                        hybrid=metrics(hybrid,target[test]))
        if seed==0:
            fast.prepare(1,False)
            wrapper=Hybrid(fast,exact,threshold)
            xs=np.ascontiguousarray(x[test])
            comparison["timing_batch1"]=benchmark({"exact":exact,"approximate":fast,"hybrid":wrapper},xs,1,200)
            comparison["joint_persistent_bytes"]=exact.bytes+fast.stored
        results.append(comparison)
        print(json.dumps(comparison),flush=True)
    (HERE/"results.json").write_text(json.dumps(dict(protocol=protocol,rows=results),indent=2)+"\n")


if __name__ == "__main__":
    main()
