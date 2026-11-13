"""Audit the matched three-seed placement control at epochs 800 and 1200."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT.parent.parent/"work"/"mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT="62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"
STYLES={"logit_pre":("BiReLU before sum","#12867b"),"logit_post":("BiReLU after sum","#b97425")}


def read(path):
    return json.loads(path.read_text())


def plot(runs):
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    for method,(label,color) in STYLES.items():
        history=[[e for e in runs[method,seed]["history"] if "validation" in e] for seed in range(3)]
        epochs=[e["epoch"] for e in history[0]]
        assert all([e["epoch"] for e in h]==epochs for h in history)
        for ax,metric,scale,ylabel in ((axes[0],"ce",1,"Validation cross-entropy"),
                                      (axes[1],"accuracy",100,"Validation accuracy (%)")):
            values=np.array([[scale*e["validation"][metric] for e in h] for h in history])
            ax.plot(epochs,values.mean(0),color=color,label=label)
            ax.fill_between(epochs,values.mean(0)-values.std(0,ddof=1),
                            values.mean(0)+values.std(0,ddof=1),color=color,alpha=.16)
            ax.set(xlabel="Epoch",ylabel=ylabel)
    for ax in axes:
        ax.axvline(800,color="#656b70",linestyle="--",alpha=.6)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("CIFAR-10 · Rank 8 · Identical factorized logits and initialization\n"
                 "Mean ± SD across 3 seeds")
    fig.savefig(ROOT/"learning_curves.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    args=p.parse_args()
    manifest=read(ROOT/"colab_payload.json")
    assert hashlib.sha256((ROOT/"colab_sources.tar.gz").read_bytes()).hexdigest()==manifest["archive_sha256"]
    for name,sha in manifest["files"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    rows,runs=[],{}
    prior=ROOT.parent/"cifar10_compact_routing_800"/"results"
    pre_sources=("mac","colab_friend3","mac_recovery")
    for method in STYLES:
        position=method.split("_")[-1]
        for seed in range(3):
            name=f"{method}_r8_s{seed}"
            if seed==0:
                source=args.work/f"mac_{position}"
            elif seed==1:
                fallback=args.work/f"mac_{position}_s1"
                source=fallback if fallback.exists() else args.work/"workers"/("friend1" if position=="pre" else "friend2")
            else:
                source=args.work/f"mac_{position}_s2"
            config,status=read(source/"config.json"),read(source/"status.json")
            assert status=={"phase":"complete","completed":[name]}
            assert config["methods"]==[method] and config["ranks"]==[8] and config["seeds"]==[seed]
            assert read(source/"data_manifest.json")["split_sha256"]==SPLIT
            run=read(source/"runs"/f"{name}.json")
            assert run["epochs"]==1200 and run["seed"]==seed and run["method"]==method and run["parameters"]==36426
            assert [e["epoch"] for e in run["history"]]==list(range(1,1201))
            if method=="logit_pre":
                old=read(prior/pre_sources[seed]/"runs"/f"{name}.json")
                assert old["history"]==run["history"][:800]
                assert all(run["at_800"][key]==old[key] for key in ("best_epoch","best","final"))
            runs[method,seed]=run
            row=dict(method=method,seed=seed,parameters=36426,device=config["device"])
            for epoch in (800,1200):
                val=[e for e in run["history"] if e["epoch"]<=epoch and "validation" in e]
                item=run[f"at_{epoch}"]
                selected=max(val,key=lambda e:e["validation"]["accuracy"])
                assert item["epoch"]==epoch and item["best_epoch"]==selected["epoch"]
                assert abs(item["best"]["validation"]["ce"]-selected["validation"]["ce"])<=.0001
                assert abs(item["best"]["validation"]["accuracy"]-selected["validation"]["accuracy"])<=.00021
                assert item["final"]["validation"]==val[-1]["validation"]
                row[f"best_epoch_through_{epoch}"]=item["best_epoch"]
                row[f"val_ce_min_through_{epoch}"]=min(e["validation"]["ce"] for e in val)
                for split in ("train","validation","test"):
                    for stage in ("best","final"):
                        metrics=item[stage][split]
                        row[f"{stage}_{split}_accuracy_{epoch}_pct"]=100*metrics["accuracy"]
                        row[f"{stage}_{split}_ce_{epoch}"]=metrics["ce"]
            rows.append(row)
            dest=ROOT/"results"/name
            dest.mkdir(parents=True,exist_ok=True)
            for item in ("config.json","data_manifest.json","split.npz","status.json",
                         f"runs/{name}.json",f"checkpoints/{name}.pt"):
                shutil.copyfile(source/item,dest/Path(item).name)
    with (ROOT/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric=[k for k in rows[0] if k not in ("method","seed","device")]
    stats={}
    for method in STYLES:
        group=[r for r in rows if r["method"]==method]
        stats[method]={k:{"mean":mean(r[k] for r in group),"sd":stdev(r[k] for r in group)}
                       for k in numeric}
    lookup={(r["method"],r["seed"]):r for r in rows}
    deltas={k:[lookup["logit_pre",s][k]-lookup["logit_post",s][k] for s in range(3)]
            for k in numeric}
    (ROOT/"summary.json").write_text(json.dumps(dict(rows=rows,statistics=stats,pre_minus_post=deltas),indent=2)+"\n")
    plot(runs)
    with zipfile.ZipFile(ROOT/"reproducibility.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.suffix not in {".zip",".pyc",".tmp",".partial"}:
                archive.write(path,path.relative_to(ROOT))


if __name__=="__main__":
    main()
