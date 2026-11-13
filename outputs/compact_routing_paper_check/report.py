"""Summarize the paper check and generate paired comparison plots."""
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT.parent.parent/"work"/"mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS={"full_dense":"Full Dense","factor_dense":"Factorized Dense",
        "logit_pre":"BiReLU before sum","logit_post":"BiReLU after sum"}
COLORS={"full_dense":"#52627b","factor_dense":"#725585",
        "logit_pre":"#12867b","logit_post":"#b97425"}


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assessment=json.loads((ROOT/"assessment.json").read_text())
    rows=[row for item in assessment.values() for row in item["rows"]]
    write_csv(ROOT/"metrics.csv",rows)
    paired,summary=[],{}
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout="constrained")
    for axis_row,(dataset,item) in zip(axes,assessment.items()):
        groups={}
        for row in item["rows"]:
            groups.setdefault(row["method"],[]).append(row)
        name="CIFAR-10" if dataset=="cifar10" else "Fashion-MNIST"
        summary[dataset]={}
        for index,(method,group) in enumerate(groups.items()):
            assert len(group)==3
            stats={key:{"mean":mean(r[key] for r in group),"sd":stdev(r[key] for r in group)}
                   for key in ("test_accuracy","test_ce","calibrated_test_ce")}
            stats["parameters"]=group[0]["parameters"]
            summary[dataset][method]=stats
            color=COLORS[method]
            axis_row[0].errorbar(stats["parameters"],stats["test_accuracy"]["mean"],
                                 yerr=stats["test_accuracy"]["sd"],fmt="o",capsize=4,
                                 color=color,label=LABELS[method])
            axis_row[1].scatter(index-.10,stats["test_ce"]["mean"],color=color,marker="o")
            axis_row[1].scatter(index+.10,stats["calibrated_test_ce"]["mean"],color=color,marker="s")
            axis_row[1].plot([index-.10,index+.10],
                             [stats["test_ce"]["mean"],stats["calibrated_test_ce"]["mean"]],
                             color=color,alpha=.6)
        axis_row[0].set(xscale="log",xlabel="Trainable parameters",ylabel="Test accuracy (%)",
                        title=f"{name} · mean ± SD, 3 seeds")
        ticks = [36000, 100000, 571000] if dataset == "cifar10" else [18000, 50000, 204000]
        axis_row[0].set_xticks(ticks,[f"{round(t/1000)}k" for t in ticks])
        axis_row[0].set_xticks([],minor=True)
        axis_row[0].legend(fontsize=8)
        axis_row[1].set(title=f"{name} · circle: raw CE · square: calibrated CE",ylabel="Test cross-entropy")
        axis_row[1].set_xticks(np.arange(len(groups)),[LABELS[m] for m in groups],rotation=12,ha="right",fontsize=8)
        for ax in axis_row: ax.grid(alpha=.2)
        for control,result in item["paired_pre_minus_control"].items():
            paired.append(dict(dataset=dataset,control=control,mean_pp=result["mean_pp"],
                               low_95_pp=result["interval_95_pp"][0],high_95_pp=result["interval_95_pp"][1],
                               seed0_pp=result["by_seed_pp"][0],seed1_pp=result["by_seed_pp"][1],
                               seed2_pp=result["by_seed_pp"][2]))
    fig.suptitle("Compact static routing · paired paper check")
    fig.savefig(ROOT/"comparison.png",dpi=170)
    plt.close(fig)
    write_csv(ROOT/"paired_comparisons.csv",paired)
    (ROOT/"summary.json").write_text(json.dumps(dict(metrics=summary,paired=paired,
            cifar10_latency=assessment["cifar10"]["latency"]),indent=2)+"\n")
    with zipfile.ZipFile(ROOT/"reproducibility.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.suffix not in {".zip",".pyc",".tmp",".partial"}:
                archive.write(path,path.relative_to(ROOT))


if __name__=="__main__":
    main()
