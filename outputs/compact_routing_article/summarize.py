"""Aggregate the fixed experiments and plot the observed storage/accuracy trade-off."""
import csv
import json
import os
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CACHED_STORAGE = 2_287_440
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parents[1] / "work" / "mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def read(path):
    return json.loads((HERE / path).read_text())


def record(rows, dataset, name, label, source, size, mode, probability=True):
    rows = sorted(rows, key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == [0,1,2]
    accuracy = [100*r["test"]["accuracy"] if probability else r["accuracy_pct"] for r in rows]
    ce = [r["test"]["ce"] if probability else r["ce"] for r in rows]
    return dict(dataset=dataset,method=name,label=label,source=source,bytes=size,
                storage_description=mode,accuracy_pct=mean(accuracy),ce=mean(ce),
                seed0_pct=accuracy[0],seed1_pct=accuracy[1],seed2_pct=accuracy[2])


def selected(rows, key, method):
    return [row for row in rows if row[key] == method]


def table():
    results=[]
    c10=read("support/test_results.json")["rows"]
    results.append(record(selected(c10,"method","teacher"),"cifar10","teacher","Exact route cache",
                          "support/test_results.json",CACHED_STORAGE,"FP32 cache",False))
    results.append(record(selected(c10,"method","topk_refit_ridge"),"cifar10","refit","Sparse tail + refit",
                          "support/test_results.json",137000,"actual packed payload excluding JSON",False))
    quant=read("quantization/results.json")
    for method,label in (("dense_fp32","Full Dense FP32"),("dense_linear8","Full Dense INT8")):
        rr=selected(quant,"method",method)
        results.append(record(rr,"cifar10",method,label,"quantization/results.json",
                              rr[0]["packed_bytes_estimate"],"weight storage estimate"))
    calibrated=read("quantization/calibrated_matched/results.json")["rows"]
    for method,label in (("dense_linear4_ridge","Full Dense INT4 + ridge"),
                         ("route_codebook2_ridge","Route 2-bit codebook + ridge"),
                         ("route_codebook4_ridge","Route 4-bit codebook + ridge")):
        rr=selected(calibrated,"method",method)
        results.append(record(rr,"cifar10",method,label,"quantization/calibrated_matched/results.json",
                              round(rr[0]["packed_bytes_estimate"]),"weight storage estimate"))
    fashion=[r for r in read("support/fashion_transfer_results.json")["rows"] if r["split"]=="test"]
    for method,name,label in (("teacher","teacher","Exact route cache"),("topk_refit_ridge","refit","Sparse tail + refit")):
        rr=selected(fashion,"method",method)
        results.append(record(rr,"fashion",name,label,"support/fashion_transfer_results.json",
                              rr[0]["resident_bytes_with_affine_and_head"],"format size estimate",False))
    independent=read("independent/eval_frozen/results.json")["rows"]
    for method,name,label in (("teacher","teacher","Exact route cache"),("focus32_topk_refit_ridge","refit","Sparse tail + refit"),
                              ("full_dense","dense_fp32","Full Dense FP32"),("factor_dense","factor_fp32","Factor Dense r8 FP32"),
                              ("dense_linear4","dense_linear4","Full Dense INT4"),("dense_linear8","dense_linear8","Full Dense INT8")):
        rr=selected(independent,"method",method)
        results.append(record(rr,"cifar100",name,label,"independent/eval_frozen/results.json",
                              rr[0]["packed_bytes_estimate"],"format size estimate"))
    for dataset in ("cifar10","cifar100"):
        path=f"matched_dense/eval_{dataset}/results.json"
        rr=read(path)["rows"]
        for method,label in (("factor_int8","Factor Dense INT8"),("factor_int8_ridge","Factor Dense INT8 + ridge")):
            subset=selected(rr,"method",method)
            results.append(record(subset,dataset,method,label,path,subset[0]["packed_bytes_estimate"],
                                  "weight storage estimate"))
    return results


def plot(rows):
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    for ax,dataset in zip(axes[:2],("cifar10","cifar100")):
        methods=("teacher","refit","dense_fp32","factor_int8","factor_int8_ridge")
        group={r["method"]:r for r in rows if r["dataset"]==dataset and r["method"] in methods}
        for name,row in group.items():
            color="#12867b" if name=="refit" else "#52627b" if name in ("teacher","dense_fp32") else "#b67425"
            ax.scatter(row["bytes"]/1000,row["accuracy_pct"],color=color,s=40)
        ax.set(xscale="log",xlabel="Packed or cached bytes (kB)",ylabel="Test accuracy (%)",
               title="CIFAR-10" if dataset=="cifar10" else "CIFAR-100",xlim=(112,2900))
        ax.set_xticks((137,200,500,1000,2000),("137","200","500","1000","2000"),fontsize=8)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
    axes[0].scatter([],[],color="#12867b",label="Sparse tail + refit")
    axes[0].scatter([],[],color="#b67425",label="Matched INT8 factor Dense")
    axes[0].scatter([],[],color="#52627b",label="Exact cache / full Dense")
    axes[0].legend(fontsize=7)
    systems=read("systems/timing_final.json")
    names=("dense_fp32","cached_exact","tile96","refit_fp32_buckets","refit_fp16_buckets")
    labels=("Dense FP32","Exact cache","Tile96","Refit FP32","Refit FP16")
    for name,label in zip(names,labels):
        mem=systems["memory"]["normalized_fp32"]["1"].get(name,
                {"stored_bytes":CACHED_STORAGE,"explicit_scratch_bytes":1320})
        footprint=(mem["stored_bytes"]+mem["explicit_scratch_bytes"])/1000
        time=systems["timing"]["normalized_fp32"]["1"][name]["median_ms"]*1000
        axes[2].scatter(footprint,time,color="#12867b" if "refit" in name else "#52627b",s=40)
        offsets={"refit_fp16_buckets":(-6,-13),"refit_fp32_buckets":(7,8),"tile96":(7,-11),
                 "dense_fp32":(-7,-12),"cached_exact":(-7,8)}
        axes[2].annotate(label,(footprint,time),xytext=offsets[name],textcoords="offset points",fontsize=8,
                         ha="right" if name in ("dense_fp32","cached_exact") else "left")
    axes[2].set(xscale="log",xlabel="Stored + explicit scratch (kB)",ylabel="CPU latency (μs)",
                title="CIFAR-10, batch 1",xlim=(120,2900),ylim=(8,69))
    axes[2].set_xticks((160,220,500,1000,2000),("160","220","500","1000","2000"),fontsize=8)
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].grid(alpha=.2)
    fig.suptitle("Static routing: accuracy and storage (3 seeds) · CPU execution (seed 0)")
    fig.savefig(HERE/"evidence.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    rows=table()
    with (HERE/"evidence.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
