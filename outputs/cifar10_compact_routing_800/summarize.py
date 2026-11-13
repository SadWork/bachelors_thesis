"""Audit the continuation and compare the paired 400- and 800-epoch metrics."""
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

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent.parent / "work" / "mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT = "62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"
STYLES = {"narrow_dense": ("Narrow Dense", "#52627b"),
          "factor_dense": ("Factorized Dense", "#7a578a"),
          "mixture_post": ("Mixture · BiReLU after sum", "#b97425"),
          "mixture_pre": ("Mixture · BiReLU before sum", "#12867b"),
          "logit_pre": ("Factorized logits · BiReLU before sum", "#27649b")}


def read(path):
    return json.loads(path.read_text())


def validation(run, start=1, stop=800):
    return [entry for entry in run["history"]
            if start <= entry["epoch"] <= stop and "validation" in entry]


def plot(rows, runs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained")
    for method, (label, color) in STYLES.items():
        series = [validation(runs[method, 8, seed]) for seed in range(3)]
        epochs = [e["epoch"] for e in series[0]]
        assert all([e["epoch"] for e in s] == epochs for s in series)
        for ax, metric, scale, ylabel in ((axes[0, 0], "ce", 1, "Validation cross-entropy"),
                                         (axes[0, 1], "accuracy", 100, "Validation accuracy (%)")):
            values = np.array([[e["validation"][metric]*scale for e in s] for s in series])
            ax.plot(epochs, values.mean(0), label=label, color=color)
            ax.fill_between(epochs, values.mean(0)-values.std(0, ddof=1),
                            values.mean(0)+values.std(0, ddof=1), color=color, alpha=.13)
            ax.set(xlabel="Epoch", ylabel=ylabel, title="Rank 8 · mean ± SD across 3 seeds")
        seed0 = sorted((r for r in rows if r["method"] == method and r["seed"] == 0), key=lambda r:r["rank"])
        for ax, metric, ylabel in ((axes[1, 0], "val_ce_800", "Validation CE at 800"),
                                   (axes[1, 1], "test_accuracy_best_800_pct", "Test accuracy (%)")):
            ax.plot([r["parameters"] for r in seed0], [r[metric] for r in seed0],
                    marker="o", label=label, color=color)
            ax.set(xscale="log", xlabel="Trainable parameters", ylabel=ylabel)
    for ax in axes[0]:
        ax.axvline(400, color="#626770", linestyle="--", alpha=.55)
    for ax in axes[1]:
        ax.set_xticks([20000, 36000, 70000, 137000], ["20k", "36k", "70k", "137k"])
        ax.set_xticks([], minor=True)
    axes[1, 0].set_title("Seed 0 · fixed epoch 800")
    axes[1, 1].set_title("Seed 0 · selected by validation accuracy through 800")
    axes[0, 1].legend(fontsize=7, loc="lower right")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("CIFAR-10 · Same AdamW state resumed at epoch 400")
    fig.savefig(ROOT / "learning_curves.png", dpi=170)
    plt.close(fig)


def close(actual, expected):
    return abs(actual["accuracy"]-expected["accuracy"]) <= .00021 and abs(actual["ce"]-expected["ce"]) <= .0001


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    args=p.parse_args()
    payload=read(ROOT/"colab_payload.json")["sources"]
    assert hashlib.sha256((ROOT/"colab_sources.tar.gz").read_bytes()).hexdigest()==payload["archive_sha256"]
    for name,digest in payload["files"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    previous=ROOT.parent/"cifar10_compact_routing"/"results"
    partitions=(("mac",args.work/"mac","mac"),
                ("mac_high",args.work/"mac_high","colab_friend1"),
                ("colab_friend1",args.work/"workers"/"friend1","colab_friend1"),
                ("colab_friend3",args.work/"workers"/"friend3","colab_friend3"),
                ("mac_recovery",args.work/"mac_recovery","colab_friend3"))
    rows,runs=[],{}
    for backend,folder,prior in partitions:
        before_root=previous/prior
        config,status=read(folder/"config.json"),read(folder/"status.json")
        old_cfg=read(before_root/"config.json")
        assert config["epochs"]==800
        assert set(config["ranks"]) <= set(old_cfg["ranks"])
        assert set(config["methods"]) <= set(old_cfg["methods"])
        assert set(config["seeds"]) <= set(old_cfg["seeds"])
        for key in ("learning_rate","decay","batch"):
            assert config[key]==old_cfg[key]
        assert read(folder/"data_manifest.json")["split_sha256"]==read(before_root/"data_manifest.json")["split_sha256"]==SPLIT
        expected={f"{m}_r{r}_s{s}" for m in config["methods"] for r in config["ranks"] for s in config["seeds"]}
        if backend=="colab_friend3":
            assert set(status["completed"]) <= expected
            expected=set(status["completed"])
        else:
            assert status["phase"]=="complete" and set(status["completed"])==expected
        dest=ROOT/"results"/backend
        dest.mkdir(parents=True,exist_ok=True)
        for item in ("config.json","data_manifest.json","split.npz","status.json"):
            shutil.copyfile(folder/item,dest/item)
        for name in sorted(expected):
            before=read(before_root/"runs"/f"{name}.json")
            run=read(folder/"runs"/f"{name}.json")
            for key in ("name","rank","seed","method","parameters","width"):
                assert before[key]==run[key]
            assert run["start_epoch"]==before["epochs"]==400 and run["epochs"]==800
            assert close(run["start_validation"],before["final"]["validation"])
            assert run["history"][:400]==before["history"]
            assert [e["epoch"] for e in run["history"]]==list(range(1,801))
            assert [e["epoch"] for e in validation(run,401)]==list(range(405,801,5))
            best=max(validation(run),key=lambda e:e["validation"]["accuracy"])
            assert best["epoch"]==run["best_epoch"] and close(best["validation"],run["best"]["validation"])
            best_ce=min(validation(run,401),key=lambda e:e["validation"]["ce"])
            assert best_ce["epoch"]==run["continuation_best_ce_epoch"]
            assert best_ce["validation"]==run["continuation_best_ce"]["validation"]
            assert validation(run)[-1]["validation"]==run["final"]["validation"]
            key=run["method"],run["rank"],run["seed"]
            assert key not in runs
            runs[key]=run
            row={k:run[k] for k in ("method","rank","seed","parameters","width")}
            row.update(device_400=old_cfg["device"],device_800=config["device"],best_epoch_400=before["best_epoch"],
                       best_epoch_800=run["best_epoch"],best_ce_epoch_after_400=best_ce["epoch"])
            for epoch,result in ((400,before),(800,run)):
                for split,short in (("train","train"),("validation","val"),("test","test")):
                    row[f"{short}_accuracy_{epoch}_pct"]=100*result["final"][split]["accuracy"]
                    row[f"{short}_ce_{epoch}"]=result["final"][split]["ce"]
                row[f"test_accuracy_best_{epoch}_pct"]=100*result["best"]["test"]["accuracy"]
                row[f"test_ce_best_{epoch}"]=result["best"]["test"]["ce"]
            row["val_ce_min_through_400"]=min(e["validation"]["ce"] for e in validation(run,1,400))
            row["val_ce_min_after_400"]=best_ce["validation"]["ce"]
            row["val_ce_change"]=row["val_ce_800"]-row["val_ce_400"]
            row["test_accuracy_at_best_ce_after_400_pct"]=100*run["continuation_best_ce"]["test"]["accuracy"]
            rows.append(row)
            for item in (f"runs/{name}.json",f"checkpoints/{name}.pt"):
                (dest/item).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(folder/item,dest/item)
    assert set(runs)=={(m,r,0) for m in STYLES for r in (4,8,16,32)} | {(m,8,s) for m in STYLES for s in (1,2)}
    rows.sort(key=lambda r:(r["rank"],r["seed"],list(STYLES).index(r["method"])))
    with (ROOT/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rank8={}
    for method in STYLES:
        subset=[r for r in rows if r["rank"]==8 and r["method"]==method]
        rank8[method]={k:{"mean":mean(r[k] for r in subset),"sd":stdev(r[k] for r in subset)}
                      for k in ("val_ce_400","val_ce_800","val_ce_change","val_accuracy_400_pct","val_accuracy_800_pct",
                                "test_accuracy_400_pct","test_accuracy_800_pct",
                                "test_accuracy_best_400_pct","test_accuracy_best_800_pct",
                                "test_ce_best_800","val_ce_min_through_400","val_ce_min_after_400")}
    (ROOT/"summary.json").write_text(json.dumps(dict(rows=rows,rank8=rank8),indent=2)+"\n")
    plot(rows,runs)
    with zipfile.ZipFile(ROOT/"reproducibility.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.suffix not in {".zip",".pyc",".tmp",".partial"}:
                archive.write(path,path.relative_to(ROOT))


if __name__=="__main__":
    main()
