"""Audit the two Dense MLP continuations and summarize accuracy and CE."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent.parent / "work" / "mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT = "62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"
CONDITION = {"name": "dense", "hidden_decay": .01, "head_decay": 0., "kl": 0.}


def read(path):
    return json.loads(path.read_text())


def audit(before, after, job, config):
    start = before["epochs"]
    assert start == after["start_epoch"] == job["start_epoch"]
    assert before["seed"] == after["seed"] == config["seed"] == 0
    assert before["depth"] == after["depth"] == config["depth"]
    assert before["condition"] == after["condition"] == job["condition"] == CONDITION
    assert after["epochs"] == config["target_epoch"] == 2000
    assert before["validation_at_stop"] == job["validation_at_start"]
    assert abs(after["start_validation"]["ce"] - job["validation_at_start"]["ce"]) <= .0001
    assert abs(after["start_validation"]["accuracy"] - job["validation_at_start"]["accuracy"]) <= .00021
    assert [e["epoch"] for e in before["history"]] == list(range(1, start + 1))
    assert [e["epoch"] for e in after["history"]] == list(range(start + 1, 2001))
    validation = [e for e in after["history"] if "validation" in e]
    assert [e["epoch"] for e in validation] == list(range(start + 5, 2001, 5))
    for key, operation, field in (("best_accuracy", max, "accuracy"), ("best_ce", min, "ce")):
        best = operation(validation, key=lambda e: e["validation"][field])
        assert best["epoch"] == after[key]["epoch"] and best["validation"] == after[key]["validation"]
    assert after["last"]["epoch"] == 2000 and after["last"]["validation"] == validation[-1]["validation"]


def plot(groups):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout="constrained")
    for row, (depth, (before, after)) in zip(axes, groups.items()):
        history = before["history"] + after["history"]
        validation = [e for e in history if "validation" in e]
        for ax, key, scale, ylabel in zip(row, ("accuracy", "ce"), (100, 1), ("Accuracy (%)", "Cross-entropy")):
            ax.plot([e["epoch"] for e in history], [scale * e[f"train_{key}"] for e in history],
                    color="#25689d", alpha=.84, label="Train")
            ax.plot([e["epoch"] for e in validation], [scale * e["validation"][key] for e in validation],
                    color="#b16d24", label="Validation")
            ax.axvline(before["epochs"], color="#626770", alpha=.55, linestyle="--")
            ax.set(title=f"{depth} hidden layers", xlabel="Epoch", ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10 · Dense MLP · BiReLU · Width 160 · Seed 0\n"
                 "Dashed: resumed from saved AdamW state")
    fig.savefig(ROOT / "learning_curves.png", dpi=170)
    plt.close(fig)


def write_metrics(row, prefix, item):
    row[prefix + "_epoch"] = item["epoch"]
    for split in ("train", "validation", "test"):
        row[f"{prefix}_{split}_accuracy_pct"] = 100 * item[split]["accuracy"]
        row[f"{prefix}_{split}_ce"] = item[split]["ce"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path)
    args = parser.parse_args()
    rows, groups = [], {}
    for depth in (3, 4):
        name = f"dense_d{depth}_s0"
        folder = ROOT / "results" / f"depth{depth}"
        if args.workers:
            worker = args.workers / f"depth{depth}"
            assert read(worker / "status.json") == {"phase": "complete", "completed": [name]}
            folder.mkdir(parents=True, exist_ok=True)
            for item in ("config.json", "data_manifest.json", "status.json", "split.npz",
                         f"runs/{name}.json", f"checkpoints/{name}.pt"):
                shutil.copyfile(worker / item, folder / Path(item).name)
        config = read(folder / "config.json")
        spec = read(ROOT / "references" / f"depth{depth}_jobs.json")
        job, = spec["jobs"]
        assert config["jobs"] == [job] and config["depth"] == depth
        assert read(folder / "data_manifest.json")["split_sha256"] == SPLIT == spec["split_sha256"]
        with (ROOT / "references" / f"{name}_start.pt").open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == job["checkpoint_sha256"]
        before = read(ROOT / "references" / f"{name}_previous.json")
        after = read(folder / f"{name}.json")
        audit(before, after, job, config)
        groups[depth] = before, after
        row = dict(depth=depth, seed=0, gpu=config["gpu"], parameters=before["parameters"],
                   start_epoch=before["epochs"], target_epoch=after["epochs"])
        for split in ("train", "validation"):
            row[f"start_{split}_accuracy_pct"] = 100 * before[f"{split}_at_stop"]["accuracy"]
            row[f"start_{split}_ce"] = before[f"{split}_at_stop"]["ce"]
        old_best = {"epoch": before["best_epoch"], "train": before["train_at_best"],
                    "validation": before["validation_at_best"], "test": before["test_at_best"]}
        write_metrics(row, "prior_best_accuracy", old_best)
        write_metrics(row, "continuation_best_accuracy", after["best_accuracy"])
        write_metrics(row, "continuation_best_ce", after["best_ce"])
        write_metrics(row, "at_2000", after["last"])
        overall = max((old_best, after["best_accuracy"]), key=lambda e: e["validation"]["accuracy"])
        write_metrics(row, "overall_best_accuracy", overall)
        all_entries = [e for e in before["history"] + after["history"] if "validation" in e]
        minimum = min(all_entries, key=lambda e: e["validation"]["ce"])
        row["overall_best_ce_epoch"] = minimum["epoch"]
        row["overall_best_validation_ce"] = minimum["validation"]["ce"]
        rows.append(row)
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=2))
    plot(groups)
    with zipfile.ZipFile(ROOT / "reproducibility.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.suffix not in {".zip", ".pyc", ".tmp", ".partial"}:
                archive.write(path, path.relative_to(ROOT))


if __name__ == "__main__":
    main()
