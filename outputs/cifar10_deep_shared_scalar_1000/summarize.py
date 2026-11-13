"""Verify four fixed-horizon continuations and compare validation metrics."""
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
STYLES = {"no_kl": ("No KL", "#25689d"), "both_10": ("KL 0.10", "#12867b")}


def read(path):
    return json.loads(path.read_text())


def audit(before, after, job):
    assert before["epochs"] == after["start_epoch"] == job["start_epoch"]
    assert before["depth"] == after["depth"] and before["seed"] == after["seed"] == 0
    assert before["condition"] == after["condition"] == job["condition"]
    assert before["validation_at_stop"] == job["validation_at_start"]
    assert after["epochs"] == 1000
    start = after["start_epoch"]
    assert [e["epoch"] for e in before["history"]] == list(range(1, start + 1))
    assert [e["epoch"] for e in after["history"]] == list(range(start + 1, 1001))
    val = [e for e in after["history"] if "validation" in e]
    assert [e["epoch"] for e in val] == list(range(start + 5, 1001, 5))
    for key, operation, metric in (("best_ce", min, "ce"), ("best_accuracy", max, "accuracy")):
        best = operation(val, key=lambda e: e["validation"][metric])
        assert after[key]["epoch"] == best["epoch"] and after[key]["validation"] == best["validation"]
    assert after["last"]["epoch"] == 1000 and after["last"]["validation"] == val[-1]["validation"]


def plot(groups):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for row, (depth, runs) in zip(axes, groups.items()):
        for kind, (before, after) in runs.items():
            label, color = STYLES[kind]
            entries = [entry for entry in before["history"] + after["history"]
                       if entry["epoch"] >= 140 and "validation" in entry]
            for ax, field, scale in ((row[0], "accuracy", 100), (row[1], "ce", 1)):
                ax.plot([e["epoch"] for e in entries], [scale * e["validation"][field] for e in entries],
                        color=color, label=label)
                ax.axvline(after["start_epoch"], color=color, alpha=.42, linestyle="--")
            chosen = after["best_ce"]
            row[1].scatter(chosen["epoch"], chosen["validation"]["ce"], color=color, zorder=3)
        for ax, label in zip(row, ("Validation accuracy (%)", "Validation cross-entropy")):
            ax.set(title=f"{depth} hidden layers", xlabel="Epoch", ylabel=label)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10 · Shared scalar · Seed 0 · 1000 total epochs\n"
                 "Dashed: original stop · Dot: best validation CE during continuation")
    fig.savefig(ROOT / "learning_curves.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path)
    args = parser.parse_args()
    manifest = read(ROOT / "colab_payload.json")
    assert hashlib.sha256((ROOT / "colab_sources.tar.gz").read_bytes()).hexdigest() == manifest["archive_sha256"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    rows, groups = [], {}
    for profile, depth in (("friend1", 3), ("friend2", 4)):
        folder = ROOT / "results" / f"depth{depth}"
        names = {kind: f"{kind}_d{depth}_s0" for kind in STYLES}
        if args.workers:
            worker = args.workers / profile
            status = read(worker / "status.json")
            assert status["phase"] == "complete" and set(status["completed"]) == set(names.values())
            folder.mkdir(parents=True, exist_ok=True)
            files = ["config.json", "data_manifest.json", "split.npz", "status.json"]
            files.extend(f"{kind}/{name}{ext}" for name in names.values()
                         for kind, ext in (("runs", ".json"), ("checkpoints", ".pt")))
            for item in files:
                shutil.copyfile(worker / item, folder / Path(item).name)
        config = read(folder / "config.json")
        spec = read(ROOT / "references" / f"depth{depth}_jobs.json")
        assert config["depth"] == depth and config["seed"] == 0 and config["target_epoch"] == 1000
        assert config["jobs"] == spec["jobs"]
        if config["device"] == "cuda":
            assert config["gpu"] == spec["gpu"] and config["torch"] == spec["torch"]
        assert read(folder / "data_manifest.json")["split_sha256"] == SPLIT == spec["split_sha256"]
        groups[depth] = {}
        for job in spec["jobs"]:
            kind = job["condition"]["name"]
            name = names[kind]
            with (ROOT / "references" / f"{name}_start.pt").open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == job["checkpoint_sha256"]
            before = read(ROOT / "references" / f"{name}_previous.json")
            after = read(folder / f"{name}.json")
            audit(before, after, job)
            expected = job["validation_at_start"]
            actual = after["start_validation"]
            assert abs(actual["ce"] - expected["ce"]) <= .0001
            assert abs(actual["accuracy"] - expected["accuracy"]) <= .00021
            groups[depth][kind] = before, after
            prior_ce = min((e for e in before["history"] if "validation" in e),
                           key=lambda e: e["validation"]["ce"])
            row = dict(job["condition"], depth=depth, seed=0, gpu=config["gpu"], start_epoch=after["start_epoch"],
                       epochs=1000, train_ce_start=before["train_at_stop"]["ce"],
                       train_ce_1000=after["last"]["train"]["ce"],
                       train_accuracy_start_pct=100*before["train_at_stop"]["accuracy"],
                       train_accuracy_1000_pct=100*after["last"]["train"]["accuracy"],
                       validation_ce_start=before["validation_at_stop"]["ce"],
                       validation_accuracy_start_pct=100*before["validation_at_stop"]["accuracy"],
                       prior_best_ce_epoch=prior_ce["epoch"], prior_best_validation_ce=prior_ce["validation"]["ce"],
                       prior_best_accuracy_epoch=before["best_epoch"],
                       prior_best_validation_accuracy_pct=100*before["validation_at_best"]["accuracy"],
                       prior_best_test_accuracy_pct=100*before["test_at_best"]["accuracy"])
            for label, record in (("best_ce_after", after["best_ce"]), ("best_accuracy_after", after["best_accuracy"]),
                                  ("at_1000", after["last"])):
                row[f"{label}_epoch"] = record["epoch"]
                row[f"{label}_validation_ce"] = record["validation"]["ce"]
                row[f"{label}_validation_accuracy_pct"] = 100 * record["validation"]["accuracy"]
                row[f"{label}_test_accuracy_pct"] = 100 * record["test"]["accuracy"]
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
