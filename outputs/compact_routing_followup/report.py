"""Summarize fixed compiled configurations, the ridge test, and the training pilot."""
import csv
import json
import os
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parents[1] / "work" / "mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

LABELS = {"dense": "Dense", "cached_full": "Exact cache", "k64_first": "Global tail · K64",
          "k128_first": "Global tail · K128", "tiled64_first": "Tile tail · K64",
          "tiled96_first": "Tile tail · K96"}


def read(path):
    return json.loads((HERE / path).read_text())


def csv_write(path, rows):
    with (HERE / path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    test, validation, timing = (read(name + ".json") for name in ("test", "validation", "timing"))
    test_base = [row["test"]["accuracy"] for row in test if row["method"] == "cached_full"]
    validation_base = [row["validation"]["accuracy"] for row in validation if row["method"] == "cached_full"]
    rows = []
    for name, label in LABELS.items():
        a = sorted((r for r in test if r["method"] == name), key=lambda r: r["seed"])
        b = sorted((r for r in validation if r["method"] == name), key=lambda r: r["seed"])
        assert [r["seed"] for r in a] == [r["seed"] for r in b] == [0, 1, 2]
        assert all(r["resident_bytes"] == a[0]["resident_bytes"] for r in a + b)
        difference = [100*(row["test"]["accuracy"] - base) for row, base in zip(a, test_base)]
        rows.append(dict(method=name, label=label, stored_bytes=a[0]["resident_bytes"],
                         test_accuracy_pct=100*mean(r["test"]["accuracy"] for r in a),
                         test_ce=mean(r["test"]["ce"] for r in a), paired_test_delta_pp=mean(difference),
                         seed0_delta_pp=difference[0], seed1_delta_pp=difference[1], seed2_delta_pp=difference[2],
                         validation_accuracy_pct=100*mean(r["validation"]["accuracy"] for r in b),
                         paired_validation_delta_pp=100*mean(r["validation"]["accuracy"]-v
                                                             for r, v in zip(b, validation_base)),
                         cpu_batch1_ms=timing["1"].get(name), cpu_batch128_ms=timing["128"].get(name)))
    csv_write("compiled_summary.csv", rows)
    pilot = []
    for dataset in ("cifar10", "fashion"):
        for file in sorted((HERE / "pilot_runs" / dataset).glob("*_s0.json")):
            r = json.loads(file.read_text())
            best = max((e for e in r["history"] if "validation" in e), key=lambda e: e["validation"]["accuracy"])
            assert best["epoch"] == r["best_epoch"] and r["validation"] == best["validation"]
            assert r["epochs"] == r["history"][-1]["epoch"] == 300
            pilot.append(dict(dataset=dataset, method=r["method"], seed=r["seed"], parameters=r["parameters"],
                              best_epoch=r["best_epoch"], validation_accuracy_pct=100*r["validation"]["accuracy"],
                              validation_ce=r["validation"]["ce"]))
    assert len(pilot) == 8
    csv_write("pilot_summary.csv", pilot)
    ridge = read("sparse_probe/frozen_test_results.json")
    assert ridge["protocol"]["ks"] == [128,64,128,64] and not ridge["protocol"]["test_retuning"]
    assert len(ridge["rows"]) == 9

    fig, (accuracy, latency) = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    offsets = {"dense": (-8, 13), "cached_full": (-8, -13), "k64_first": (8, -6),
               "k128_first": (6, -10), "tiled64_first": (14, -4), "tiled96_first": (7, 13)}
    for row in rows:
        name = row["method"]
        color = "#12867b" if name.startswith("tiled") else ("#a35c24" if name.startswith("k") else "#52627b")
        accuracy.scatter(row["stored_bytes"]/1e6, row["test_accuracy_pct"], color=color, s=38)
        accuracy.annotate(row["label"], (row["stored_bytes"]/1e6, row["test_accuracy_pct"]),
                          xytext=offsets[name], textcoords="offset points",
                          ha="right" if name in ("dense", "cached_full") else "left", fontsize=8)
    accuracy.set(xscale="log", xlim=(.33, 2.7), ylim=(47.9, 51.25),
                 xlabel="Stored inference buffers (MB)", ylabel="CIFAR-10 test accuracy (%)")
    accuracy.set_xticks((.4, .6, 1., 2.), ("0.4", "0.6", "1", "2"))
    accuracy.xaxis.set_minor_formatter(NullFormatter())
    accuracy.grid(alpha=.2)
    names = ("cached_full", "k128_first", "tiled64_first", "tiled96_first")
    for i, batch in enumerate(("1", "128")):
        values = [timing[batch][name]/timing[batch]["dense"] for name in names]
        latency.bar([j+(.17 if i else -.17) for j in range(len(names))], values, width=.32,
                    color=("#25689d", "#12867b")[i], label=f"Batch {batch}")
    latency.axhline(1, color="#626770", linestyle="--", linewidth=1)
    latency.set(yscale="log", ylim=(.8, 50), ylabel="CPU time relative to Dense (×)")
    latency.set_xticks(range(len(names)), [LABELS[name].replace(" · ", "\n") for name in names], fontsize=8)
    latency.grid(axis="y", alpha=.2)
    latency.legend(fontsize=8)
    fig.suptitle("Static routing · CIFAR-10 test over three seeds · Apple M3 Pro CPU timing at seed 0")
    fig.savefig(HERE / "memory_accuracy_latency.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
