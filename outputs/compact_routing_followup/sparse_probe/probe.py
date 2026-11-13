"""Validation-only diagnostics for sparse deployment of static scalar routing."""

import gzip
import hashlib
import json
from math import sqrt
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKPOINTS = ROOT / "outputs" / "compact_routing_paper_check" / "results"
WORK = ROOT / "work"
GRID = json.loads((HERE / "grid.json").read_text())["configurations"]


def validation_data(dataset, calibration_size=0):
    meta = json.loads((CHECKPOINTS / dataset / "logit_pre_r8_s0" / "data_manifest.json").read_text())
    if dataset == "cifar10":
        path = WORK / "cifar10_shared_scalar" / "data" / "cifar-10-python.tar.gz"
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == meta["archive_sha256"]
        with tarfile.open(path) as arc:
            batches = [pickle.load(arc.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="latin1")
                       for i in range(1, 6)]
        x = np.concatenate([batch["data"] for batch in batches])
        y = np.concatenate([batch["labels"] for batch in batches])
    else:
        folder = WORK / "compact_routing_paper_check" / "fashion"
        for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
            assert hashlib.md5((folder / name).read_bytes()).hexdigest() == meta["source_md5"][name]
        with gzip.open(folder / "train-images-idx3-ubyte.gz", "rb") as stream:
            x = np.frombuffer(stream.read()[16:], dtype=np.uint8).reshape(60000, 784)
        with gzip.open(folder / "train-labels-idx1-ubyte.gz", "rb") as stream:
            y = np.frombuffer(stream.read()[8:], dtype=np.uint8)
    rng = np.random.default_rng(meta["split_seed"])
    splits = [rng.permutation(np.flatnonzero(y == i)) for i in range(10)]
    train = np.concatenate([split[500:] for split in splits]).astype(np.int64)
    val = np.concatenate([split[:500] for split in splits]).astype(np.int64)
    assert hashlib.sha256(train.tobytes() + val.tobytes()).hexdigest() == meta["split_sha256"]
    channels = len(meta["channel_mean"])
    mean = torch.tensor(meta["channel_mean"]).view(1, -1, 1)
    std = torch.tensor(meta["channel_std"]).view(1, -1, 1)
    def normalized(indices):
        pixels = torch.from_numpy(x[indices].copy()).float().reshape(len(indices), channels, -1)
        pixels.div_(255).sub_(mean).div_(std)
        return pixels.flatten(1)
    validation = (normalized(val), torch.from_numpy(y[val].astype(np.int64)))
    if not calibration_size:
        return validation
    rng = np.random.default_rng(93187)
    calibration = rng.choice(train, calibration_size, replace=False)
    assert not np.intersect1d(val, calibration).size
    return *validation, normalized(calibration), hashlib.sha256(calibration.tobytes()).hexdigest()


def all_activations_sum(x, weight, bias):
    values = x.sort(1).values.contiguous()
    prefix = F.pad(values.cumsum(1), (1, 0))
    threshold = (-bias / torch.where(weight != 0, weight, 1)).expand(len(x), -1).contiguous()
    count_below = torch.searchsorted(values, threshold)
    below = prefix.gather(1, count_below)
    total = prefix[:, -1:]
    n = x.size(1)
    hinge = torch.where(weight > 0, weight * (total - below) + bias * (n - count_below),
                        weight * below + bias * count_below)
    hinge = torch.where(weight != 0, hinge, bias.clamp_min(0) * n)
    return .5 * (weight * total + bias * n) + 1.5 * hinge


def deploy_layers(state, ks, method):
    layers = []
    for layer in range(4):
        def get(key):
            return state[f"hidden.{layer}.{key}"]
        prob = (get("left") @ get("right") / sqrt(8)).softmax(-1)
        weight, bias = get("weight"), get("bias")
        if method == "full":
            layers.append((weight, bias, prob, None, None))
        else:
            selected, indices = prob.topk(ks[layer], -1)
            assert torch.equal(indices, indices.to(torch.uint16).long())
            tail = 1 - selected.sum(-1)
            if method == "renorm":
                selected = selected / selected.sum(-1, keepdim=True)
            # Round to the deployed representation, then accumulate in fp32.
            layers.append((weight, bias, selected.half().float(), indices, tail))
    return layers


@torch.no_grad()
def evaluate(x, target, state, ks, method):
    total_correct, total_ce = 0, 0.
    layers = deploy_layers(state, ks, method)
    for xx, yy in zip(x.split(128), target.split(128)):
        for weight, bias, p, indices, tail in layers:
            layer_input = xx
            z = (xx[:, None, :] if indices is None else xx[:, indices]) * weight[:, None] + bias[:, None]
            f = torch.where(z > 0, 2 * z, .5 * z)
            xx = (f * p).sum(-1)
            if method == "uniform_tail":
                average_tail = (all_activations_sum(layer_input, weight, bias) - f.sum(-1))
                xx += tail * average_tail / (layer_input.shape[1] - indices.shape[-1])
        logits = F.linear(xx, state["head.weight"], state["head.bias"])
        total_correct += (logits.argmax(1) == yy).sum().item()
        total_ce += F.cross_entropy(logits, yy, reduction="sum").item()
    return {"accuracy_pct": 100 * total_correct / len(x), "ce": total_ce / len(x)}


def budget(inputs, ks, method):
    common_bytes = 4 * (2 * 4 * 160 + (160 + 1) * 10)
    if method == "full":
        edges = 160 * (inputs + 3 * 160)
        return {"explicit_scalar_edges": edges, "routing_cache_bytes": 4 * edges,
                "resident_bytes_with_affine_and_head": 4 * edges + common_bytes}
    edges = 160 * sum(ks)
    cache_bytes = 4 * edges + (4 * 4 * 160 if method == "uniform_tail" else 0)
    return {"explicit_scalar_edges": edges, "routing_cache_bytes": cache_bytes,
            "resident_bytes_with_affine_and_head": cache_bytes + common_bytes}


def main():
    torch.set_num_threads(2)
    output = []
    for dataset in ("cifar10", "fashion"):
        x, y = validation_data(dataset)
        for seed in range(3):
            name = f"logit_pre_r8_s{seed}"
            directory = CHECKPOINTS / dataset / name
            checkpoint = torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
            state = checkpoint["best"]["state"]
            for label, ks in GRID.items():
                for method in ("full",) if ks is None else ("renorm", "uniform_tail"):
                    row = dict(dataset=dataset, seed=seed, configuration=label, ks=ks, method=method)
                    row.update(evaluate(x, y, state, ks, method), **budget(x.size(-1), ks, method))
                    if method == "full":
                        log = json.loads((directory / f"{name}.json").read_text())
                        record = log.get("at_1200", log)
                        assert abs(record["best"]["validation"]["accuracy"] * 100 - row["accuracy_pct"]) <= .021
                    print(json.dumps(row), flush=True)
                    output.append(row)
    (HERE / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
