"""Evaluate the single frozen sparse-tail experiment on CIFAR-10 test."""
import hashlib
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch
from torch.nn import functional as F

from calibrate import CALIBRATION_SIZE, representations, ridge_head
from probe import CHECKPOINTS, HERE, WORK, budget, deploy_layers, validation_data


PROTOCOL = json.loads((HERE / "frozen_test_protocol.json").read_text())


def original_test_set():
    meta = json.loads((CHECKPOINTS / "cifar10" / "logit_pre_r8_s0" / "data_manifest.json").read_text())
    path = WORK / "cifar10_shared_scalar" / "data" / "cifar-10-python.tar.gz"
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == meta["archive_sha256"]
    with tarfile.open(path) as archive:
        record = pickle.load(archive.extractfile("cifar-10-batches-py/test_batch"), encoding="latin1")
    x = torch.as_tensor(record["data"]).float().reshape(-1, 3, 1024)
    mean = torch.tensor(meta["channel_mean"]).reshape(1, 3, 1)
    sd = torch.tensor(meta["channel_std"]).reshape(1, 3, 1)
    x.div_(255).sub_(mean).div_(sd)
    return x.flatten(1), torch.tensor(record["labels"])


def bootstrap(difference):
    rng = np.random.default_rng(37201)
    means = np.empty(3000)
    for i in range(len(means)):
        seeds = rng.integers(0, difference.shape[0], difference.shape[0])
        images = rng.integers(0, difference.shape[1], difference.shape[1])
        means[i] = 100 * difference[seeds[:, None], images].mean()
    return {"mean_delta_pp": 100 * difference.mean(),
            "per_seed_delta_pp": (100 * difference.mean(1)).tolist(),
            "exploratory_interval_95_pp": np.quantile(means, [.025, .975]).tolist()}


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    assert PROTOCOL["ridge_calibration_size"] == CALIBRATION_SIZE
    _, _, cal, calibration_hash = validation_data("cifar10", calibration_size=CALIBRATION_SIZE)
    assert calibration_hash == PROTOCOL["ridge_calibration_index_hash"]
    x, target = original_test_set()
    ks = PROTOCOL["ks"]
    rows, correct = [], {}
    for seed in PROTOCOL["seeds"]:
        name = f"logit_pre_r8_s{seed}"
        c = torch.load(CHECKPOINTS / "cifar10" / name / f"{name}.pt", weights_only=True, map_location="cpu")
        state = c["best"]["state"]
        head = state["head.weight"], state["head.bias"]
        full = deploy_layers(state, None, "full")
        sparse = deploy_layers(state, ks, "uniform_tail")
        full_cal = representations(cal, full)[-1]
        sparse_cal = representations(cal, sparse)[-1]
        adapted_head = ridge_head(sparse_cal, F.linear(full_cal, *head), head[0])
        full_test = representations(x, full)[-1]
        sparse_test = representations(x, sparse)[-1]
        for method, features, readout in (
            ("full", full_test, head),
            ("uniform_tail", sparse_test, head),
            ("uniform_tail_ridge", sparse_test, adapted_head),
        ):
            logits = F.linear(features, *readout)
            hit = logits.argmax(1) == target
            metrics = {"accuracy_pct": 100 * hit.double().mean().item(),
                       "ce": F.cross_entropy(logits, target).item()}
            correct[method, seed] = hit.numpy().astype(np.int8)
            row = dict(seed=seed, checkpoint_epoch=c["best"]["epoch"], method=method, **metrics)
            row.update(budget(x.shape[1], ks, "full" if method == "full" else "uniform_tail"))
            rows.append(row)
            print(json.dumps(row), flush=True)
    pairs = (("uniform_tail", "full"), ("uniform_tail_ridge", "full"),
             ("uniform_tail_ridge", "uniform_tail"))
    comparisons = {
        a + "_minus_" + b: bootstrap(np.stack([correct[a, seed] - correct[b, seed] for seed in PROTOCOL["seeds"]]))
        for a, b in pairs
    }
    result = {"protocol": PROTOCOL, "rows": rows, "paired_test_accuracy": comparisons}
    (HERE / "frozen_test_results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(HERE / "frozen_test_correctness.npz",
                        **{f"{method}_seed{seed}": values for (method, seed), values in correct.items()})
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
