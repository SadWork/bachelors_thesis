"""Run the frozen CIFAR-selected focus32 calibration on Fashion-MNIST."""
import gzip
import hashlib
import json
import struct

import numpy as np
import torch

from activation_support import (
    CALIBRATION, GRIDS, HERE, SEEDS, bootstrap, deploy_layers, evaluate,
    fit_models, representations, validation_data, write_json,
)
from probe import CHECKPOINTS, WORK, budget


def test_data(meta):
    folder = WORK / "compact_routing_paper_check" / "fashion"
    image_file = folder / "t10k-images-idx3-ubyte.gz"
    label_file = folder / "t10k-labels-idx1-ubyte.gz"
    for file in (image_file, label_file):
        assert hashlib.md5(file.read_bytes()).hexdigest() == meta["source_md5"][file.name]
    image = gzip.decompress(image_file.read_bytes())
    label = gzip.decompress(label_file.read_bytes())
    assert struct.unpack(">IIII", image[:16]) == (2051, 10000, 28, 28)
    assert struct.unpack(">II", label[:8]) == (2049, 10000)
    pixels = torch.from_numpy(np.frombuffer(image[16:], dtype=np.uint8).copy()).float().reshape(10000, 784)
    pixels.div_(255).sub_(meta["channel_mean"][0]).div_(meta["channel_std"][0])
    y = torch.from_numpy(np.frombuffer(label[8:], dtype=np.uint8).copy()).long()
    return pixels, y


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    frozen = json.loads((HERE / "frozen.json").read_text())
    schedule = frozen["selected"]["schedule"]
    assert schedule == "focus32"
    config = GRIDS[schedule]
    modes = ("teacher", "topk_ridge", "topk_refit_ridge")
    xv, yv, cal, calibration_hash = validation_data("fashion", calibration_size=CALIBRATION)
    protocol = {"dataset": "fashion", "schedule": schedule, "ks": config, "methods": modes,
                "seeds": SEEDS, "calibration_size": CALIBRATION,
                "calibration_hash": calibration_hash, "transfer_no_retuning": True}
    plan = HERE / "fashion_transfer_protocol.json"
    if plan.exists():
        assert json.loads(plan.read_text()) == json.loads(json.dumps(protocol))
    else:
        write_json(plan, protocol)
    rows, test_correct = [], {}
    test = None
    for seed in SEEDS:
        name = f"logit_pre_r8_s{seed}"
        folder = CHECKPOINTS / "fashion" / name
        meta = json.loads((folder / "data_manifest.json").read_text())
        if test is None:
            test = test_data(meta)
        checkpoint = torch.load(folder / f"{name}.pt", weights_only=True, map_location="cpu")
        state = checkpoint["best"]["state"]
        layers = deploy_layers(state, None, "full")
        teacher_repr = representations(cal, layers)
        for mode in modes:
            if mode == "teacher":
                fitted_layers, head = layers, (state["head.weight"], state["head.bias"])
            else:
                fitted_layers, head = fit_models(state, cal, layers, teacher_repr, config, mode)
            for name, split in (("validation", (xv, yv)), ("test", test)):
                metric, hit = evaluate(split, fitted_layers, head)
                row = dict(seed=seed, method=mode, split=name, **metric,
                           **budget(784, config, "full" if mode == "teacher" else "uniform_tail"))
                rows.append(row)
                if name == "test":
                    test_correct[mode, seed] = hit.astype(np.int8)
                print(json.dumps(row), flush=True)
    comparisons = {}
    for mode, control in (("topk_ridge", "teacher"), ("topk_refit_ridge", "teacher"),
                          ("topk_refit_ridge", "topk_ridge")):
        delta = np.stack([test_correct[mode, s] - test_correct[control, s] for s in SEEDS])
        comparisons[mode + "_minus_" + control] = bootstrap(delta)
    result = {"protocol_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
              "rows": rows, "paired_test_accuracy": comparisons}
    write_json(HERE / "fashion_transfer_results.json", result)
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
