"""CIFAR-10 validation ablation isolating spatial grouping at identical group counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from compression import expand_tail, groups_for

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "outputs" / "compact_routing_followup"))
from compiled_probe import CompiledRoute, library, load_model, np_float, validation
from data import load_data
from models import birelu


@torch.no_grad()
def compare_mechanism(model, first_probability, approximations, x):
    first = model.hidden[0]
    weights = first.weight[None, :, None]
    biases = first.bias[None, :, None]
    total = {name: 0. for name in approximations}
    for batch in x.split(16):
        f = birelu(weights * batch[:, None, :] + biases)
        teacher = (f * first_probability).sum(-1)
        for name, p in approximations.items():
            actual = (f * p).sum(-1)
            total[name] += (teacher - actual).double().square().sum().item()
    return {name: value / (len(x) * len(first_probability)) for name,value in total.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    reference = WORKSPACE / "outputs" / "compact_routing_paper_check"
    source = WORKSPACE / "work" / "cifar10_shared_scalar" / "data"
    data, metadata = load_data(source, args.out, "cpu")
    x, y = data[1]
    sample = torch.Generator().manual_seed(9829)
    indices = torch.randperm(len(x), generator=sample)[:512]
    measurement = x[indices]
    dll = library(WORKSPACE / "outputs" / "compact_routing_article" / "systems" / "compiled_reference.dylib")
    strategies = {"global": groups_for("global"), "channel": groups_for("channel"),
                  "tiles": groups_for("tiles"),
                  **{f"random_tiles_s{i}": groups_for("random_tiles", 7781+i) for i in range(3)}}
    rows = []
    for seed in (0, 1, 2):
        model = load_model(reference, "logit_pre", seed)
        exact = CompiledRoute(model, (3072,160,160,160), metadata, dll)
        p = exact.expanded[0]
        approximations = {k: expand_tail(p, 64, g) for k,g in strategies.items()}
        error = compare_mechanism(model, p, approximations, measurement)
        teacher = validation(exact, np_float(x), y.numpy())
        for name, q in approximations.items():
            approximated = CompiledRoute(model, (3072,160,160,160), metadata, dll)
            approximated.expanded[0] = q
            approximated.arrays[1] = np_float(torch.cat([x.flatten() for x in approximated.expanded]))
            row = {"seed": seed, "k":64, "grouping":name, "groups": int(strategies[name].max())+1,
                   "teacher_validation":teacher, "validation":validation(approximated, np_float(x), y.numpy()),
                   "layer_mse":error[name], "mse_samples":len(measurement)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    sample_hash = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    (args.out/"results.json").write_text(json.dumps(dict(sample_sha256=sample_hash, rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
