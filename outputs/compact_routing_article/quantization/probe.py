"""Fixed weight-only format controls for CIFAR-10 static routing and Dense."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[3]
FOLLOWUP = WORKSPACE / "outputs" / "compact_routing_followup"
sys.path.insert(0, str(FOLLOWUP))
from compiled_probe import CompiledDense, CompiledRoute, library, load_model, np_float, validation
from data import load_data


@torch.no_grad()
def quantize_probability(p, bits, mode):
    if bits == 16:
        value = p.half().float()
    elif mode == "linear":
        max_code = 2**bits - 1
        scale = p.amax(-1, keepdim=True) / max_code
        value = (p / scale).round().clamp(0, max_code) * scale
    else:
        # Quantize relative log probabilities with a fixed twelve-nat floor.
        relative = p.clamp_min(torch.finfo(p.dtype).tiny).log()
        relative -= relative.amax(-1, keepdim=True)
        step = 12 / (2**bits - 1)
        value = ((relative.clamp(-12, 0)/step).round()*step).exp()
    return value / value.sum(-1, keepdim=True)


def route_variant(model, cfg, bits, mode, dll):
    compiled = CompiledRoute(model, (3072, 160, 160, 160), cfg, dll)
    quantized = [quantize_probability(p, bits, mode) for p in compiled.expanded]
    compiled.arrays[1] = np_float(torch.cat([p.flatten() for p in quantized]))
    compiled.expanded = quantized
    matrices = sum(math.ceil(p.numel()*bits/8) for p in quantized)
    # Uniform log codes decode to relative positive weights. Store a row
    # normalization factor for linear/log codes, but not per-row extrema.
    # For absolute linear probabilities the denominator absorbs the step.
    row_norm = 0 if bits == 16 else 4 * sum(len(p) for p in quantized)
    other = sum(a.nbytes for a in compiled.arrays[4:])
    return compiled, matrices + row_norm + other


def dense_variant(model, bits, dll):
    model = copy.deepcopy(model)
    weights = [layer.weight for layer in model.hidden] + [model.head.weight]
    with torch.no_grad():
        for w in weights:
            if bits == 16:
                w.copy_(w.half().float())
            else:
                # Symmetric scaling uses a signed code with a dedicated zero.
                max_code = 2**(bits-1) - 1
                scale = w.abs().amax(-1, keepdim=True).clamp_min(1e-8) / max_code
                w.copy_((w/scale).round().clamp(-max_code,max_code)*scale)
    matrices = sum(math.ceil(w.numel()*bits/8) for w in weights)
    scales = 0 if bits == 16 else 4 * sum(len(w) for w in weights)
    biases = 4*sum(layer.bias.numel() for layer in [*model.hidden, model.head])
    return CompiledDense(model, dll), matrices + scales + biases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    data, metadata = load_data(args.data, args.out, "cpu")
    assert metadata["split_sha256"] == "62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"
    val, test = [(np_float(x), y.numpy()) for x,y in data[1:]]
    dll = library(args.library)
    result = []
    for seed in range(3):
        pre = load_model(args.reference, "logit_pre", seed)
        dense = load_model(args.reference, "full_dense", seed)
        methods = {}
        full = CompiledRoute(pre, (3072,160,160,160), metadata, dll)
        full_dense = CompiledDense(dense, dll)
        methods["route_fp32"] = full, full.bytes
        methods["dense_fp32"] = full_dense, full_dense.bytes
        methods["route_fp16"] = route_variant(pre, metadata, 16, "linear", dll)
        methods["dense_fp16"] = dense_variant(dense, 16, dll)
        for bits in (2,3,4,8):
            for mode in ("linear","log"):
                methods[f"route_{mode}{bits}"] = route_variant(pre, metadata, bits, mode, dll)
            methods[f"dense_linear{bits}"] = dense_variant(dense, bits, dll)
        for name,(model,stored_bytes) in methods.items():
            row = dict(seed=seed, method=name, packed_bytes_estimate=stored_bytes,
                       validation=validation(model, *val), test=validation(model, *test))
            result.append(row)
            print(json.dumps(row), flush=True)
    (args.out/"results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
