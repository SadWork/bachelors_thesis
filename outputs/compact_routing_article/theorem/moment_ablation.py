"""Compare exact BiReLU tile aggregation against a single-mean tail proxy."""
import argparse
import ctypes
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from window_probe import BUDGETS, CHECKPOINTS, HERE, first_layer, groups_for, outputs_after_first, validation_data

METHODS = ("exact_nonlinear", "omitted_mean_proxy", "whole_mean_proxy")


def compute_moments(x, w, b, p, groups):
    group_by_index = np.empty(p.shape[1], dtype=np.int32)
    group_by_index[groups.flatten()] = np.repeat(np.arange(len(groups)), groups.shape[1])
    budget = np.array(BUDGETS, dtype=np.int32)
    selected, ids = torch.from_numpy(p).topk(int(budget[-1]))
    group_mass = p[:, groups].sum(-1, dtype=np.float64)
    arrays = (x.numpy(), w.numpy(), b.numpy(), groups.astype(np.int32), group_by_index,
              ids.numpy().astype(np.int32), selected.numpy(), group_mass, budget)
    arrays = tuple(np.ascontiguousarray(a) for a in arrays)
    result = np.empty((len(BUDGETS)*len(METHODS), len(x), len(w)), dtype=np.float32)
    lib = ctypes.CDLL(str(HERE/"moment_eval.dylib"))
    fn = lib.first_moments
    fn.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_int]*7 + [ctypes.c_void_p]
    fn.restype = None
    ptr = lambda a: ctypes.c_void_p(a.ctypes.data)
    fn(*[ptr(a) for a in arrays], len(x), x.shape[-1], len(w), *groups.shape, int(budget[-1]), len(budget), ptr(result))
    return torch.from_numpy(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10","fashion"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, y = validation_data(args.dataset)
    groups = groups_for(args.dataset)
    full_lib = ctypes.CDLL(str(HERE/"window_eval.dylib"))
    rows = []
    for seed in range(3):
        name = f"logit_pre_r8_s{seed}"
        state = torch.load(CHECKPOINTS / args.dataset / name / f"{name}.pt",
                           weights_only=True, map_location="cpu")["best"]["state"]
        w, b = state["hidden.0.weight"], state["hidden.0.bias"]
        p = (state["hidden.0.left"] @ state["hidden.0.right"] / math.sqrt(8)).softmax(-1).numpy()
        full = first_layer(full_lib, x, w, b, {"full":p})[0]
        compressed = compute_moments(x, w, b, p, groups)
        margins = compressed[::3].repeat_interleave(len(METHODS), 0) - compressed
        assert margins.min().item() >= -2e-5
        inputs = torch.cat((full[None], compressed), 0)
        logits = outputs_after_first(inputs.flatten(0, 1), state).reshape(len(inputs),len(x),-1)
        for j, (method, k) in enumerate([("full",0)]+[(m,k) for k in BUDGETS for m in METHODS]):
            z = logits[j]
            row = dict(dataset=args.dataset, seed=seed, method=method, k=k,
                       first_layer_mse=(inputs[j]-full).square().mean().item(),
                       mean_jensen_gap=0. if not j else margins[j-1].mean().item(),
                       teacher_prediction_agreement=(z.argmax(-1)==logits[0].argmax(-1)).double().mean().item(),
                       validation_accuracy_pct=100*(z.argmax(-1)==y).double().mean().item(),
                       validation_ce=F.cross_entropy(z,y).item())
            rows.append(row)
            print(json.dumps(row), flush=True)
    (HERE/f"{args.dataset}_moments.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    main()
