"""Validation-only test of optimal support under block-exchangeable activations."""
import argparse
import ctypes
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "outputs" / "compact_routing_followup"
sys.path.insert(0, str(PREVIOUS / "sparse_probe"))
from probe import CHECKPOINTS, validation_data

BUDGETS = (32, 64, 96)


def groups_for(dataset):
    side, tile, channels = (32, 8, 3) if dataset == "cifar10" else (28, 7, 1)
    return np.arange(channels * side * side).reshape(
        channels, side // tile, tile, side // tile, tile
    ).transpose(0, 1, 3, 2, 4).reshape(-1, tile * tile)


def window_costs(sorted_weights, windows):
    """All possible contiguous windows at each omitted size, sorted ascending."""
    d = sorted_weights.shape[-1]
    zero = np.zeros((*sorted_weights.shape[:-1], 1))
    s = np.concatenate((zero, sorted_weights.cumsum(-1)), axis=-1)
    q = np.concatenate((zero, (sorted_weights ** 2).cumsum(-1)), axis=-1)
    cost, index = [], []
    for k in range(d):
        length = d - k
        mass = s[..., length:] - s[..., :-length]
        energy = q[..., length:] - q[..., :-length]
        costs = np.maximum(energy - mass ** 2 / length, 0)
        start = costs.argmin(-1) if windows else np.zeros(costs.shape[:-1], dtype=np.int64)
        cost.append(np.take_along_axis(costs, start[..., None], axis=-1)[..., 0])
        index.append(start)
    cost.append(np.zeros_like(cost[0]))
    index.append(np.zeros_like(index[0]))
    return np.stack(cost, axis=-1), np.stack(index, axis=-1)


def allocate(cost, k):
    rows, g, d1 = cost.shape
    d = d1 - 1
    dp = np.full((rows, k + 1), np.inf)
    dp[:, 0] = 0.0
    trace = np.empty((g, rows, k + 1), dtype=np.int16)
    for j in range(g):
        new = np.full_like(dp, np.inf)
        for used in range(min(d, k) + 1):
            candidate = dp[:, :k + 1 - used] + cost[:, j, used, None]
            best = new[:, used:]
            improved = candidate < best
            best[improved] = candidate[improved]
            trace[j, :, used:][improved] = used
        dp = new
    remaining = np.full(rows, k)
    ks = np.empty((rows, g), dtype=np.int64)
    ids = np.arange(rows)
    for j in reversed(range(g)):
        ks[:, j] = trace[j, ids, remaining]
        remaining -= ks[:, j]
    assert (remaining == 0).all() and (ks.sum(-1) == k).all()
    return ks


def mask_from_windows(weights, order, window_starts, budget):
    starts = np.take_along_axis(window_starts, budget[..., None], axis=-1)
    idx = np.arange(weights.shape[-1])
    inside = (starts <= idx) & (idx < starts + weights.shape[-1] - budget[..., None])
    selected = np.empty_like(inside)
    np.put_along_axis(selected, order, ~inside, axis=-1)
    return selected


def average_tail(weights, keep):
    tail = np.where(keep, 0, weights).sum(-1, keepdims=True)
    count = (~keep).sum(-1, keepdims=True)
    return np.where(keep, weights, tail / np.maximum(count, 1))


def test_optimization():
    rng = np.random.default_rng(211)
    p = rng.random((3, 3, 4))
    order = p.argsort(-1)
    cost, starts = window_costs(np.take_along_axis(p, order, axis=-1), True)
    for k in range(1, 10):
        ks = allocate(cost, k)
        keep = mask_from_windows(p, order, starts, ks)
        predicted = ((p - average_tail(p, keep)) ** 2).sum((1, 2))
        expected = np.full(len(p), np.inf)
        for combo in itertools.combinations(range(12), k):
            m = np.zeros((3, 3, 4), bool)
            m.reshape(3, 12)[:, combo] = True
            expected = np.minimum(expected, ((p - average_tail(p, m)) ** 2).sum((1, 2)))
        np.testing.assert_allclose(predicted, expected, atol=1e-13)


@torch.no_grad()
def estimate_gamma(x, w, b, groups):
    features = torch.zeros(len(w), *groups.shape, dtype=torch.float64)
    energy = torch.zeros(len(w), len(groups), dtype=torch.float64)
    for batch in x.split(12):
        z = w[None, :, None] * batch[:, None] + b[None, :, None]
        z = torch.where(z > 0, 2*z, .5*z)[:, :, groups]
        contrast = z - z.mean(-1, keepdim=True)
        features += contrast.sum(0, dtype=torch.float64)
        energy += contrast.square().sum((0, 3), dtype=torch.float64)
    return ((energy / len(x) - (features / len(x)).square().sum(-1))
            / (groups.shape[-1] - 1)).clamp_min(0).numpy()


def build_approximations(p, groups, gamma):
    weights = p[:, groups].astype(np.float64)
    order = weights.argsort(-1)
    sorted_weights = np.take_along_axis(weights, order, axis=-1)
    window, window_starts = window_costs(sorted_weights, True)
    top, top_starts = window_costs(sorted_weights, False)
    approximations = {"full": p}
    diagnostics = {"full": {}}
    for k in BUDGETS:
        idx = np.argpartition(-p, k-1, axis=-1)[:, :k]
        flat_keep = np.zeros_like(p, dtype=bool)
        np.put_along_axis(flat_keep, idx, True, axis=-1)
        global_top = flat_keep[:, groups]
        global_budget = global_top.sum(-1)
        choices = {
            "global_top": global_top,
            "window_fixed": mask_from_windows(weights, order, window_starts, global_budget),
            "dp_top": mask_from_windows(weights, order, top_starts, allocate(top, k)),
            "dp_window": mask_from_windows(weights, order, window_starts, allocate(window, k)),
            "dp_window_gamma": mask_from_windows(
                weights, order, window_starts, allocate(window*gamma[..., None], k)),
        }
        for method, keep in choices.items():
            assert (keep.sum((1, 2)) == k).all()
            selected = average_tail(weights, keep)
            np.testing.assert_allclose(selected.sum(-1), weights.sum(-1), atol=1e-10)
            matrix = np.empty_like(p)
            matrix[:, groups] = selected
            name = f"{method}_{k}"
            approximations[name] = matrix
            tail_sum = np.where(keep, 0, weights).sum(-1, keepdims=True)
            tail_avg = tail_sum / np.maximum((~keep).sum(-1, keepdims=True), 1)
            row_error = (selected - weights)**2
            diagnostics[name] = dict(
                predicted_unweighted_mse=float(row_error.sum((1, 2)).mean()),
                predicted_exchangeable_mse=float((row_error.sum(-1)*gamma).sum(-1).mean()),
                negative_correction_fraction=float((keep & (weights < tail_avg)).sum()/keep.sum()),
            )
    return approximations, diagnostics


def first_layer(library, x, weight, bias, p):
    fn = library.first_outputs
    fn.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    fn.restype = None
    matrices = np.ascontiguousarray(np.stack(list(p.values()), axis=1))
    a = np.ascontiguousarray(x.numpy())
    w = np.ascontiguousarray(weight.numpy())
    b = np.ascontiguousarray(bias.numpy())
    result = np.empty((len(p), len(x), len(w)), np.float32)
    ptr = lambda t: ctypes.c_void_p(t.ctypes.data)
    fn(ptr(a), ptr(w), ptr(b), ptr(matrices), len(x), a.shape[-1], len(w), len(p), ptr(result))
    return torch.from_numpy(result)


@torch.no_grad()
def outputs_after_first(values, state):
    z = values
    for i in range(1, 4):
        layer = f"hidden.{i}."
        p = (state[layer+"left"] @ state[layer+"right"] / math.sqrt(8)).softmax(-1)
        w, b = state[layer+"weight"], state[layer+"bias"]
        out = []
        for batch in z.split(512):
            a = w[None, :, None] * batch[:, None] + b[None, :, None]
            out.append((torch.where(a > 0, 2*a, .5*a)*p).sum(-1))
        z = torch.cat(out)
    return F.linear(z, state["head.weight"], state["head.bias"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "fashion"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,1,2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    test_optimization()
    x, y, xcal, cal_hash = validation_data(args.dataset, calibration_size=512)
    groups = groups_for(args.dataset)
    assert np.array_equal(np.sort(groups.flatten()), np.arange(x.shape[1]))
    lib = ctypes.CDLL(str(HERE / "window_eval.dylib"))
    rows = []
    for seed in args.seeds:
        name = f"logit_pre_r8_s{seed}"
        state = torch.load(CHECKPOINTS / args.dataset / name / f"{name}.pt",
                           weights_only=True, map_location="cpu")["best"]["state"]
        w, b = (state[f"hidden.0.{key}"] for key in ("weight", "bias"))
        p = (state["hidden.0.left"] @ state["hidden.0.right"] / math.sqrt(8)).softmax(-1).numpy()
        gamma = estimate_gamma(xcal, w, b, groups)
        matrices, diagnostics = build_approximations(p, groups, gamma)
        layer_outputs = first_layer(lib, x, w, b, matrices)
        z = w[None, :, None] * x[:8, None, :] + b[None, :, None]
        expected = (torch.where(z > 0, 2*z, .5*z)*torch.from_numpy(p)).sum(-1)
        torch.testing.assert_close(layer_outputs[0, :8], expected, atol=2e-5, rtol=2e-5)
        flat_logits = outputs_after_first(layer_outputs.flatten(0, 1), state)
        logits = flat_logits.reshape(len(matrices), len(x), -1)
        for j, name in enumerate(matrices):
            val_logits = logits[j]
            row = dict(dataset=args.dataset, seed=seed, method=name, cal_hash=cal_hash,
                       first_layer_mse=(layer_outputs[j]-layer_outputs[0]).square().mean().item(),
                       validation_accuracy_pct=100.*(val_logits.argmax(-1)==y).double().mean().item(),
                       validation_ce=F.cross_entropy(val_logits, y).item(), **diagnostics[name])
            rows.append(row)
            print(json.dumps(row), flush=True)
        (HERE / f"{args.dataset}_validation.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
