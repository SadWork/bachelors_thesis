"""Certify prediction invariance for compressed routing on fixed validation examples."""
import argparse
import ctypes
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from window_probe import CHECKPOINTS, HERE, groups_for, validation_data


def approx_and_bounds(x, p, w, b, groups, k):
    weights = p[:, groups].numpy().astype(np.float64)
    selected, ids = p.topk(k)
    keep = np.zeros(p.shape, bool)
    np.put_along_axis(keep, ids.numpy(), True, axis=-1)
    tail_weights = np.where(keep[:, groups], 0, weights)
    tail = tail_weights.sum(-1)
    tail_avg = tail / np.maximum((~keep[:, groups]).sum(-1), 1)
    residual = np.where(keep[:, groups], 0, weights - tail_avg[..., None])
    np.testing.assert_allclose(residual.sum(-1), 0, atol=1e-12)
    cap = selected[:, -1].double().numpy()
    assert np.all(tail_weights <= cap[:, None, None])
    lib = ctypes.CDLL(str(HERE / "capped_eval.dylib"))
    fn = lib.first_capped
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    fn.restype = None
    arrays = (x.numpy(), w.numpy(), b.numpy(), groups.astype(np.int32),
              ids.numpy().astype(np.int32), selected.numpy(), keep.astype(np.uint8), tail, cap,
              np.abs(residual).sum(-1), np.sqrt(np.square(residual).sum(-1)))
    arrays = tuple(np.ascontiguousarray(a) for a in arrays)
    result = np.empty((len(x), len(w), 5), dtype=np.float64)
    ptr = lambda a: ctypes.c_void_p(a.ctypes.data)
    fn(*[ptr(a) for a in arrays], len(x), x.size(-1), len(w), *groups.shape, k, ptr(result))
    return tuple(torch.from_numpy(result[..., i].copy()) for i in range(5))


def params(state, i):
    key = f"hidden.{i}."
    w, b = state[key+"weight"].double(), state[key+"bias"].double()
    p = (state[key+"left"] @ state[key+"right"] / math.sqrt(8)).softmax(-1).double()
    return w, b, p


def act(z):
    return torch.where(z > 0, 2*z, .5*z)


def layer(z, w, b, p):
    return (act(z[:, None, :]*w[None, :, None] + b[None, :, None])*p).sum(-1)


def linear_bounds(lo, hi, w, b, p):
    """Independent edgewise affine bounds on pre-pooled BiReLU terms."""
    zl = lo[:, None, :] * w[None, :, None] + b[None, :, None]
    zu = hi[:, None, :] * w[None, :, None] + b[None, :, None]
    zl, zu = torch.minimum(zl, zu), torch.maximum(zl, zu)
    crosses = (zl < 0) & (zu > 0)
    exact_slope = torch.where(zu > 0, 2., .5)
    upper_slope = torch.where(crosses, (act(zu)-act(zl))/(zu-zl).clamp_min(1e-20), exact_slope)
    lower_slope = torch.where(crosses, torch.where(zu > -zl, 2., .5), exact_slope)
    intercept = torch.where(crosses, act(zl)-upper_slope*zl, 0.)
    w = w[None, :, None]
    b = b[None, :, None]
    lower_a, upper_a = lower_slope*w*p, upper_slope*w*p
    lower_b, upper_b = (lower_slope*b*p).sum(-1), ((upper_slope*b + intercept)*p).sum(-1)
    return lower_a, upper_a, lower_b, upper_b


def margins(lo, hi, approximation, teacher, downstream, head, bias):
    intervals = []
    final_lo, final_hi = lo, hi
    for w, b, p in downstream:
        intervals.append((final_lo, final_hi))
        z0 = final_lo[:, None, :]*w[None, :, None] + b[None, :, None]
        z1 = final_hi[:, None, :]*w[None, :, None] + b[None, :, None]
        final_lo = (act(torch.minimum(z0, z1))*p).sum(-1)
        final_hi = (act(torch.maximum(z0, z1))*p).sum(-1)
    winner = approximation.argmax(-1)
    other = torch.arange(head.shape[0])[None, :].expand(len(winner), -1)
    other = other[other != winner[:, None]].reshape(len(winner), -1)
    true_margins = teacher.gather(1, winner[:, None]) - teacher.gather(1, other)
    coefficient = head[winner, None] - head[other]
    offset = bias[winner, None] - bias[other]
    direct = (coefficient.clamp_min(0)*final_lo[:, None, :]
              + coefficient.clamp_max(0)*final_hi[:, None, :]).sum(-1) + offset
    for (a, z), (w, b, p) in zip(reversed(intervals), reversed(downstream)):
        low_a, up_a, low_b, up_b = linear_bounds(a, z, w, b, p)
        pos, neg = coefficient.clamp_min(0), coefficient.clamp_max(0)
        offset = offset + (pos * low_b[:, None, :] + neg * up_b[:, None, :]).sum(-1)
        coefficient = pos @ low_a + neg @ up_a
    relaxed = (coefficient.clamp_min(0)*lo[:, None, :] + coefficient.clamp_max(0)*hi[:, None, :]).sum(-1) + offset
    assert (true_margins >= direct-1e-8).all()
    assert (true_margins >= relaxed-1e-8).all()
    return (direct.min(1).values > 1e-7).sum().item(), (relaxed.min(1).values > 1e-7).sum().item()


def evaluate(x, target, state, groups, k):
    w, b, p = params(state, 0)
    center, low_cap, high_cap, low_norm, high_norm = approx_and_bounds(
        x, p.float(), w.float(), b.float(), groups, k)
    downstream = [params(state, j) for j in range(1, 4)]
    head, bias = state["head.weight"].double(), state["head.bias"].double()
    metrics = {}
    agrees, correct = 0, 0
    for a in range(0, len(x), 12):
        end = min(a+12, len(x))
        window = slice(a, end)
        tc = layer(x[window].double(), w, b, p)
        cc = center[window]
        assert (low_cap[window] <= tc + 1e-8).all() and (tc <= high_cap[window] + 1e-8).all()
        assert (low_norm[window] <= tc + 1e-8).all() and (tc <= high_norm[window] + 1e-8).all()
        for wi, bi, pi in downstream:
            tc, cc = layer(tc, wi, bi, pi), layer(cc, wi, bi, pi)
        teacher, approx = F.linear(tc, head, bias), F.linear(cc, head, bias)
        agrees += (teacher.argmax(-1) == approx.argmax(-1)).sum().item()
        correct += (approx.argmax(-1) == target[window]).sum().item()
        for name, lo, hi in (("cap", low_cap, high_cap), ("norm", low_norm, high_norm),
                             ("intersection", torch.maximum(low_cap,low_norm), torch.minimum(high_cap,high_norm))):
            direct, relaxed = margins(lo[window], hi[window], approx, teacher, downstream, head, bias)
            for m, count in (("interval", direct), ("linear", relaxed)):
                key = name+"_"+m
                metrics[key] = metrics.get(key, 0) + count
    assert all(n <= agrees for n in metrics.values())
    return dict(prediction_agreement=agrees/len(x), approximation_accuracy=correct/len(x),
                first_layer_avg_width_cap=(high_cap-low_cap).mean().item(),
                first_layer_avg_width_norm=(high_norm-low_norm).mean().item(),
                certified={key:n/len(x) for key,n in metrics.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10","fashion"), required=True)
    parser.add_argument("--examples-per-class", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, target = validation_data(args.dataset)
    ids = torch.cat([(target==i).nonzero().flatten()[:args.examples_per_class] for i in range(10)])
    x, target = x[ids], target[ids]
    groups = groups_for(args.dataset)
    rows = []
    for seed in range(3):
        name = f"logit_pre_r8_s{seed}"
        state = torch.load(CHECKPOINTS / args.dataset / name / f"{name}.pt",
                           weights_only=True, map_location="cpu")["best"]["state"]
        for k in (32, 64, 96):
            result = evaluate(x, target, state, groups, k)
            row = dict(dataset=args.dataset, seed=seed, k=k, examples=len(x), **result)
            rows.append(row)
            print(json.dumps(row), flush=True)
        (HERE/f"{args.dataset}_certificate.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
