"""Ablate block mass versus mass plus spatial first moments, at fixed top-K."""
import argparse
import ctypes
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from window_probe import BUDGETS, CHECKPOINTS, HERE, first_layer, groups_for, outputs_after_first, validation_data

METHODS = ("uniform", "affine_signed", "affine_nonnegative")


def tail_projection(p, groups, k):
    # Choose the same global probability top-K for every tail model.
    positions = np.argpartition(-p, k-1, axis=-1)[:, :k] if k else np.empty((len(p), 0), dtype=int)
    mask = np.ones_like(p, dtype=bool)
    np.put_along_axis(mask, positions, False, axis=-1)
    omitted = mask[:, groups]
    values = p[:, groups].astype(np.float64)
    y = np.where(omitted, values, 0)
    tile = math.isqrt(groups.shape[1])
    u, v = np.meshgrid(np.linspace(-1,1,tile), np.linspace(-1,1,tile))
    design = np.stack((np.ones_like(u), u, v), -1).reshape(-1,3)
    assert design.shape[0] == groups.shape[1]
    # Solve moment matching in the smallest-norm affine function space on omitted pixels.
    gram = np.einsum("mgi,ij,ik->mgjk", omitted.astype(np.float64), design, design)
    target = y @ design
    coefficients = (np.linalg.pinv(gram, rcond=1e-11, hermitian=True) @ target[...,None])[...,0]
    affine = coefficients @ design.T
    affine = np.where(omitted, affine, 0.)
    np.testing.assert_allclose(affine @ design, target, atol=2e-11)
    count = omitted.sum(-1)
    uniform = np.where(omitted, (y.sum(-1)/np.maximum(1,count))[...,None], 0.)
    negative = affine < -1e-15
    # Preserve group mass while clipping the largest feasible amount of the affine direction.
    with np.errstate(divide="ignore", invalid="ignore"):
        alphas = np.where(negative, uniform/(uniform-affine), 1.)
    alpha = alphas.min(-1)
    positive = uniform + alpha[..., None]*(affine-uniform)
    assert np.min(positive) >= -1e-12
    assert np.max(np.abs(positive.sum(-1)-y.sum(-1))) < 2e-11
    stats = {
        "signed_negative_fraction": float(negative.sum()/omitted.sum()),
        "blocks_needing_shrink": float((alpha<1-1e-10).mean()),
        "average_shrink": float(alpha.mean()),
    }
    matrices = {}
    for method, q in zip(METHODS,(uniform,affine,positive)):
        value = np.empty_like(p)
        value[:,groups] = np.where(omitted,q,values)
        matrices[method] = value
    return matrices, stats


def fine_groups_for(dataset):
    side, channels = (32, 3) if dataset == "cifar10" else (28, 1)
    tile = 4
    return np.arange(channels*side*side).reshape(
        channels, side//tile, tile, side//tile, tile
    ).transpose(0,1,3,2,4).reshape(-1,tile*tile)


def uniform_projection(p, groups, k):
    positions = np.argpartition(-p, k-1, axis=-1)[:, :k] if k else np.empty((len(p), 0), dtype=int)
    mask = np.ones_like(p, dtype=bool)
    np.put_along_axis(mask, positions, False, axis=-1)
    omitted = mask[:, groups]
    values = p[:, groups].astype(np.float64)
    mass = np.where(omitted,values,0).sum(-1,keepdims=True)
    count = np.maximum(omitted.sum(-1,keepdims=True),1)
    result = np.empty_like(p)
    result[:,groups] = np.where(omitted,mass/count,values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10","fashion"), required=True)
    parser.add_argument("--equal-coarse-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, y = validation_data(args.dataset)
    groups = groups_for(args.dataset)
    fine_groups = fine_groups_for(args.dataset)
    full_lib = ctypes.CDLL(str(HERE/"window_eval.dylib"))
    rows = []
    for seed in range(3):
        name = f"logit_pre_r8_s{seed}"
        state = torch.load(CHECKPOINTS / args.dataset / name / f"{name}.pt",
                           weights_only=True, map_location="cpu")["best"]["state"]
        w, b = state["hidden.0.weight"], state["hidden.0.bias"]
        p = (state["hidden.0.left"] @ state["hidden.0.right"] / math.sqrt(8)).softmax(-1).numpy()
        matrices = {"full":p}
        details = {"full":{"first_layer_bytes":int(p.nbytes)}}
        if args.equal_coarse_only:
            # Compare to the strongest simple alternative at (approximately)
            # the same byte budget: spend both extra moments on more routes.
            for k in BUDGETS:
                budget = k + (8*len(groups))//6
                name = f"uniform_coarse_{budget}"
                matrices[name] = uniform_projection(p, groups, budget)
                details[name] = dict(first_layer_bytes=int(len(p)*(6*budget+4*len(groups))),
                                     groups=len(groups), coefficients_per_group=1)
        else:
            for k in BUDGETS:
                approximations, stats = tail_projection(p, groups, k)
                for method, value in approximations.items():
                    name = f"{method}_{k}"
                    matrices[name] = value
                    ncoef = 1 if method == "uniform" else 3
                    memory = dict(first_layer_bytes=int(len(p)*(6*k+4*len(groups)*ncoef)),
                                  groups=len(groups), coefficients_per_group=ncoef)
                    details[name] = {**memory, **(stats if method != "uniform" else {})}
            for k in (0,*BUDGETS):
                name = f"uniform_fine_{k}"
                matrices[name] = uniform_projection(p, fine_groups, k)
                details[name] = dict(first_layer_bytes=int(len(p)*(6*k+4*len(fine_groups))),
                                     groups=len(fine_groups), coefficients_per_group=1)
        values = first_layer(full_lib, x, w, b, matrices)
        logits = outputs_after_first(values.flatten(0,1), state).reshape(len(values),len(x),-1)
        for j, name in enumerate(matrices):
            z = logits[j]
            row = dict(dataset=args.dataset, seed=seed, method=name,
                       first_layer_mse=(values[j]-values[0]).square().mean().item(),
                       validation_accuracy_pct=100*(z.argmax(-1)==y).double().mean().item(),
                       validation_ce=F.cross_entropy(z,y).item(), **details[name])
            rows.append(row)
            print(json.dumps(row), flush=True)
    suffix = "coarse_budget" if args.equal_coarse_only else "spatial_moments"
    (HERE/f"{args.dataset}_{suffix}.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    main()
