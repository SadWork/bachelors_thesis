"""Fit cheap affine/ridge compensation on train only, then evaluate validation."""
import argparse
import json

import torch
from torch.nn import functional as F

from probe import CHECKPOINTS, GRID, HERE, all_activations_sum, deploy_layers, validation_data

CONFIGS = ("uniform_k64", "focus_13", "moderate_13", "uniform_k128")
CALIBRATION_SIZE = 1536


def apply_layer(x, layer):
    weight, bias, p, indices, tail = layer
    z = (x[:, None, :] if indices is None else x[:, indices]) * weight[:, None] + bias[:, None]
    f = torch.where(z > 0, 2 * z, .5 * z)
    out = (f * p).sum(-1)
    if indices is not None:
        avg_tail = (all_activations_sum(x, weight, bias) - f.sum(-1)) / (x.shape[1] - indices.shape[-1])
        out += tail * avg_tail
    return out


def representations(x, layers):
    outputs = []
    for layer in layers:
        x = torch.cat([apply_layer(batch, layer) for batch in x.split(128)])
        outputs.append(x)
    return outputs


def fit_affine(x, layers, targets):
    parameters = []
    for layer, target in zip(layers, targets):
        current = torch.cat([apply_layer(batch, layer) for batch in x.split(128)])
        var = current.var(dim=0, unbiased=False)
        u, v = current.mean(0), target.mean(0)
        cov = ((current-u) * (target-v)).mean(0)
        ridge = 1e-8 + .01 * var
        scale = ((cov + ridge) / (var + ridge)).clamp(.25, 4)
        shift = v - scale * u
        x = current * scale + shift
        parameters.append((scale, shift))
    return x, parameters


def apply_affine(x, layers, parameters):
    for layer, (scale, shift) in zip(layers, parameters):
        x = torch.cat([apply_layer(batch, layer) for batch in x.split(128)]) * scale + shift
    return x


def ridge_head(hidden, logits, initial):
    h, z = hidden.double(), logits.double()
    h_mean, z_mean = h.mean(0), z.mean(0)
    h, z = h-h_mean, z-z_mean
    prior = initial.T.double()
    gram = h.T @ h
    penalty = 1e-6 + .01 * gram.diag().mean()
    delta = torch.linalg.solve(gram + penalty * torch.eye(gram.size(0), dtype=gram.dtype), h.T @ (z - h @ prior))
    weight = prior + delta
    bias = z_mean - h_mean @ weight
    return weight.T.float(), bias.float()


def evaluate(x, y, weight, bias):
    z = F.linear(x, weight, bias)
    return {"accuracy_pct": 100*(z.argmax(1)==y).double().mean().item(), "ce": F.cross_entropy(z,y).item()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "fashion"), default="cifar10")
    dataset = parser.parse_args().dataset
    torch.set_num_threads(2)
    x, y, xcal, index_hash = validation_data(dataset, calibration_size=CALIBRATION_SIZE)
    rows = []
    for seed in range(3):
        name = f"logit_pre_r8_s{seed}"
        checkpoint = torch.load(CHECKPOINTS / dataset / name / f"{name}.pt",
                                weights_only=True, map_location="cpu")
        state = checkpoint["best"]["state"]
        original_head = state["head.weight"], state["head.bias"]
        teacher_layers = deploy_layers(state, None, "full")
        teacher = representations(xcal, teacher_layers)
        teacher_logits = F.linear(teacher[-1], *original_head)
        for config in CONFIGS:
            ks = GRID[config]
            layers = deploy_layers(state, ks, "uniform_tail")
            raw_cal = representations(xcal, layers)[-1]
            raw_val = representations(x, layers)[-1]
            affine_cal, affines = fit_affine(xcal, layers, teacher)
            affine_val = apply_affine(x, layers, affines)
            ridge_raw = ridge_head(raw_cal, teacher_logits, original_head[0])
            ridge_affine = ridge_head(affine_cal, teacher_logits, original_head[0])
            for method, val, head in (
                ("uniform_tail", raw_val, original_head),
                ("affine", affine_val, original_head),
                ("ridge", raw_val, ridge_raw),
                ("affine_plus_ridge", affine_val, ridge_affine),
            ):
                row = dict(dataset=dataset, seed=seed, configuration=config, ks=ks, method=method,
                           calibration_size=CALIBRATION_SIZE, calibration_index_hash=index_hash)
                row.update(evaluate(val, y, *head))
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = "calibration_results.json" if dataset == "cifar10" else "calibration_fashion_results.json"
    (HERE / output).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
