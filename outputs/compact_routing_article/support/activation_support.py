"""Calibration-only activation-aware compression of CIFAR-10 routing layers.

Select supports with regularized forward selection and fit nonnegative
corrections to a uniform route distribution. Both learned-coefficient
variants receive the same layerwise and head calibration budget.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "compact_routing_followup" / "sparse_probe"))

from calibrate import apply_layer, representations, ridge_head
from probe import CHECKPOINTS, all_activations_sum, budget, deploy_layers, validation_data
from test_frozen import bootstrap, original_test_set

CALIBRATION = 1536
SELECTION_ROWS = 512
SEEDS = (0, 1, 2)
GRIDS = {
    "all32": (32, 32, 32, 32),
    "focus32": (64, 32, 64, 32),
    "all64": (64, 64, 64, 64),
    "all16": (16, 16, 16, 16),
}
MODES = ("topk_ridge", "topk_refit_ridge", "activation_refit_ridge")


def features(x, w, b, ids=None):
    if ids is None:
        values = x.unsqueeze(0)
    else:
        values = x[:, ids].transpose(0, 1)
    u = values * w[:, None, None] + b[:, None, None]
    return torch.where(u > 0, 2 * u, .5 * u)


def project_capped_simplex(x):
    positive = x.clamp_min(0)
    descending = x.sort(dim=-1, descending=True).values
    ranks = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
    shifts = (descending.cumsum(-1) - 1) / ranks
    active = (descending > shifts).sum(-1, keepdim=True).clamp_min(1) - 1
    projected = (x - shifts.gather(-1, active)).clamp_min(0)
    return torch.where(positive.sum(-1, keepdim=True) <= 1, positive, projected)


def activation_support(x, w, b, p, anchor, target, k):
    x = x[:SELECTION_ROWS]
    residual = target[:, :len(x)].unsqueeze(-1)
    means = anchor[:, :len(x)].unsqueeze(-1)
    f = features(x, w, b) - means
    corr = (f.transpose(1, 2) @ residual).squeeze(-1)
    norm = f.square().sum(1).clamp_min(1e-12).sqrt()
    top = min(64, p.size(-1))
    high_mass = p.topk(top, dim=-1).indices
    high_signal = (corr / norm).topk(top, dim=-1).indices
    priority = p * .01
    priority.scatter_add_(1, high_mass, torch.ones_like(high_mass, dtype=p.dtype))
    priority.scatter_add_(1, high_signal, torch.ones_like(high_signal, dtype=p.dtype))
    pool = priority.topk(min(128, p.size(-1)), dim=-1).indices
    z = f.gather(-1, pool[:, None].expand(-1, len(x), -1))
    gram = z.transpose(1, 2) @ z / len(x)
    gradient = (z.transpose(1, 2) @ residual).squeeze(-1) / len(x)
    ridge = 1e-8 + 1e-4 * gram.diagonal(dim1=1, dim2=2).mean(-1)
    gram.diagonal(dim1=1, dim2=2).add_(ridge[:, None])
    chosen = []
    disabled = torch.zeros_like(pool, dtype=torch.bool)
    row = torch.arange(len(w))
    for _ in range(k):
        diagonal = gram.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        score = gradient / diagonal.sqrt()
        best = score.masked_fill(disabled, -torch.inf).argmax(-1)
        chosen.append(pool[row, best])
        disabled[row, best] = True
        column = gram[row, :, best]
        pivot = diagonal[row, best]
        gradient = gradient - column * (gradient[row, best] / pivot)[:, None]
        gram = gram - column.unsqueeze(2) * (column / pivot[:, None]).unsqueeze(1)
    return torch.stack(chosen, dim=-1)


def fit_coefficients(x, w, b, ids, p, anchor, target):
    design = features(x, w, b, ids) - anchor.unsqueeze(-1)
    gram = design.transpose(1, 2) @ design / len(x)
    cross = (design.transpose(1, 2) @ target.unsqueeze(-1)).squeeze(-1) / len(x)
    selected = p.gather(1, ids)
    initial_tail = (1 - selected.sum(-1, keepdim=True)).clamp_min(0) / (p.size(-1) - ids.size(-1))
    prior = project_capped_simplex(selected - initial_tail)
    penalty = 1e-8 + 1e-4 * gram.diagonal(dim1=1, dim2=2).mean(-1)
    gram.diagonal(dim1=1, dim2=2).add_(penalty[:, None])
    cross = cross + penalty[:, None] * prior
    step = gram.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-12).reciprocal()
    coefficient = accelerated = prior
    momentum = 1.
    for _ in range(160):
        gradient = (gram @ accelerated.unsqueeze(-1)).squeeze(-1) - cross
        estimate = project_capped_simplex(accelerated - step * gradient)
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
        accelerated = estimate + ((momentum - 1) / next_momentum) * (estimate - coefficient)
        coefficient, momentum = estimate, next_momentum
    uniform = (1 - coefficient.sum(-1)).clamp(0, 1)
    mass = coefficient + uniform[:, None] / p.size(-1)
    return mass.half().float(), uniform * (1 - ids.size(-1) / p.size(-1))


def fit_layer(x, teacher, layer, k, mode):
    weight, bias, p, _, _ = layer
    n = x.size(-1)
    anchor = torch.cat([all_activations_sum(xx, weight, bias) for xx in x.split(128)]) / n
    residual = teacher - anchor
    values, supports, tails = [], [], []
    for rows in torch.arange(len(weight)).split(8):
        w, b, prob = weight[rows], bias[rows], p[rows]
        ref, target = anchor[:, rows].T, residual[:, rows].T
        if mode == "activation_refit_ridge":
            ids = activation_support(x, w, b, prob, ref, target, k)
        else:
            ids = prob.topk(k, dim=-1).indices
        mass, tail = fit_coefficients(x, w, b, ids, prob, ref, target)
        values.append(mass)
        supports.append(ids)
        tails.append(tail)
    return weight, bias, torch.cat(values), torch.cat(supports), torch.cat(tails)


def load_teacher(seed):
    name = f"logit_pre_r8_s{seed}"
    return torch.load(CHECKPOINTS / "cifar10" / name / f"{name}.pt",
                      weights_only=True, map_location="cpu")["best"]["state"]


def fit_models(state, cal, teacher_layers, teacher_repr, config, mode):
    head = state["head.weight"], state["head.bias"]
    if mode == "topk_ridge":
        layers = deploy_layers(state, config, "uniform_tail")
        hidden = representations(cal, layers)[-1]
    else:
        layers, hidden = [], cal
        for k, teacher_layer, target in zip(config, teacher_layers, teacher_repr):
            layer = fit_layer(hidden, target, teacher_layer, k, mode)
            layers.append(layer)
            hidden = torch.cat([apply_layer(x, layer) for x in hidden.split(128)])
    logits = F.linear(teacher_repr[-1], *head)
    adjusted_head = ridge_head(hidden, logits, head[0])
    return layers, adjusted_head


def evaluate(split, layers, head):
    x, y = split
    logits = F.linear(representations(x, layers)[-1], *head)
    correct = logits.argmax(-1) == y
    return {"accuracy_pct": 100 * correct.double().mean().item(), "ce": F.cross_entropy(logits, y).item()}, correct.numpy()


def write_json(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validation():
    xval, yval, cal, calibration_hash = validation_data("cifar10", calibration_size=CALIBRATION)
    protocol = {
        "dataset": "cifar10", "calibration_size": CALIBRATION, "calibration_hash": calibration_hash,
        "selection_rows": SELECTION_ROWS, "budgets": GRIDS, "methods": MODES, "seeds": SEEDS,
        "freeze_rule": ("Smallest explicit-edge count among activation-aware schedules with at most 1 pp mean "
                        "validation drop vs teacher. If none qualify, select best activation-aware mean accuracy. "
                        "Ties resolve by mean validation accuracy, then CE."),
    }
    protocol_path = HERE / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(protocol))
    else:
        write_json(protocol_path, protocol)
    results_path = HERE / "validation.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    completed = {(r["seed"], r["schedule"], r["method"]) for r in rows}
    for seed in SEEDS:
        state = load_teacher(seed)
        layers = deploy_layers(state, None, "full")
        teacher_repr = representations(cal, layers)
        original_head = state["head.weight"], state["head.bias"]
        if (seed, "full", "teacher") not in completed:
            metrics, _ = evaluate((xval, yval), layers, original_head)
            row = dict(seed=seed, schedule="full", method="teacher", **metrics)
            rows.append(row)
            write_json(results_path, rows)
            print(json.dumps(row), flush=True)
        for schedule, config in GRIDS.items():
            for mode in MODES:
                if (seed, schedule, mode) in completed:
                    continue
                compressed, head = fit_models(state, cal, layers, teacher_repr, config, mode)
                metrics, _ = evaluate((xval, yval), compressed, head)
                row = dict(seed=seed, schedule=schedule, ks=config, method=mode, **metrics,
                           **budget(xval.size(1), config, "uniform_tail"))
                # Preserve exact deployed probabilities, indices, tail and head for frozen test.
                path = HERE / f"{schedule}_{mode}_seed{seed}.pt"
                temp = path.with_suffix(".partial")
                torch.save({"layers": compressed, "head": head}, temp)
                temp.replace(path)
                rows.append(row)
                write_json(results_path, rows)
                print(json.dumps(row), flush=True)


def freeze():
    rows = json.loads((HERE / "validation.json").read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    teacher = {r["seed"]: r["accuracy_pct"] for r in rows if r["method"] == "teacher"}
    assert set(teacher) == set(SEEDS)
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] != "teacher":
            grouped[row["schedule"], row["method"]].append(row)
    assert set(grouped) == {(s, m) for s in GRIDS for m in MODES}
    stats = []
    for (schedule, method), values in grouped.items():
        assert {v["seed"] for v in values} == set(SEEDS)
        stats.append({"schedule": schedule, "method": method, "explicit_scalar_edges": values[0]["explicit_scalar_edges"],
                      "accuracy_pct": float(np.mean([v["accuracy_pct"] for v in values])),
                      "ce": float(np.mean([v["ce"] for v in values])),
                      "delta_vs_teacher_pp": float(np.mean([v["accuracy_pct"]-teacher[v["seed"]] for v in values]))})
    candidates = [s for s in stats if s["method"] == "activation_refit_ridge"]
    qualified = [s for s in candidates if s["delta_vs_teacher_pp"] >= -1.]
    if qualified:
        selected = min(qualified, key=lambda s: (s["explicit_scalar_edges"], -s["accuracy_pct"], s["ce"]))
    else:
        selected = min(candidates, key=lambda s: (-s["accuracy_pct"], s["ce"]))
    plan = {"protocol": protocol, "selected": selected, "validation_summary": stats, "no_test_retuning": True}
    path = HERE / "frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        write_json(path, plan)
    print(json.dumps(selected), flush=True)


def frozen_test():
    frozen_path = HERE / "frozen.json"
    plan = json.loads(frozen_path.read_text())
    schedule = plan["selected"]["schedule"]
    split = original_test_set()
    rows, correctness = [], {}
    for seed in SEEDS:
        state = load_teacher(seed)
        config = GRIDS[schedule]
        exact = (deploy_layers(state, None, "full"), (state["head.weight"], state["head.bias"]))
        for mode in ("teacher", *MODES):
            if mode == "teacher":
                layers, head = exact
            else:
                path = HERE / f"{schedule}_{mode}_seed{seed}.pt"
                payload = torch.load(path, weights_only=True, map_location="cpu")
                layers, head = payload["layers"], payload["head"]
            metrics, correct = evaluate(split, layers, head)
            row = dict(seed=seed, method=mode, schedule=schedule, **metrics)
            correctness[mode, seed] = correct.astype(np.int8)
            rows.append(row)
            print(json.dumps(row), flush=True)
    comparisons = {}
    for mode in MODES:
        for control in ("teacher",) if mode != "activation_refit_ridge" else ("teacher", "topk_ridge", "topk_refit_ridge"):
            diff = np.stack([correctness[mode, s] - correctness[control, s] for s in SEEDS])
            comparisons[f"{mode}_minus_{control}"] = bootstrap(diff)
    result = {"frozen_protocol_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "selection": plan["selected"], "rows": rows, "paired_test_accuracy": comparisons}
    write_json(HERE / "test_results.json", result)
    print(json.dumps(comparisons), flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validation", "freeze", "test"))
    phase = parser.parse_args().phase
    torch.set_num_threads(2)
    {"validation": validation, "freeze": freeze, "test": frozen_test}[phase]()


if __name__ == "__main__":
    main()
