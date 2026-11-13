"""Matched-depth MNIST comparison of a dense MLP and shared-scalar mean layers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
MEAN_SOURCE = ROOT.parent / "cifar10_shared_scalar_both_10_500"
MNIST_SOURCE = ROOT.parent / "ablation_birelu"
sys.path[:0] = [str(MEAN_SOURCE), str(MNIST_SOURCE)]
from base_experiment import load_data
from model import MeanLayer, birelu, row_mean_kl

WIDTH = 160
BATCH = 512
LR = 0.003


class Classifier(nn.Module):
    def __init__(self, kind, depth):
        super().__init__()
        layer = MeanLayer if kind == "mean" else nn.Linear
        self.hidden = nn.ModuleList([layer(784, WIDTH), layer(WIDTH, WIDTH)])
        self.head = nn.Linear(WIDTH, 10)
        self.hidden.extend(layer(WIDTH, WIDTH) for _ in range(depth - 2))
        self.kind = kind

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
            if self.kind == "dense":
                x = birelu(x)
        return self.head(x)

    def kl(self):
        return torch.stack([row_mean_kl(layer.logits) for layer in self.hidden]).mean()

    def optimizer(self, decay):
        weights = [layer.weight for layer in self.hidden] + [self.head.weight]
        weight_ids = {id(p) for p in weights}
        return torch.optim.AdamW([
            {"params": weights, "weight_decay": decay},
            {"params": [p for p in self.parameters() if id(p) not in weight_ids], "weight_decay": 0.0},
        ], lr=LR)


def conditions():
    for decay, coefficient in ((0., 0.), (.03, .10)):
        for depth in (2, 4):
            for kind in ("dense", "mean"):
                yield {"model": kind, "hidden_layers": depth, "width": WIDTH, "decay": decay,
                       "kl": coefficient if kind == "mean" else 0.}


def run_name(spec):
    return f"{spec['model']}_h{spec['hidden_layers']}_wd{spec['decay']:g}_kl{spec['kl']:g}"


def write_json(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, x, y):
    totals = x.new_zeros(2)
    for xx, yy in zip(x.split(BATCH), y.split(BATCH)):
        z = model(xx)
        totals[0] += (z.argmax(-1) == yy).sum()
        totals[1] += F.cross_entropy(z, yy, reduction="sum")
    accuracy, ce = (totals / len(x)).tolist()
    return {"accuracy": accuracy, "ce": ce}


def train_one(spec, data, args, output):
    name = run_name(spec)
    result_path = output / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    x, y, vx, vy, tx, ty = data
    torch.manual_seed(args.seed)
    model = Classifier(spec["model"], spec["hidden_layers"]).to(args.device)
    optimizer = model.optimizer(spec["decay"])
    rng = torch.Generator().manual_seed(123000 + args.seed)
    history, best, best_epoch = [], -1., 0
    resume_path = output / "resume.pt"
    best_path = output / "best.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, weights_only=True, map_location="cpu")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        history, best, best_epoch = saved["history"], saved["best"], saved["best_epoch"]
    print("START", name, flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=rng).to(args.device)
        totals = x.new_zeros(2)
        for ids in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids])
            ce = F.cross_entropy(logits, y[ids])
            loss = ce + spec["kl"] * model.kl() if spec["kl"] else ce
            loss.backward()
            optimizer.step()
            totals[0] += ce.detach() * len(ids)
            totals[1] += (logits.detach().argmax(-1) == y[ids]).sum()
        train_ce, train_accuracy = (totals / len(x)).tolist()
        if not math.isfinite(train_ce):
            raise RuntimeError(f"{name}: non-finite train CE at epoch {epoch}")
        row = {"epoch": epoch, "train_ce": train_ce, "train_accuracy": train_accuracy}
        history.append(row)
        if epoch <= 20 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val = evaluate(model, vx, vy)
            row["validation"] = val
            if spec["model"] == "mean":
                with torch.no_grad():
                    row["kl_by_layer"] = [row_mean_kl(layer.logits).item() for layer in model.hidden]
            if val["accuracy"] > best:
                best, best_epoch = val["accuracy"], epoch
                save(best_path, {"state": cpu_state(model), "condition": spec, "epoch": epoch})
            save(resume_path, {"state": cpu_state(model), "optimizer": optimizer.state_dict(),
                               "rng": rng.get_state(), "history": history, "best": best,
                               "best_epoch": best_epoch})
            write_json(output / "history.json", history)
            print(json.dumps({"run": name, **row}), flush=True)
    model.eval()
    final_test = evaluate(model, tx, ty)
    model.load_state_dict(torch.load(best_path, weights_only=True, map_location="cpu")["state"])
    validation_rows = [row for row in history if "validation" in row]
    lowest_ce = min(validation_rows, key=lambda row: row["validation"]["ce"])
    result = {"name": name, "condition": spec, "seed": args.seed, "epochs": len(history),
              "parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
              "validation_at_best": evaluate(model, vx, vy),
              "test_at_best": evaluate(model, tx, ty),
              "minimum_validation_ce": lowest_ce["validation"]["ce"],
              "minimum_validation_ce_epoch": lowest_ce["epoch"],
              "validation_at_last": history[-1]["validation"], "test_at_last": final_test}
    write_json(result_path, result)
    print("RESULT", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    specs = list(conditions())
    config = {"seed": args.seed, "epochs": args.epochs, "device": args.device, "lr": LR,
              "batch": BATCH, "conditions": specs,
              "sources": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), MEAN_SOURCE / "model.py",
                                    MEAN_SOURCE / "metal_mean.py", MEAN_SOURCE / "mean.metal",
                                    MNIST_SOURCE / "base_experiment.py")}}
    config_path = ROOT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("The existing run uses a different configuration")
    write_json(config_path, config)
    data = load_data(args.data, ROOT)
    assert json.loads((ROOT / "data_manifest.json").read_text()) == json.loads(
        (MNIST_SOURCE / "reference_manifest.json").read_text())
    with np.load(ROOT / "split.npz") as actual, np.load(MNIST_SOURCE / "reference_split.npz") as expected:
        assert all(np.array_equal(actual[key], expected[key]) for key in ("train", "validation"))
    data = tuple(t.to(args.device) for t in data)
    results = []
    for spec in specs:
        out = ROOT / "runs" / run_name(spec)
        out.mkdir(parents=True, exist_ok=True)
        results.append(train_one(spec, data, args, out))
        write_json(ROOT / "summary.json", results)
    print("COMPLETE", len(results), flush=True)


if __name__ == "__main__":
    main()
