"""Validation-only pilot for factorized routing over shared hinge features."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from data import load_data
from fashion import load_fashion
from models import Classifier, WIDTH, DEPTH, birelu

METHODS = ("factor_dense", "route_linear", "signed_hinge", "route_hinge")
LR, BATCH = .003, 512


class BasisPool(nn.Module):
    def __init__(self, inputs, outputs, rank, method):
        super().__init__()
        self.method = method
        self.rank = rank
        self.pool = nn.Parameter(torch.randn(rank, inputs) * .6)
        self.hinge = method.endswith("hinge")
        self.readout = nn.Linear(rank * (3 if self.hinge else 1), outputs)
        if self.hinge:
            self.knots = nn.Parameter(torch.tensor([0., .25]))
        with torch.no_grad():
            self.readout.weight.mul_(math.sqrt(3))

    def probabilities(self):
        return self.pool.softmax(-1)

    def forward(self, x):
        n = x.shape[-1]
        if self.method.startswith("route"):
            # Each pooled feature is an exact categorical expectation minus its uniform reference.
            pool = (self.probabilities() - 1 / n) * math.sqrt(n)
        else:
            pool = (self.pool - self.pool.mean(-1, keepdim=True)) / math.sqrt(n)
        if self.hinge:
            features = torch.cat((x[:, None], F.relu(x[:, None] - self.knots[None, :, None])), dim=1)
            pooled = features.flatten(0, 1) @ pool.T
            return self.readout(pooled.reshape(len(x), -1))
        return self.readout(x @ pool.T)


class BasisMLP(nn.Module):
    def __init__(self, method, inputs):
        super().__init__()
        rank = (6 if inputs == 3072 else 5) if method.endswith("hinge") else 8
        self.hidden = nn.ModuleList(BasisPool(inputs if i == 0 else WIDTH, WIDTH, rank, method)
                                    for i in range(DEPTH))
        self.head = nn.Linear(WIDTH, 10)

    def forward(self, x):
        for layer in self.hidden:
            x = birelu(layer(x))
        return self.head(x)

    def optimizer(self):
        decayed = [layer.readout.weight for layer in self.hidden]
        ids = {id(p) for p in decayed}
        return torch.optim.AdamW([
            {"params": decayed, "weight_decay": .01},
            {"params": [p for p in self.parameters() if id(p) not in ids], "weight_decay": 0.},
        ], lr=LR)


def model_for(method, inputs, device="cpu"):
    if method == "factor_dense":
        model = Classifier(method, 8, inputs).to(device)
        return model, model.optimizer(LR, .01)
    model = BasisMLP(method, inputs).to(device)
    return model, model.optimizer()


@torch.no_grad()
def evaluate(model, split):
    model.eval()
    x, y = split
    correct, ce = x.new_zeros(()), x.new_zeros(())
    for xx, yy in zip(x.split(BATCH), y.split(BATCH)):
        z = model(xx)
        correct += (z.argmax(-1) == yy).sum()
        ce += F.cross_entropy(z, yy, reduction="sum")
    return {"accuracy": correct.item() / len(x), "ce": ce.item() / len(x)}


def save(path, content):
    temp = path.with_suffix(".partial")
    torch.save(content, temp)
    temp.replace(path)


def write_json(path, data):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run(args, data, method, seed):
    name = f"{method}_s{seed}"
    result_path = args.out / f"{name}.json"
    if result_path.exists():
        return
    torch.manual_seed(seed)
    model, optimizer = model_for(method, data[0][0].shape[1], args.device)
    shuffle = torch.Generator().manual_seed(123000 + seed)
    resume = args.out / f"{name}.resume.pt"
    history, best = [], None
    if resume.exists():
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        shuffle.set_state(ckpt["shuffle"])
        history, best = ckpt["history"], ckpt["best"]
    x, y = data[0]
    parameters = sum(p.numel() for p in model.parameters())
    print("START", name, parameters, len(history), flush=True)
    start = time.perf_counter()
    for epoch in range(len(history) + 1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=shuffle).to(x.device)
        correct, loss_sum = x.new_zeros(()), x.new_zeros(())
        for ids in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids])
            targets = y[ids]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(ids)
            correct += (logits.detach().argmax(-1) == targets).sum()
        ce = loss_sum.item() / len(x)
        if not math.isfinite(ce):
            raise FloatingPointError((method, epoch, ce))
        row = dict(epoch=epoch, train_accuracy=correct.item() / len(x), train_ce=ce)
        history.append(row)
        if epoch <= 10 or epoch % 5 == 0 or epoch == args.epochs:
            row["validation"] = evaluate(model, data[1])
            if best is None or row["validation"]["accuracy"] > best["validation"]["accuracy"]:
                best = dict(epoch=epoch, validation=row["validation"],
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print("CHECK", name, epoch, row["validation"], flush=True)
        if epoch % 25 == 0 or epoch == args.epochs:
            save(resume, dict(state=model.state_dict(), optimizer=optimizer.state_dict(),
                              shuffle=shuffle.get_state(), history=history, best=best))
    model.load_state_dict(best["state"])
    result = dict(method=method, seed=seed, parameters=parameters, epochs=args.epochs,
                  best_epoch=best["epoch"], validation=best["validation"], train=evaluate(model, data[0]),
                  history=history, elapsed_seconds=time.perf_counter()-start)
    save(args.out / f"{name}.pt", best["state"])
    write_json(result_path, result)
    resume.unlink(missing_ok=True)
    print("RESULT", name, {k: v for k, v in result.items() if k != "history"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "fashion"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    loader = load_data if args.dataset == "cifar10" else load_fashion
    data, metadata = loader(args.data, args.out, args.device)
    write_json(args.out / "data_manifest.json", metadata)
    write_json(args.out / "config.json", dict(dataset=args.dataset, methods=args.methods, epochs=args.epochs,
                                             seeds=args.seeds, batch=BATCH, lr=LR, torch=torch.__version__,
                                             device=args.device, split_sha256=metadata["split_sha256"]))
    for seed in args.seeds:
        for method in args.methods:
            run(args, data, method, seed)


if __name__ == "__main__":
    main()
