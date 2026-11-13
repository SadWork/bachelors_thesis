"""Train fresh Dense controls and replicate placement on Fashion-MNIST."""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import torch
from torch.nn import functional as F

from data import load_data
from fashion import load_fashion
from models import Classifier, METHODS, routing_parameters

SPLIT = "62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"
LR, DECAY, BATCH = .003, .01, 512


def write_json(path, item):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save(path, item):
    temp = path.with_suffix(".partial")
    torch.save(item, temp)
    temp.replace(path)


def publish(args, item):
    write_json(args.out / "status.json", item)
    if args.device == "cuda":
        temp = args.out / "metrics.partial"
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("status.json", "config.json", "data_manifest.json", "split.npz"):
                archive.write(args.out / name, name)
            for path in sorted((args.out / "runs").glob("*.json")):
                archive.write(path, path.relative_to(args.out))
        temp.replace(args.out / "metrics.zip")


def state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    x, y = data
    correct = x.new_zeros(())
    ce = x.new_zeros(())
    for xx, yy in zip(x.split(BATCH), y.split(BATCH)):
        z = model(xx)
        correct += (z.argmax(-1) == yy).sum()
        ce += F.cross_entropy(z, yy, reduction="sum")
    return {"accuracy": correct.item() / len(x), "ce": ce.item() / len(x)}


def train_epoch(model, optimizer, data, shuffle):
    model.train()
    x, y = data
    order = torch.randperm(len(x), generator=shuffle).to(x.device)
    correct, ce = x.new_zeros(()), x.new_zeros(())
    for ids in order.split(BATCH):
        optimizer.zero_grad(set_to_none=True)
        z, t = model(x[ids]), y[ids]
        loss = F.cross_entropy(z, t)
        loss.backward()
        optimizer.step()
        ce += loss.detach() * len(ids)
        correct += (z.detach().argmax(-1) == t).sum()
    return {"accuracy": correct.item() / len(x), "ce": ce.item() / len(x)}


def train(args, data, method, rank, seed, completed):
    name = f"{method}_r{rank}_s{seed}"
    path = args.out / "runs" / f"{name}.json"
    if path.exists():
        result = json.loads(path.read_text())
        assert result["epochs"] == args.epochs
        return name
    torch.manual_seed(seed)
    inputs = data[0][0].shape[1]
    model = Classifier(method, rank, inputs=inputs).to(args.device)
    parameters = sum(p.numel() for p in model.parameters())
    if not method.endswith("dense"):
        assert parameters == routing_parameters(rank, inputs)
    optimizer = model.optimizer(LR, DECAY)
    shuffle = torch.Generator().manual_seed(123000 + seed)
    history, best = [], None
    resume = args.out / "resume" / f"{name}.pt"
    if resume.exists():
        item = torch.load(resume, weights_only=True, map_location="cpu")
        model.load_state_dict(item["state"])
        optimizer.load_state_dict(item["optimizer"])
        shuffle.set_state(item["shuffle"])
        history, best = item["history"], item["best"]
    start = len(history)
    print("START", name, parameters, start, flush=True)
    for epoch in range(start + 1, args.epochs + 1):
        started = time.perf_counter()
        tr = train_epoch(model, optimizer, data[0], shuffle)
        if not math.isfinite(tr["ce"]):
            raise FloatingPointError((name, epoch, tr))
        row = dict(epoch=epoch, train=tr, seconds=time.perf_counter()-started)
        history.append(row)
        if epoch <= 20 or epoch % 5 == 0 or epoch == args.epochs:
            row["validation"] = evaluate(model, data[1])
            if best is None or row["validation"]["accuracy"] > best["validation"]["accuracy"]:
                best = dict(epoch=epoch, validation=row["validation"], state=state(model))
            print("CHECK", name, json.dumps(row), flush=True)
            publish(args, dict(phase="training", run=name, epoch=epoch,
                       validation=row["validation"], completed=completed))
        if epoch % 20 == 0 or epoch == args.epochs:
            save(resume, dict(state=state(model), optimizer=optimizer.state_dict(),
                              shuffle=shuffle.get_state(), history=history, best=best))
    final_state = state(model)
    final = {split: evaluate(model, data[index]) for index, split in enumerate(("train", "validation", "test"))}
    model.load_state_dict(best["state"])
    chosen = {split: evaluate(model, data[index]) for index, split in enumerate(("train", "validation", "test"))}
    assert chosen["validation"] == best["validation"]
    result = dict(name=name, method=method, rank=rank, seed=seed, width=model.width, parameters=parameters,
                  epochs=args.epochs, best_epoch=best["epoch"], best=chosen, final=final, history=history)
    save(args.out / "checkpoints" / f"{name}.pt", dict(epoch=args.epochs, state=final_state,
         optimizer=optimizer.state_dict(), shuffle=shuffle.get_state(), best=best, method=method, rank=rank, seed=seed))
    write_json(path, result)
    resume.unlink(missing_ok=True)
    print("RESULT", name, json.dumps({k:v for k,v in result.items() if k != "history"}), flush=True)
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("cifar10","fashion"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    p.add_argument("--ranks", nargs="+", type=int, default=(8, 4, 16, 32))
    p.add_argument("--seeds", nargs="+", type=int, default=(0,))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = p.parse_args()
    torch.set_num_threads(2)
    for folder in (args.out, *(args.out / x for x in ("runs", "checkpoints", "resume"))):
        folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(dataset=args.dataset, ranks=list(args.ranks), seeds=list(args.seeds), methods=list(args.methods), epochs=args.epochs,
               device=args.device, torch=torch.__version__, learning_rate=LR, decay=DECAY, batch=BATCH)
    cfg_file = args.out / "config.json"
    if cfg_file.exists():
        assert json.loads(cfg_file.read_text()) == cfg
    write_json(cfg_file, cfg)
    loader = load_data if args.dataset == "cifar10" else load_fashion
    data, meta = loader(args.data, args.out, args.device)
    if args.dataset == "cifar10":
        assert meta["split_sha256"] == SPLIT
    write_json(args.out / "data_manifest.json", meta)
    print("CONFIG", json.dumps(cfg), flush=True)
    if args.device == "cuda":
        print("GPU", torch.cuda.get_device_name(), flush=True)
    completed = []
    try:
        for rank in args.ranks:
            for seed in args.seeds:
                for method in args.methods:
                    completed.append(train(args, data, method, rank, seed, completed))
    except Exception as error:
        publish(args, dict(phase="error", error=str(error), completed=completed))
        raise
    publish(args, dict(phase="complete", completed=completed))


if __name__ == "__main__":
    main()
