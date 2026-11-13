"""Shared-scalar CIFAR-10 mean networks and a validation-selected KL sweep."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from statistics import mean
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from check import check
from data import load_data
from model import Classifier

ROOT = Path(__file__).resolve().parent
MIN_EPOCHS = 60
PATIENCE = 30
MIN_DELTA = 0.001
STRONGER_KL = (0.10, 0.30)
SPLIT_SHA256 = "62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"


class Condition(NamedTuple):
    name: str
    hidden_decay: float
    head_decay: float
    kl: float


DECAY_CONDITIONS = (
    Condition("baseline", 0.01, 0.00, 0.03),
    Condition("hidden_decay", 0.03, 0.00, 0.03),
    Condition("head_decay", 0.01, 0.03, 0.03),
)
OTHER_CONDITIONS = (
    Condition("no_kl", 0.01, 0.00, 0.00),
    Condition("both_03", 0.03, 0.03, 0.03),
    Condition("both_10", 0.03, 0.03, 0.10),
)


def write_json(path, obj):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def save(path, obj):
    tmp = path.with_suffix(".partial")
    torch.save(obj, tmp)
    tmp.replace(path)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, data, batch):
    x, y = data
    correct = x.new_zeros(())
    ce = x.new_zeros(())
    for xx, yy in zip(x.split(batch), y.split(batch)):
        logits = model(xx)
        correct += (logits.argmax(-1) == yy).sum()
        ce += F.cross_entropy(logits, yy, reduction="sum")
    return {"accuracy": correct.item() / len(x), "ce": ce.item() / len(x)}


def snapshot(out, status):
    write_json(out / "status.json", status)
    tmp = out / "metrics.partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["status.json", "config.json", "data_manifest.json", "split.npz", "selection.json"]:
            if not (out / name).exists():
                continue
            z.write(out / name, name)
        for path in sorted((out / "runs").glob("*.json")):
            z.write(path, path.relative_to(out))
        history = out / "active_history.json"
        if history.exists():
            z.write(history, history.name)
    tmp.replace(out / "metrics.zip")


@torch.no_grad()
def weight_stats(model):
    stats = {}
    for name, layer in (("first", model.first), ("second", model.second), ("head", model.head)):
        stats[name] = {"rms": layer.weight.square().mean().sqrt().item(),
                       "bias_rms": layer.bias.square().mean().sqrt().item()}
    return stats


def train_one(condition, seed, data, args, completed):
    name = f"{condition.name}_s{seed}"
    path = args.out / "runs" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    checkpoint = args.out / "checkpoints" / f"{name}.pt"
    resume = args.out / "resume" / f"{name}.pt"
    x, y = data[0]
    torch.manual_seed(seed)
    model = Classifier(args.width1, args.width2).to(args.device)
    optimizer = model.optimizer(args.lr, condition.hidden_decay, condition.head_decay)
    shuffle = torch.Generator().manual_seed(123000 + seed)
    start = 0
    best, best_epoch, best_state = -1.0, 0, None
    significant_best, significant_epoch = -1.0, 0
    history, previous_seconds = [], 0.0
    plateau = False
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        shuffle.set_state(saved["shuffle_rng"])
        history, start = saved["history"], saved["history"][-1]["epoch"]
        best, best_epoch, best_state = saved["best"], saved["best_epoch"], saved["best_state"]
        significant_best, significant_epoch = saved["significant_best"], saved["significant_epoch"]
        previous_seconds = saved["seconds"]
        plateau = saved["plateau"]
    begin = time.perf_counter()
    epoch = start
    print("START", name, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    remaining_epochs = () if plateau else range(start + 1, args.max_epochs + 1)
    for epoch in remaining_epochs:
        epoch_begin = time.perf_counter()
        order = torch.randperm(len(x), generator=shuffle).to(args.device)
        train_ce, train_correct = x.new_zeros(()), x.new_zeros(())
        model.train()
        for ids in order.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids])
            targets = y[ids]
            ce = F.cross_entropy(logits, targets)
            loss = ce
            if condition.kl:
                first_kl, second_kl = model.layer_kls()
                loss = ce + condition.kl * (first_kl + second_kl) / 2
            loss.backward()
            optimizer.step()
            train_ce += ce.detach() * len(ids)
            train_correct += (logits.detach().argmax(-1) == targets).sum()
        entry = {"epoch": epoch, "train_ce": train_ce.item() / len(x),
                 "train_accuracy": train_correct.item() / len(x),
                 "train_seconds": time.perf_counter() - epoch_begin}
        if not np.isfinite(entry["train_ce"]):
            raise FloatingPointError(f"Non-finite cross-entropy in {name}, epoch {epoch}")
        history.append(entry)
        if epoch > 20 and epoch % 5 != 0 and epoch != args.max_epochs:
            continue
        model.eval()
        entry["validation"] = evaluate(model, data[1], args.batch)
        with torch.no_grad():
            entry["layer_kl"] = [v.item() for v in model.layer_kls()]
        value = entry["validation"]["accuracy"]
        if value > best:
            best, best_epoch, best_state = value, epoch, cpu_state(model)
        if value > significant_best + MIN_DELTA:
            significant_best, significant_epoch = value, epoch
        values = [h["validation"]["accuracy"] for h in history if "validation" in h]
        recent_gain = float(np.mean(values[-3:]) - np.mean(values[-6:-3])) if len(values) >= 6 else None
        plateau = (epoch >= MIN_EPOCHS and epoch - significant_epoch >= PATIENCE
                   and recent_gain is not None and recent_gain <= MIN_DELTA)
        seconds = previous_seconds + time.perf_counter() - begin
        print(json.dumps({"run": name, **entry, "best_epoch": best_epoch,
                          "plateau": plateau, "last_significant_epoch": significant_epoch}), flush=True)
        save(resume, {"model": cpu_state(model), "optimizer": optimizer.state_dict(),
                      "shuffle_rng": shuffle.get_state(), "history": history, "best": best,
                      "best_epoch": best_epoch, "best_state": best_state,
                      "significant_best": significant_best, "significant_epoch": significant_epoch,
                      "seconds": seconds, "plateau": plateau})
        write_json(args.out / "active_history.json", {"run": name, "history": history})
        snapshot(args.out, {"phase": "training", "run": name, "epoch": epoch,
                            "validation_accuracy": value, "completed": completed})
        if plateau:
            break
    if not plateau:
        raise RuntimeError(f"{name} reached the {args.max_epochs}-epoch limit. Resume checkpoint is saved.")

    last_train = evaluate(model, data[0], args.batch)
    last_weights = weight_stats(model)
    model.load_state_dict(best_state)
    best_train = evaluate(model, data[0], args.batch)
    best_validation = evaluate(model, data[1], args.batch)
    best_test = evaluate(model, data[2], args.batch)
    with torch.no_grad():
        best_kl = [v.item() for v in model.layer_kls()]
    result = {"name": name, "variant": condition.name, "condition": condition._asdict(), "seed": seed,
              "parameters": sum(p.numel() for p in model.parameters()),
              "epochs": epoch, "best_epoch": best_epoch, "plateau": True,
              "last_significant_epoch": significant_epoch, "recent_gain": recent_gain,
              "train_at_best": best_train, "validation_at_best": best_validation, "test_at_best": best_test,
              "train_at_stop": last_train, "validation_at_stop": history[-1]["validation"],
              "layer_kl_at_best": best_kl, "weight_stats_at_best": weight_stats(model),
              "weight_stats_at_stop": last_weights, "seconds": seconds, "history": history}
    save(checkpoint, {"state": best_state, "architecture": "shared_scalar", "condition": condition._asdict(),
                      "widths": [args.width1, args.width2]})
    write_json(path, result)
    resume.unlink()
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    return result


def select_decay(by_condition):
    ranking = []
    for condition, runs in by_condition.items():
        ranking.append({
            "condition": condition._asdict(),
            "validation_accuracy": mean(r["validation_at_best"]["accuracy"] for r in runs),
            "validation_ce": mean(r["validation_at_best"]["ce"] for r in runs),
        })
    ranking.sort(key=lambda entry: (-entry["validation_accuracy"], entry["validation_ce"]))
    return Condition(**ranking[0]["condition"]), ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/content"))
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--width1", type=int, default=160)
    parser.add_argument("--width2", type=int, default=160)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=600)
    args = parser.parse_args()
    for path in [args.out, args.out / "runs", args.out / "checkpoints", args.out / "resume"]:
        path.mkdir(parents=True, exist_ok=True)
    cfg = {"widths": [args.width1, args.width2], "input_width": 3072, "batch": args.batch,
           "lr": args.lr, "architecture": "shared_scalar",
           "decay_conditions": [c._asdict() for c in DECAY_CONDITIONS],
           "other_conditions": [c._asdict() for c in OTHER_CONDITIONS],
           "stronger_kl": list(STRONGER_KL), "seeds": args.seeds, "device": args.device,
           "gpu": torch.cuda.get_device_name() if args.device == "cuda" else (
               "Apple M3 Pro (Metal)" if args.device == "mps" else None),
           "torch": torch.__version__, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
           "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ("train.py", "model.py", "data.py", "cuda_mean.py", "check.py",
                                   "metal_mean.py", "mean.metal")}}
    config_path = args.out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Existing experiment configuration differs")
    write_json(config_path, cfg)
    print("CONFIG", json.dumps(cfg), flush=True)
    check(args.device)
    data, meta = load_data(args.data, args.out, args.device)
    assert meta["split_sha256"] == SPLIT_SHA256
    write_json(args.out / "data_manifest.json", meta)
    print("DATA", json.dumps(meta), flush=True)
    completed = []
    try:
        by_condition = {c: [] for c in DECAY_CONDITIONS}
        for seed in args.seeds:
            for condition in DECAY_CONDITIONS:
                result = train_one(condition, seed, data, args, completed)
                by_condition[condition].append(result)
                completed.append(result["name"])
                snapshot(args.out, {"phase": "training", "completed": completed})
        selected, ranking = select_decay(by_condition)
        sweep = [Condition(f"kl_{int(round(kl * 100)):02d}", selected.hidden_decay, selected.head_decay, kl)
                 for kl in STRONGER_KL]
        selection = {"selected": selected._asdict(), "ranking": ranking, "sweep": [c._asdict() for c in sweep]}
        write_json(args.out / "selection.json", selection)
        print("DECAY_SELECTION", json.dumps(selection), flush=True)
        snapshot(args.out, {"phase": "kl_sweep", "completed": completed})
        for seed in args.seeds:
            for condition in (*sweep, *OTHER_CONDITIONS):
                result = train_one(condition, seed, data, args, completed)
                completed.append(result["name"])
                snapshot(args.out, {"phase": "kl_sweep", "completed": completed})
    except Exception as error:
        snapshot(args.out, {"phase": "error", "completed": completed, "error": str(error)})
        raise
    (args.out / "active_history.json").unlink(missing_ok=True)
    snapshot(args.out, {"phase": "complete", "completed": completed})
    print("EXPERIMENT_COMPLETE", len(completed), flush=True)


if __name__ == "__main__":
    main()
