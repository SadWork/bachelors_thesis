"""Train matched Dense MLP and shared-scalar models to validation accuracy plateau."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from deep_model import DeepClassifier
from train import (Condition, MIN_DELTA, MIN_EPOCHS, PATIENCE, SPLIT_SHA256,
                   check, cpu_state, evaluate, load_data, save, snapshot, write_json)

ROOT = Path(__file__).resolve().parent
LR, BATCH, WIDTH, SEED = .003, 512, 160, 0
CONDITIONS = (
    Condition("dense", .01, 0., 0.),
    Condition("no_kl", .01, 0., 0.),
    Condition("both_10", .03, .03, .10),
)


def check_deep():
    for depth in (3, 4):
        dense = DeepClassifier(depth, dense=True)
        shared = DeepClassifier(depth)
        assert sum(p.numel() for p in shared.parameters()) - sum(p.numel() for p in dense.parameters()) == depth * WIDTH
        for layer in shared.hidden:
            assert layer.weight.shape == layer.bias.shape == (WIDTH,)
        tiny = DeepClassifier(depth, inputs=9, width=7)
        opt = tiny.optimizer(LR, .03, .03)
        assert [group["weight_decay"] for group in opt.param_groups] == [.03, .03, 0.]
        assert {id(p) for p in opt.param_groups[0]["params"]} == {id(layer.weight) for layer in tiny.hidden}
        with torch.no_grad():
            for i, layer in enumerate(tiny.hidden):
                layer.logits.normal_(std=i + 1)
        expected = []
        for layer in tiny.hidden:
            p = layer.logits.double().softmax(-1)
            expected.append((p * (p.log() + math.log(p.shape[-1]))).sum(-1).mean())
        torch.testing.assert_close(tiny.layer_kls().mean().double(), torch.stack(expected).mean())
        output = tiny(torch.randn(4, 9))
        assert output.shape == (4, 10)
        output.square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in tiny.parameters())
    print("DEEP_CHECKS_OK depths 3 and 4, scalar shapes, mean KL and optimizer groups", flush=True)


def train_epoch(model, optimizer, split, shuffle, condition):
    x, y = split
    order = torch.randperm(len(x), generator=shuffle).to(x.device)
    ce_sum, correct = x.new_zeros(()), x.new_zeros(())
    model.train()
    for ids in order.split(BATCH):
        optimizer.zero_grad(set_to_none=True)
        logits, target = model(x[ids]), y[ids]
        ce = F.cross_entropy(logits, target)
        loss = ce
        if condition.kl:
            loss = ce + condition.kl * model.layer_kls().mean()
        loss.backward()
        optimizer.step()
        ce_sum += ce.detach() * len(ids)
        correct += (logits.detach().argmax(-1) == target).sum()
    return ce_sum.item() / len(x), correct.item() / len(x)


def run_one(args, data, condition, completed):
    name = f"{condition.name}_d{args.depth}_s0"
    result_path = args.out / "runs" / f"{name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    resume = args.out / "resume" / f"{name}.pt"
    torch.manual_seed(SEED)
    model = DeepClassifier(args.depth, dense=condition.name == "dense").to("cuda")
    optimizer = model.optimizer(LR, condition.hidden_decay, condition.head_decay)
    shuffle = torch.Generator().manual_seed(123000 + SEED)
    history = []
    track = {"best": None, "significant_best": -1., "significant_epoch": 0, "plateau": False}
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        shuffle.set_state(saved["shuffle"])
        history, track = saved["history"], saved["track"]
    start = history[-1]["epoch"] if history else 0
    parameters = sum(p.numel() for p in model.parameters())
    print("START", name, parameters, flush=True)
    epochs = () if track["plateau"] else range(start + 1, args.max_epochs + 1)
    for epoch in epochs:
        ce, accuracy = train_epoch(model, optimizer, data[0], shuffle, condition)
        if not math.isfinite(ce):
            raise FloatingPointError(f"Non-finite CE in {name}, epoch {epoch}")
        entry = {"epoch": epoch, "train_ce": ce, "train_accuracy": accuracy}
        history.append(entry)
        if epoch > 20 and epoch % 5:
            continue
        model.eval()
        entry["validation"] = validation = evaluate(model, data[1], BATCH)
        if not model.dense:
            with torch.no_grad():
                entry["layer_kl"] = model.layer_kls().tolist()
        value = validation["accuracy"]
        if track["best"] is None or value > track["best"]["validation"]["accuracy"]:
            track["best"] = {"epoch": epoch, "validation": validation, "state": cpu_state(model)}
        if value > track["significant_best"] + MIN_DELTA:
            track["significant_best"], track["significant_epoch"] = value, epoch
        values = [e["validation"]["accuracy"] for e in history if "validation" in e]
        gain = mean(values[-3:]) - mean(values[-6:-3]) if len(values) >= 6 else None
        track["plateau"] = (epoch >= MIN_EPOCHS and epoch - track["significant_epoch"] >= PATIENCE
                            and gain is not None and gain <= MIN_DELTA)
        entry["recent_gain"] = gain
        print(json.dumps({"run": name, **entry, "plateau": track["plateau"]}), flush=True)
        save(resume, {"state": cpu_state(model), "optimizer": optimizer.state_dict(),
                      "shuffle": shuffle.get_state(), "history": history, "track": track})
        write_json(args.out / "active_history.json", {"run": name, "history": history})
        snapshot(args.out, {"phase": "training", "run": name, "epoch": epoch,
                            "validation_accuracy": value, "completed": completed})
        if track["plateau"]:
            break
    if not track["plateau"]:
        raise RuntimeError(f"{name} reached {args.max_epochs} epochs before validation plateau; resume saved.")

    final_epoch = history[-1]["epoch"]
    last_state = cpu_state(model)
    train_stop = evaluate(model, data[0], BATCH)
    val_stop = history[-1]["validation"]
    best = track["best"]
    model.load_state_dict(best["state"], strict=True)
    train_best = evaluate(model, data[0], BATCH)
    test_best = evaluate(model, data[2], BATCH)
    result = {"name": name, "condition": condition._asdict(), "depth": args.depth, "seed": SEED,
              "parameters": parameters, "epochs": final_epoch, "best_epoch": best["epoch"],
              "plateau": True, "last_significant_epoch": track["significant_epoch"],
              "train_at_stop": train_stop, "validation_at_stop": val_stop,
              "train_at_best": train_best, "validation_at_best": best["validation"],
              "test_at_best": test_best, "history": history}
    save(args.out / "checkpoints" / f"{name}.pt", {
        "condition": condition._asdict(), "depth": args.depth, "seed": SEED, "epoch": final_epoch,
        "state": best["state"], "last_state": last_state, "best_epoch": best["epoch"],
        "optimizer": optimizer.state_dict(), "shuffle": shuffle.get_state(),
    })
    write_json(result_path, result)
    resume.unlink(missing_ok=True)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(3, 4), required=True)
    parser.add_argument("--max-epochs", type=int, default=800)
    args = parser.parse_args()
    for folder in (args.out, args.out / "runs", args.out / "checkpoints", args.out / "resume"):
        folder.mkdir(parents=True, exist_ok=True)
    source_names = ("train_deep.py", "deep_model.py", "model.py", "cuda_mean.py", "data.py", "train.py", "check.py")
    config = {"depth": args.depth, "width": WIDTH, "seed": SEED, "lr": LR, "batch": BATCH,
              "conditions": [c._asdict() for c in CONDITIONS], "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
              "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}}
    path = args.out / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config
    write_json(path, config)
    print("CONFIG", json.dumps(config), flush=True)
    check("cuda")
    check_deep()
    data, metadata = load_data(args.data, args.out, "cuda")
    assert metadata["split_sha256"] == SPLIT_SHA256
    write_json(args.out / "data_manifest.json", metadata)
    print("DATA", json.dumps(metadata), flush=True)
    completed = []
    try:
        for condition in CONDITIONS:
            result = run_one(args, data, condition, completed)
            completed.append(result["name"])
            snapshot(args.out, {"phase": "training", "completed": completed})
    except Exception as error:
        snapshot(args.out, {"phase": "error", "completed": completed, "error": str(error)})
        raise
    (args.out / "active_history.json").unlink(missing_ok=True)
    snapshot(args.out, {"phase": "complete", "completed": completed})


if __name__ == "__main__":
    main()
