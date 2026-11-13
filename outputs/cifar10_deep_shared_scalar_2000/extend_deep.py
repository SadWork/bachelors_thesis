"""Resume the shared-scalar depth runs with AdamW through epoch 2000."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from deep_model import DeepClassifier
from train import Condition, SPLIT_SHA256, check, cpu_state, evaluate, load_data, save, snapshot, write_json
from train_deep import BATCH, LR, check_deep, train_epoch

TARGET = 2000
UPDATES_PER_EPOCH = math.ceil(45000 / BATCH)
CONDITIONS = {"no_kl": Condition("no_kl", .01, 0., 0.),
              "both_10": Condition("both_10", .03, .03, .10)}


def candidate(model, epoch, validation):
    return {"epoch": epoch, "validation": validation, "state": cpu_state(model)}


def score_candidate(model, item, data):
    model.load_state_dict(item["state"], strict=True)
    return {"epoch": item["epoch"], "validation": item["validation"],
            "train": evaluate(model, data[0], BATCH), "test": evaluate(model, data[2], BATCH)}


def verify_steps(optimizer, epoch):
    assert {entry["step"].item() for entry in optimizer.state.values()} == {epoch * UPDATES_PER_EPOCH}


def run(args, data, depth, job, completed):
    condition = Condition(**job["condition"])
    assert condition == CONDITIONS[condition.name]
    name = f"{condition.name}_d{depth}_s0"
    result_path = args.out / "runs" / f"{name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    checkpoint = args.source / "checkpoints" / f"{name}.pt"
    with checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == job["checkpoint_sha256"]
    initial = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert initial["epoch"] == job["start_epoch"] and initial["seed"] == 0 and initial["depth"] == depth
    assert initial["condition"] == job["condition"]

    model = DeepClassifier(depth).to(args.device)
    optimizer = model.optimizer(LR, condition.hidden_decay, condition.head_decay)
    shuffle = torch.Generator()
    resume = args.out / "resume" / f"{name}.pt"
    saved = torch.load(resume, weights_only=True, map_location="cpu") if resume.exists() else initial
    model.load_state_dict(saved["last_state"] if saved is initial else saved["state"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    shuffle.set_state(saved["shuffle"])
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        condition.hidden_decay, condition.head_decay, 0.,
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [LR] * 3
    verify_steps(optimizer, saved["epoch"])
    if saved is initial:
        model.eval()
        actual = evaluate(model, data[1], BATCH)
        expected = job["validation_at_start"]
        if args.device == "cuda":
            assert actual == expected, (actual, expected)
        else:
            assert abs(actual["ce"] - expected["ce"]) <= .0001, (actual, expected)
            assert abs(actual["accuracy"] - expected["accuracy"]) <= .00021, (actual, expected)
        start_validation = actual
        print("RESUME_VERIFIED", name, saved["epoch"], json.dumps(actual), flush=True)
    else:
        start_validation = saved["start_validation"]

    history = saved.get("history", [])
    best = saved.get("best", {"ce": None, "accuracy": None})
    for epoch in range(saved["epoch"] + 1, TARGET + 1):
        ce, accuracy = train_epoch(model, optimizer, data[0], shuffle, condition)
        if not math.isfinite(ce):
            raise FloatingPointError(f"Non-finite CE in {name} at epoch {epoch}")
        entry = {"epoch": epoch, "train_ce": ce, "train_accuracy": accuracy}
        history.append(entry)
        if epoch % 5:
            continue
        model.eval()
        validation = evaluate(model, data[1], BATCH)
        entry["validation"] = validation
        with torch.no_grad():
            entry["layer_kl"] = model.layer_kls().tolist()
        if best["ce"] is None or validation["ce"] < best["ce"]["validation"]["ce"]:
            best["ce"] = candidate(model, epoch, validation)
        if best["accuracy"] is None or validation["accuracy"] > best["accuracy"]["validation"]["accuracy"]:
            best["accuracy"] = candidate(model, epoch, validation)
        print(json.dumps({"run": name, **entry}), flush=True)
        save(resume, {"epoch": epoch, "state": cpu_state(model), "optimizer": optimizer.state_dict(),
                      "shuffle": shuffle.get_state(), "history": history, "best": best,
                      "start_validation": start_validation})
        write_json(args.out / "active_history.json", {"run": name, "history": history})
        snapshot(args.out, {"phase": "continuation", "run": name, "epoch": epoch,
                            "validation_accuracy": validation["accuracy"], "validation_ce": validation["ce"],
                            "completed": completed})

    assert [entry["epoch"] for entry in history] == list(range(initial["epoch"] + 1, TARGET + 1))
    verify_steps(optimizer, TARGET)
    last = candidate(model, TARGET, history[-1]["validation"])
    states = {"best_ce": best["ce"], "best_accuracy": best["accuracy"], "last": last}
    metrics = {key: score_candidate(model, item, data) for key, item in states.items()}
    result = {"name": name, "depth": depth, "seed": 0, "condition": condition._asdict(),
              "start_epoch": initial["epoch"], "start_validation": start_validation,
              "epochs": TARGET, "history": history, **metrics}
    save(args.out / "checkpoints" / f"{name}.pt", {
        "architecture": "shared_scalar", "condition": condition._asdict(), "depth": depth,
        "seed": 0, "epoch": TARGET, "state": last["state"], "optimizer": optimizer.state_dict(),
        "shuffle": shuffle.get_state(), "best_ce": best["ce"], "best_accuracy": best["accuracy"],
    })
    write_json(result_path, result)
    resume.unlink(missing_ok=True)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    args = parser.parse_args()
    spec = json.loads(args.jobs.read_text())
    depth = spec["depth"]
    assert depth in (3, 4)
    assert [job["condition"]["name"] for job in spec["jobs"]] == ["no_kl", "both_10"]
    for folder in (args.out, args.out / "runs", args.out / "checkpoints", args.out / "resume"):
        folder.mkdir(parents=True, exist_ok=True)
    config = {"depth": depth, "seed": 0, "target_epoch": TARGET, "lr": LR, "batch": BATCH,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "Apple Metal",
              "device": args.device, "torch": torch.__version__, "jobs": spec["jobs"]}
    if args.device == "cuda":
        assert config["gpu"] == spec["gpu"] and config["torch"] == spec["torch"]
    config_path = args.out / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    write_json(config_path, config)
    print("CONFIG", json.dumps(config), flush=True)
    check(args.device)
    check_deep()
    data, metadata = load_data(args.data, args.out, args.device)
    assert metadata["split_sha256"] == SPLIT_SHA256 == spec["split_sha256"]
    write_json(args.out / "data_manifest.json", metadata)
    print("DATA", json.dumps(metadata), flush=True)
    completed = []
    try:
        for job in spec["jobs"]:
            result = run(args, data, depth, job, completed)
            completed.append(result["name"])
            snapshot(args.out, {"phase": "continuation", "completed": completed})
    except Exception as error:
        snapshot(args.out, {"phase": "error", "completed": completed, "error": str(error)})
        raise
    (args.out / "active_history.json").unlink(missing_ok=True)
    snapshot(args.out, {"phase": "complete", "completed": completed})


if __name__ == "__main__":
    main()
