"""Compare BiReLU placement for the same factorized categorical logits."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from data import load_data
from models import Classifier, routing_parameters
from train import BATCH, DECAY, LR, SPLIT, evaluate, publish, save, state, train_epoch, write_json

RANK, CUT, TARGET = 8, 800, 1200


def score(model, data):
    return {label: evaluate(model, split) for label, split in zip(("train", "validation", "test"), data)}


def score_epoch(model, data, best, epoch):
    last_state = state(model)
    final = score(model, data)
    model.load_state_dict(best["state"], strict=True)
    selected = score(model, data)
    model.load_state_dict(last_state, strict=True)
    return dict(epoch=epoch, best_epoch=best["epoch"], best=selected, final=final)


def fit(args, data, seed, completed):
    name = f"{args.method}_r{RANK}_s{seed}"
    path = args.out / "runs" / f"{name}.json"
    if path.exists():
        assert json.loads(path.read_text())["epochs"] == TARGET
        return name
    torch.manual_seed(seed)
    model = Classifier(args.method, RANK).to(args.device)
    assert sum(p.numel() for p in model.parameters()) == routing_parameters(RANK)
    optimizer = model.optimizer(LR, DECAY)
    shuffle = torch.Generator().manual_seed(123000 + seed)
    history, best, at_cut = [], None, None
    resume = args.out / "resume" / f"{name}.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        history, best, at_cut = saved["history"], saved["best"], saved["at_cut"]
    elif args.method == "logit_pre":
        original = json.loads((args.previous / "runs" / f"{name}.json").read_text())
        saved = torch.load(args.previous / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=True)
        assert saved["epoch"] == original["epochs"] == CUT and saved["seed"] == seed
        assert saved["rank"] == RANK and saved["method"] == "logit_pre"
        assert saved["best"]["epoch"] == original["best_epoch"]
        history, best = original["history"], saved["best"]
        at_cut = {key: original[key] for key in ("best_epoch", "best", "final")}
        at_cut["epoch"] = CUT
    else:
        saved = None
    if saved is not None:
        model.load_state_dict(saved["state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        shuffle.set_state(saved["shuffle"])
        expected = len(history) * math.ceil(len(data[0][0]) / BATCH)
        assert {v["step"].item() for v in optimizer.state.values()} == {expected}
        if len(history) == CUT:
            measured = evaluate(model, data[1])
            expected = at_cut["final"]["validation"]
            assert abs(measured["ce"]-expected["ce"]) <= .0001
            assert abs(measured["accuracy"]-expected["accuracy"]) <= .00021
            print("RESUME_VERIFIED", name, CUT, json.dumps(measured), flush=True)
    start = len(history)
    print("START", name, start, flush=True)
    for epoch in range(start + 1, TARGET + 1):
        started = time.perf_counter()
        tr = train_epoch(model, optimizer, data[0], shuffle)
        if not math.isfinite(tr["ce"]):
            raise FloatingPointError((name, epoch, tr))
        row = dict(epoch=epoch, train=tr, seconds=time.perf_counter() - started)
        history.append(row)
        if epoch <= 20 or epoch % 5 == 0:
            val = row["validation"] = evaluate(model, data[1])
            if best is None or val["accuracy"] > best["validation"]["accuracy"]:
                best = dict(epoch=epoch, validation=val, state=state(model))
            print("CHECK", name, json.dumps(row), flush=True)
            publish(args, dict(phase="training", run=name, epoch=epoch, validation=val, completed=completed))
        if epoch == CUT:
            at_cut = score_epoch(model, data, best, CUT)
            print("AT800", name, json.dumps(at_cut), flush=True)
        if epoch % 20 == 0:
            save(resume, dict(state=state(model), optimizer=optimizer.state_dict(), shuffle=shuffle.get_state(),
                              history=history, best=best, at_cut=at_cut))
    assert len(history) == TARGET and at_cut["epoch"] == CUT
    assert {v["step"].item() for v in optimizer.state.values()} == {TARGET * math.ceil(len(data[0][0]) / BATCH)}
    at_target = score_epoch(model, data, best, TARGET)
    result = dict(name=name, method=args.method, rank=RANK, seed=seed, parameters=routing_parameters(RANK),
                  epochs=TARGET, at_800=at_cut, at_1200=at_target, history=history)
    save(args.out / "checkpoints" / f"{name}.pt", dict(epoch=TARGET, method=args.method, rank=RANK, seed=seed,
         state=state(model), optimizer=optimizer.state_dict(), shuffle=shuffle.get_state(), best=best))
    write_json(path, result)
    resume.unlink(missing_ok=True)
    print("RESULT", name, json.dumps({k:v for k,v in result.items() if k != "history"}), flush=True)
    return name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--method", choices=("logit_pre","logit_post"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--device", choices=("cuda","mps"), required=True)
    args=p.parse_args()
    assert (args.previous is not None) == (args.method == "logit_pre")
    if args.previous:
        assert json.loads((args.previous/"data_manifest.json").read_text())["split_sha256"]==SPLIT
    for folder in (args.out,*(args.out/x for x in ("runs","checkpoints","resume"))):
        folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    config=dict(methods=[args.method],ranks=[RANK],seeds=args.seeds,epochs=TARGET,device=args.device,
                torch=torch.__version__,learning_rate=LR,decay=DECAY,batch=BATCH)
    path=args.out/"config.json"
    if path.exists():
        assert json.loads(path.read_text())==config
    write_json(path,config)
    data,metadata=load_data(args.data,args.out,args.device)
    assert metadata["split_sha256"]==SPLIT
    write_json(args.out/"data_manifest.json",metadata)
    print("CONFIG",json.dumps(config),flush=True)
    if args.device=="cuda":
        print("GPU",torch.cuda.get_device_name(),flush=True)
    completed=[]
    try:
        for seed in args.seeds:
            completed.append(fit(args,data,seed,completed))
    except Exception as error:
        publish(args,dict(phase="error",completed=completed,error=str(error)))
        raise
    publish(args,dict(phase="complete",completed=completed))


if __name__=="__main__":
    main()
