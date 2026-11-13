"""Validate and collect the fresh runs and reused CIFAR-10 controls."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from models import Classifier

ROOT=Path(__file__).resolve().parent
SPLIT="62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"


def read(path):
    return json.loads(path.read_text())


def copy(source,dest,files):
    dest.mkdir(parents=True,exist_ok=True)
    for name in files:
        shutil.copyfile(source/name,dest/Path(name).name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    origins={}
    parts=(("cifar10","mac_cifar",1200),("cifar10","mac_cifar_s1",1200),
           ("fashion","mac_fashion",500),("fashion","mac_fashion_s12",500))
    hashes={}
    for dataset,folder,epochs in parts:
        src=args.work/folder
        config,status,meta=(read(src/name) for name in ("config.json","status.json","data_manifest.json"))
        assert config["dataset"]==dataset and config["epochs"]==epochs and config["device"]=="mps"
        assert status["phase"]=="complete" and not (hashes.setdefault(dataset,meta["split_sha256"])!=meta["split_sha256"])
        expected={f"{m}_r8_s{s}" for s in config["seeds"] for m in config["methods"]}
        assert set(status["completed"])==expected
        for name in sorted(expected):
            run=read(src/"runs"/f"{name}.json")
            item=torch.load(src/"checkpoints"/f"{name}.pt",weights_only=True,map_location="cpu")
            assert run["epochs"]==item["epoch"]==epochs and run["name"]==name
            assert [e["epoch"] for e in run["history"]]==list(range(1,epochs+1))
            best=max((e for e in run["history"] if "validation" in e),key=lambda e:e["validation"]["accuracy"])
            assert best["epoch"]==run["best_epoch"]==item["best"]["epoch"]
            assert best["validation"]==run["best"]["validation"]
            assert {v["step"].item() for v in item["optimizer"]["state"].values()}=={epochs*(88 if dataset=="cifar10" else 108)}
            inputs=3072 if dataset=="cifar10" else 784
            model=Classifier(run["method"],8,inputs)
            for state in (item["state"],item["best"]["state"]):
                model.load_state_dict(state,strict=True)
                assert torch.isfinite(model(torch.zeros(2,inputs))).all()
            assert sum(p.numel() for p in model.parameters())==run["parameters"]
            key=f"{dataset}/{name}"
            assert key not in origins
            origins[key]=dict(source="fresh",device=config["device"],epochs=epochs)
            copy(src,ROOT/"results"/key,("config.json","data_manifest.json",f"runs/{name}.json",f"checkpoints/{name}.pt"))
    assert hashes["cifar10"]==SPLIT
    previous=ROOT.parent/"cifar10_logit_placement_1200"/"results"
    for method in ("logit_pre","logit_post"):
        for seed in range(3):
            name=f"{method}_r8_s{seed}"
            src=previous/name
            meta=read(src/"data_manifest.json")
            config=read(src/"config.json")
            assert meta["split_sha256"]==SPLIT and config["epochs"]==1200
            key=f"cifar10/{name}"
            assert key not in origins
            origins[key]=dict(source="placement_1200",device=config["device"],epochs=1200)
            copy(src,ROOT/"results"/key,("config.json","data_manifest.json",f"{name}.json",f"{name}.pt"))
    expected={f"cifar10/{m}_r8_s{s}" for m in ("full_dense","logit_pre","logit_post") for s in range(3)}
    expected|={f"fashion/{m}_r8_s{s}" for m in ("full_dense","factor_dense","logit_pre","logit_post") for s in range(3)}
    assert set(origins)==expected
    (ROOT/"sources.json").write_text(json.dumps(origins,indent=2)+"\n")
    print("Validated and collected 15 fresh runs and 6 reused runs.")


if __name__=="__main__":
    main()
