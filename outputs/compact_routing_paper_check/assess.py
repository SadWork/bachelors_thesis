"""Paired accuracy, calibrated CE, and same-device latency for the paper check."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from torch import nn

from data import load_data
from fashion import load_fashion
from metal_mean import MetalMean
from models import Classifier

ROOT = Path(__file__).resolve().parent
MARGIN_PP = 1.0
RESAMPLES = 3000


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def outputs(model, split):
    x, target = split
    logits = torch.cat([model(batch).cpu() for batch in x.split(512)]).numpy().astype(np.float64)
    return logits, target.cpu().numpy()


def temperature_scale(validation_logits, labels):
    centered = validation_logits - validation_logits.max(1, keepdims=True)
    targets = centered[np.arange(len(labels)), labels]
    def gradient(inverse_temperature):
        probability = np.exp(inverse_temperature * centered)
        probability /= probability.sum(1, keepdims=True)
        return float((np.sum(probability * centered, axis=1) - targets).mean())
    low, high = .0001, 100.
    for _ in range(55):
        mid = (low + high) / 2
        if gradient(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def ce(logits, target, scale=1.):
    x = logits * scale
    return float((np.logaddexp.reduce(x, axis=1) - x[np.arange(len(target)), target]).mean())


def paired_bootstrap(differences):
    """Resample the seed and example axes, sharing examples across all models/seeds."""
    rng = np.random.default_rng(37201)
    values = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        seeds = rng.integers(0, differences.shape[0], differences.shape[0])
        examples = rng.integers(0, differences.shape[1], differences.shape[1])
        values[i] = 100 * differences[seeds[:, None], examples].mean()
    return dict(mean_pp=100*float(differences.mean()),
                by_seed_pp=(100*differences.mean(1)).tolist(),
                interval_95_pp=np.quantile(values, [.025,.975]).tolist(),
                practical_margin_pp=MARGIN_PP)


class FrozenLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.register_buffer("p", ((layer.left @ layer.right) / layer.scale).softmax(-1))
        self.register_buffer("w", layer.weight.detach())
        self.register_buffer("b", layer.bias.detach())

    def forward(self, x):
        return MetalMean.apply(x, self.w, self.p, self.b)


class FrozenRouting(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.hidden = nn.Sequential(*(FrozenLayer(layer) for layer in model.hidden))
        self.head = model.head

    def forward(self, x):
        return self.head(self.hidden(x))


@torch.no_grad()
def timing(models, x):
    pre = models["logit_pre",0]
    cached = FrozenRouting(pre).eval()
    torch.testing.assert_close(pre(x[:16]), cached(x[:16]))
    choices = {"full_dense":models["full_dense",0], "logit_pre":pre, "logit_pre_cached":cached}
    times={}
    for batch in (1,512):
        sample=x[:batch]
        for model in choices.values():
            for _ in range(12):
                model(sample)
        torch.mps.synchronize()
        values={name:[] for name in choices}
        order=list(choices)
        for repeat in range(9):
            names=order[repeat%3:]+order[:repeat%3]
            for name in names:
                model=choices[name]
                torch.mps.synchronize()
                begin=time.perf_counter()
                for _ in range(32):
                    model(sample)
                torch.mps.synchronize()
                values[name].append((time.perf_counter()-begin)*1000/32)
        times[str(batch)]={name:median(items) for name,items in values.items()}
    return dict(device="Metal",seed=0,batch_ms=times,
                parameter_bytes={k:sum(p.numel()*p.element_size() for p in m.parameters())
                                 for k,m in choices.items()},
                cached_buffer_bytes=sum(b.numel()*b.element_size() for b in cached.buffers()))


def assess(dataset, data):
    models,correct,rows={},{},[]
    methods=("full_dense","logit_pre","logit_post") if dataset=="cifar10" else (
        "full_dense","factor_dense","logit_pre","logit_post")
    for seed in range(3):
        for method in methods:
            name=f"{method}_r8_s{seed}"
            source=ROOT/"results"/dataset/name
            item=torch.load(source/f"{name}.pt",weights_only=True,map_location="cpu")
            run=read(source/f"{name}.json")
            record=run["at_1200"] if "at_1200" in run else run
            model=Classifier(method,8,inputs=data[0][0].shape[1]).to("mps").eval()
            model.load_state_dict(item["best"]["state"],strict=True)
            vl,vt=outputs(model,data[1])
            tl,tt=outputs(model,data[2])
            scale=temperature_scale(vl,vt)
            ok=tl.argmax(1)==tt
            assert abs(ok.mean()-record["best"]["test"]["accuracy"])<=.00021
            raw_ce=ce(tl,tt)
            assert abs(raw_ce-record["best"]["test"]["ce"])<=.0002
            rows.append(dict(dataset=dataset,method=method,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                             best_epoch=item["best"]["epoch"],validation_accuracy=100*float((vl.argmax(1)==vt).mean()),
                             test_accuracy=100*float(ok.mean()),test_ce=raw_ce,
                             calibrated_test_ce=ce(tl,tt,scale),temperature=1/scale))
            correct[method,seed]=ok.astype(np.int8)
            if dataset=="cifar10" and seed==0:
                models[method,seed]=model
    differences={}
    for control in methods:
        if control=="logit_pre": continue
        d=np.stack([correct["logit_pre",s]-correct[control,s] for s in range(3)])
        differences[control]=paired_bootstrap(d)
    return rows,differences,(timing(models,data[2][0]) if dataset=="cifar10" else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar",type=Path,required=True)
    parser.add_argument("--fashion",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    args=parser.parse_args()
    args.scratch.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    output={}
    for dataset,source,loader in (("cifar10",args.cifar,load_data),("fashion",args.fashion,load_fashion)):
        folder=args.scratch/dataset
        folder.mkdir(exist_ok=True)
        data,meta=loader(source,folder,"mps")
        for path in (ROOT/"results"/dataset).glob("*/data_manifest.json"):
            assert read(path)["split_sha256"]==meta["split_sha256"]
        rows,differences,latency=assess(dataset,data)
        output[dataset]=dict(rows=rows,paired_pre_minus_control=differences,latency=latency)
        del data
        torch.mps.empty_cache()
    (ROOT/"assessment.json").write_text(json.dumps(output,indent=2)+"\n")
    print("Paired comparisons, calibrated CE and latency saved.",flush=True)


if __name__=="__main__":
    main()
