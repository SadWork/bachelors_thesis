"""Evaluate independent routed passes from the learned mixture factors."""
import argparse
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch

from data import fetch
from models import Classifier


def test_set(data, metadata):
    with tarfile.open(fetch(data)) as tar:
        with tar.extractfile("cifar-10-batches-py/test_batch") as stream:
            record = pickle.load(stream, encoding="latin1")
    x = np.asarray(record["data"], dtype=np.float32).reshape(-1, 3, 1024)
    x /= np.float32(255.)
    x -= np.asarray(metadata["channel_mean"], dtype=np.float32)[None, :, None]
    x /= np.asarray(metadata["channel_std"], dtype=np.float32)[None, :, None]
    return x.reshape(len(x), -1), np.asarray(record["labels"])


def cdf(logits):
    value = logits.detach().double().softmax(-1).numpy().cumsum(-1)
    value[:, -1] = 1.
    return value


def sampled_metrics(model, images, labels, draws=16):
    rng = np.random.default_rng(513001)
    layers = [(cdf(layer.left), cdf(layer.right), layer.weight.detach().numpy(), layer.bias.detach().numpy())
              for layer in model.hidden]
    head = model.head.weight.detach().numpy()
    head_bias = model.head.bias.detach().numpy()
    totals = {count: np.zeros(2) for count in (1, 4, 8)}
    for start in range(0, len(images), 128):
        x = np.repeat(images[start:start+128], draws, axis=0)
        target = labels[start:start+128]
        for mixture, atoms, weight, bias in layers:
            u = rng.random((len(x), len(weight)))
            component = (u[..., None] >= mixture[None, ...]).sum(-1)
            u = rng.random(component.shape)
            index = np.empty(component.shape, dtype=np.int64)
            for r, distribution in enumerate(atoms):
                mask = component == r
                index[mask] = np.searchsorted(distribution, u[mask], side="right")
            z = np.take_along_axis(x, index, axis=1) * weight + bias
            x = np.where(z > 0, 2 * z, .5 * z)
        logits = (x @ head.T + head_bias).reshape(len(target), draws, 10)
        for count, total in totals.items():
            groups = draws // count
            z = logits.reshape(len(target), groups, count, 10).mean(2)
            chosen = np.take_along_axis(z, np.broadcast_to(target[:, None, None], z.shape[:2] + (1,)), axis=-1)
            total[0] += (z.argmax(-1) == target[:, None]).sum() / groups
            total[1] += (np.logaddexp.reduce(z, axis=-1) - chosen[..., 0]).sum() / groups
    return {str(k): {"accuracy": v[0]/len(images), "ce": v[1]/len(images)} for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("checkpoints", nargs="+", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    images, labels = test_set(args.data, json.loads(args.metadata.read_text()))
    results = {}
    for path in args.checkpoints:
        item = torch.load(path, map_location="cpu", weights_only=True)
        assert item["method"] == "mixture_pre" and item["rank"] == 8
        model = Classifier(item["method"], item["rank"])
        model.load_state_dict(item["best"]["state"])
        result = sampled_metrics(model, images, labels)
        results[path.stem] = {"seed": item["seed"], "epoch": item["best"]["epoch"], "metrics": result}
        print(path.stem, json.dumps(result), flush=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
