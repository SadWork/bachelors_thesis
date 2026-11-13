"""Evaluate compiled static head plus uniform tail and CPU latency on CIFAR-10."""
import argparse
import ctypes
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import load_data
from models import Classifier, birelu, WIDTH

HERE = Path(__file__).resolve().parent
INTEGER = ctypes.c_int
POINTER = ctypes.c_void_p
SCHEDULES = {
    "k128_all": (128, 128, 128, 128),
    "k128_96_128_96": (128, 96, 128, 96),
    "k128_64_128_64": (128, 64, 128, 64),
    "k64_first": (64, 160, 160, 160),
    "k128_first": (128, 160, 160, 160),
    "k64_96_96_96": (64, 96, 96, 96),
    "k96_64_96_64": (96, 64, 96, 64),
}
TILED = {"tiled64_first": (64, 160, 160, 160),
         "tiled96_first": (96, 160, 160, 160),
         "tiled128_first": (128, 160, 160, 160)}
FROZEN_TEST = ("dense", "cached_full", "k64_first", "k128_first", "tiled64_first", "tiled96_first")


def np_float(value):
    return np.ascontiguousarray(value.detach().numpy(), dtype=np.float32)


def ptr(array):
    return POINTER(array.ctypes.data)


def library(path):
    dll = ctypes.CDLL(str(path.resolve()))
    dll.route_infer.argtypes = ([POINTER, INTEGER, INTEGER] + [POINTER]*8
                               + [INTEGER, POINTER, POINTER, INTEGER, POINTER, POINTER, POINTER])
    dll.dense_infer.argtypes = [POINTER, INTEGER, INTEGER] + [POINTER]*5
    dll.route_infer.restype = dll.dense_infer.restype = None
    return dll


class CompiledDense:
    def __init__(self, model, dll):
        self.function = dll.dense_infer
        self.arrays = [np_float(torch.cat([layer.weight.flatten() for layer in model.hidden])),
                       np_float(torch.cat([layer.bias for layer in model.hidden])),
                       np_float(model.head.weight), np_float(model.head.bias)]
        self.bytes = sum(a.nbytes for a in self.arrays)

    def __call__(self, x):
        result = np.empty((len(x), 10), np.float32)
        self.function(ptr(x), len(x), x.shape[1], *map(ptr, self.arrays), ptr(result))
        return result


class CompiledRoute:
    def __init__(self, model, schedule, normalization, dll, histogram=True, tile=0):
        self.model = model
        self.schedule = schedule
        self.function = dll.route_infer
        self.normalization = [np.array(normalization[k], dtype=np.float32)
                              for k in ("channel_mean", "channel_std")]
        values, indices, tails = [], [], []
        self.channels = len(self.normalization[0]) if histogram else 0
        self.tile = tile
        self.tile_uniform = np.empty(0, np.float32)
        self.tile_cuts = np.empty(0, np.int32)
        self.expanded = []
        for i, (layer, k) in enumerate(zip(model.hidden, schedule)):
            p = ((layer.left @ layer.right) / layer.scale).softmax(-1).detach()
            n = p.shape[-1]
            assert 0 < k <= n <= 65536
            if k < n:
                head, ids = p.topk(k, dim=-1)
                if i == 0 and tile:
                    assert n == 3072 and self.channels == 3 and 32 % tile == 0
                    cells = 32 // tile
                    groups = (torch.arange(3)[:, None, None] * cells**2
                              + (torch.arange(32)[None, :, None] // tile) * cells
                              + torch.arange(32)[None, None, :] // tile).reshape(-1)
                    group_ids = groups.expand(len(p), -1)
                    head_group = groups[ids]
                    mass = p.new_zeros(len(p), 3*cells**2).scatter_add_(1, group_ids, p)
                    mass.scatter_add_(1, head_group, -head)
                    count = p.new_full(mass.shape, tile**2).scatter_add_(1, head_group, -torch.ones_like(head))
                    per_block = (mass / count.clamp_min(1)).clamp_min(0)
                    per_block.masked_fill_(count == 0, 0.)
                    self.tile_uniform = np_float(per_block)
                    w, b = layer.weight.detach(), layer.bias.detach()
                    threshold = torch.where(w == 0, 0., -b/w)
                    mu, sd = [torch.from_numpy(value)[:, None] for value in self.normalization]
                    levels = (torch.arange(256, dtype=torch.float32)[None, :] / 255 - mu) / sd
                    self.tile_cuts = torch.searchsorted(levels, threshold.expand(3, -1).contiguous()).numpy().astype(np.int32)
                    expanded = per_block[:, groups]
                    corrected = head - per_block.gather(1, head_group)
                    per_input = torch.zeros(len(p))
                else:
                    per_input = ((1 - head.sum(-1)) / (n - k)).clamp_min(0)
                    corrected = head - per_input[:, None]
                    expanded = per_input[:, None].expand_as(p).clone()
                values.append(corrected.flatten())
                indices.append(ids.numpy().astype(np.uint16).reshape(-1))
                tails.append(per_input)
                self.expanded.append(expanded.scatter_(1, ids, head))
            else:
                values.append(p.flatten())
                tails.append(torch.zeros(len(p)))
                self.expanded.append(p)
        self.arrays = [
            np.array(schedule, dtype=np.int32),
            np_float(torch.cat(values)),
            np.concatenate(indices) if indices else np.empty(0, np.uint16),
            np_float(torch.cat(tails)),
            np_float(torch.cat([layer.weight for layer in model.hidden])),
            np_float(torch.cat([layer.bias for layer in model.hidden])),
            np_float(model.head.weight),
            np_float(model.head.bias),
        ]
        self.bytes = sum(a.nbytes for a in self.arrays) + sum(a.nbytes for a in self.normalization)
        self.bytes += self.tile_uniform.nbytes + self.tile_cuts.nbytes

    def __call__(self, x):
        result = np.empty((len(x), 10), np.float32)
        self.function(ptr(x), len(x), x.shape[1], *map(ptr, self.arrays), self.channels,
                      *map(ptr, self.normalization), self.tile, ptr(self.tile_uniform),
                      ptr(self.tile_cuts), ptr(result))
        return result

    @torch.no_grad()
    def verify(self, x):
        if self.channels:
            mean, sd = [torch.from_numpy(a)[:, None] for a in self.normalization]
            image = x.view(len(x), self.channels, -1)
            reconstructed = ((image*sd+mean).mul(255).round().div(255)-mean)/sd
            torch.testing.assert_close(reconstructed, image, atol=2e-6, rtol=2e-6)
        expected = x
        for layer, p in zip(self.model.hidden, self.expanded):
            z = birelu(expected[:, None, :] * layer.weight[None, :, None] + layer.bias[None, :, None])
            expected = (z * p).sum(-1)
        expected = self.model.head(expected).numpy()
        actual = self(np_float(x))
        np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=8e-5)
        return float(np.max(np.abs(actual-expected)))


def load_model(root, method, seed):
    name = f"{method}_r8_s{seed}"
    model = Classifier(method, 8)
    checkpoint = torch.load(root / "results" / "cifar10" / name / f"{name}.pt",
                            map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["best"]["state"], strict=True)
    return model.eval()


def validation(model, x, target):
    logits = np.concatenate([model(x[start:start+128]) for start in range(0, len(x), 128)])
    accuracy = float((logits.argmax(-1) == target).mean())
    ce = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(target)).item())
    return {"accuracy": accuracy, "ce": ce}


def benchmark(models, x):
    result = {}
    for batch, rounds in ((1, 150), (128, 12)):
        sample = x[:batch]
        names = list(models)
        times = {name: [] for name in names}
        for model in models.values():
            for _ in range(5):
                model(sample)
        for i in range(7):
            for name in names[i % len(names):]+names[:i % len(names)]:
                model = models[name]
                start = time.perf_counter_ns()
                for _ in range(rounds):
                    model(sample)
                times[name].append((time.perf_counter_ns()-start)/1e6/rounds)
        result[str(batch)] = {name: median(values) for name, values in times.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=HERE / "compiled_tail.dylib")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    data, meta = load_data(args.data, args.out, "cpu")
    assert meta["split_sha256"] == "62b8be5987948a6a6cf8b9999fda324a6bc2baa3025f9f832f61d96c74f204f9"
    x, y = data[1 if args.split == "validation" else 2]
    dll = library(args.library)
    models = {}
    rows = []
    for seed in range(3) if not args.timing else (0,):
        pre = load_model(args.reference, "logit_pre", seed)
        choices = {"dense": CompiledDense(load_model(args.reference, "full_dense", seed), dll),
                   "cached_full": CompiledRoute(pre, (3072,160,160,160), meta, dll)}
        if args.timing:
            for key in ("k128_all", "k128_96_128_96", "k128_64_128_64", "k128_first"):
                choices[key] = CompiledRoute(pre, SCHEDULES[key], meta, dll)
            choices["k128_all_sort"] = CompiledRoute(pre, SCHEDULES["k128_all"], meta, dll, histogram=False)
            choices["k128_first_sort"] = CompiledRoute(pre, SCHEDULES["k128_first"], meta, dll, histogram=False)
            choices["tiled64_first"] = CompiledRoute(pre, TILED["tiled64_first"], meta, dll, tile=8)
            choices["tiled96_first"] = CompiledRoute(pre, TILED["tiled96_first"], meta, dll, tile=8)
            for model in choices.values():
                if isinstance(model, CompiledRoute):
                    model.verify(x[:8])
                    del model.expanded, model.model
            times = benchmark(choices, np_float(x))
            (args.out / "timing.json").write_text(json.dumps(times, indent=2))
            print("TIMING", times, flush=True)
            return
        for name, schedule in SCHEDULES.items():
            choices[name] = CompiledRoute(pre, schedule, meta, dll)
        for name, schedule in TILED.items():
            choices[name] = CompiledRoute(pre, schedule, meta, dll, tile=8)
        if args.split == "test":
            choices = {name: choices[name] for name in FROZEN_TEST}
        dense = load_model(args.reference, "full_dense", seed)
        with torch.no_grad():
            np.testing.assert_allclose(choices["dense"](np_float(x[:8])), dense(x[:8]).numpy(),
                                       atol=1e-4, rtol=8e-5)
        for name, model in choices.items():
            error = model.verify(x[:8]) if isinstance(model, CompiledRoute) else None
            metric = validation(model, np_float(x), y.numpy())
            row = {"seed": seed, "method": name, "resident_bytes": model.bytes, args.split: metric,
                   "max_verification_error": error}
            rows.append(row)
            print("VALIDATION", row, flush=True)
    (args.out / (args.split + ".json")).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
