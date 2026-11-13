"""Apple CPU benchmark for bucketed exact aggregation of a fixed compressed route."""
import argparse
import ctypes as C
import json
import platform
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "outputs" / "compact_routing_followup"
PAPER = ROOT / "outputs" / "compact_routing_paper_check"
sys.path.insert(0, str(PREVIOUS))

from compiled_probe import (CompiledDense, CompiledRoute, library, load_model, np_float, ptr)
from data import load_data
from models import WIDTH

P = C.c_void_p
I = C.c_int
CHANNELS = 3
INPUTS = 3072


def make_library():
    lib = C.CDLL(str(HERE / "bucket_tail.dylib"))
    lib.optimized_route.argtypes = [P, P, I, I, I] + [P] * 16
    lib.optimized_route.restype = None
    lib.optimized_refit_route.argtypes = [P, P, I, I] + [P] * 19
    lib.optimized_refit_route.restype = None
    for name in ("optimized_dense", "optimized_dense_fp16"):
        fn = getattr(lib, name)
        fn.argtypes = [P, P, I] + [P] * 8
        fn.restype = None
    return lib


def intensity_data(metadata):
    mean = torch.tensor(metadata["channel_mean"], dtype=torch.float32)[:, None]
    sd = torch.tensor(metadata["channel_std"], dtype=torch.float32)[:, None]
    level = (torch.arange(256, dtype=torch.float32)[None, :] / 255.0 - mean) / sd
    norm = torch.cat((sd.flatten()*255, mean.flatten()*255 + .5))
    return mean, sd, np_float(level), np_float(norm)


def intensity_buckets(cuts):
    buckets, lut, cut_index = [], [], []
    for channel_cuts in cuts:
        distinct = np.unique(channel_cuts)
        edges = distinct[(distinct > 0) & (distinct < 256)]
        lookup = np.searchsorted(edges, np.arange(256), side="right")
        cut = np.searchsorted(edges, channel_cuts, side="right")
        cut = np.where(channel_cuts == 256, len(edges) + 1, cut)
        lut.append(lookup)
        cut_index.append(cut)
        buckets.append(len(edges)+1)
    return np.array(buckets, dtype=np.int32), np.array(lut, dtype=np.uint8), np.array(cut_index, dtype=np.uint8)


class Dense:
    """Strong batch GEMM baseline with caller-owned scratch, plus optional half storage."""
    def __init__(self, model, lib, level, half=False):
        self.half = half
        self.level = level
        fn = lib.optimized_dense_fp16 if half else lib.optimized_dense
        self.function = fn
        dtype = np.float16 if half else np.float32
        self.model_arrays = (
            np.ascontiguousarray(torch.cat([m.weight.detach().flatten() for m in model.hidden]).numpy(), dtype=dtype),
            np_float(torch.cat([m.bias for m in model.hidden])),
            np.ascontiguousarray(model.head.weight.detach().numpy(), dtype=dtype),
            np_float(model.head.bias),
        )
        self.stored = sum(a.nbytes for a in self.model_arrays)

    def prepare(self, batch, raw):
        self.output = np.empty((batch, 10), dtype=np.float32)
        self.work = np.empty((2*WIDTH*(1 if self.half else batch),), dtype=np.float32)
        self.normalization = np.empty((batch*INPUTS if raw else 0,), dtype=np.float32)
        self.scratch = self.output.nbytes + self.work.nbytes + self.normalization.nbytes
        self.args = [ptr(self.level), *map(ptr, self.model_arrays), ptr(self.normalization),
                     ptr(self.work), ptr(self.output)]

    def __call__(self, x):
        raw = x.dtype == np.uint8
        self.function(None if raw else ptr(x), ptr(x) if raw else None, len(x), *self.args)
        return self.output[:len(x)]


class BucketRoute:
    def __init__(self, original, model, lib, levels, norm):
        self.function = lib.optimized_route
        self.tile = original.tile
        self.tiles = 16 if self.tile else 1
        assert self.tile in (0, 8)
        self.keep = original.schedule[0]
        assert original.schedule[1:] == (WIDTH, WIDTH, WIDTH)
        if self.tile:
            cuts = original.tile_cuts
        else:
            threshold = np_float(torch.where(model.hidden[0].weight == 0, 0.,
                                             -model.hidden[0].bias/model.hidden[0].weight))
            cuts = np.stack([np.searchsorted(row, threshold) for row in levels])
        bucket_counts, lookup, cut_index = intensity_buckets(cuts)
        self.buckets = bucket_counts.tolist()
        _, values, indices, uniform, weight, bias, head, head_bias = original.arrays
        tail = original.tile_uniform if self.tile else np.repeat(uniform[:WIDTH, None], CHANNELS, axis=1)
        self.arrays = (bucket_counts, lookup, cut_index, tail, norm, levels,
                       values[:self.keep*WIDTH], indices, values[self.keep*WIDTH:],
                       weight, bias, head, head_bias)
        self.stored = sum(a.nbytes for a in self.arrays)

    def prepare(self, batch, raw):
        self.output = np.empty((batch, 10), dtype=np.float32)
        self.moments = np.empty((2*self.tiles*sum(n+1 for n in self.buckets),), dtype=np.float32)
        self.work = np.empty((2*WIDTH,), dtype=np.float32)
        self.scratch = self.output.nbytes + self.moments.nbytes + self.work.nbytes
        self.args = [*map(ptr, self.arrays), ptr(self.moments), ptr(self.work), ptr(self.output)]

    def __call__(self, x):
        raw = x.dtype == np.uint8
        self.function(None if raw else ptr(x), ptr(x) if raw else None, len(x), self.tiles,
                      self.keep, *self.args)
        return self.output[:len(x)]


class RefitRoute:
    def __init__(self, checkpoint, lib, levels, norm, half, hidden_buckets=False):
        self.function = lib.optimized_refit_route
        self.half = half
        layer_data = checkpoint["layers"]
        weights, biases, probabilities, indices, tails = zip(*layer_data)
        keep = np.array([p.shape[-1] for p in probabilities], dtype=np.int32)
        n = (INPUTS, WIDTH, WIDTH, WIDTH)
        uniform = np_float(torch.cat([p / (i-k) for p, i, k in zip(tails, n, keep)]))
        threshold = [torch.where(w == 0, 0., -b/w) for w, b in zip(weights, biases)]
        cuts = np.stack([np.searchsorted(level, np_float(threshold[0])) for level in levels])
        buckets, lut, cut_index = intensity_buckets(cuts)
        self.buckets = buckets.tolist()
        sorted_threshold, row_order = torch.stack(threshold[1:]).sort(dim=-1)
        if hidden_buckets:
            edge_keys = np.arange(4096, dtype=np.uint32) << 20
            edge_bits = np.where(edge_keys >> 31, edge_keys ^ np.uint32(0x80000000), ~edge_keys)
            edges = edge_bits.view(np.float32)
            table = [np.searchsorted(t, edges, side="left") for t in sorted_threshold.numpy()]
            self.hidden_lookup = np.array(table, dtype=np.uint8)
        else:
            self.hidden_lookup = np.empty((0,), dtype=np.uint8)
        pp = torch.cat([p.reshape(-1) for p in probabilities])
        assert torch.equal(pp, pp.half().float()), "Checkpoint should already use fp16-rounded probabilities"
        pp = np.ascontiguousarray(pp.numpy(), dtype=np.float16 if half else np.float32)
        ii = torch.cat([i.reshape(-1) for i in indices])
        assert torch.equal(ii, ii.to(torch.uint16).long())
        ii = ii.numpy().astype(np.uint16)
        self.arrays = (keep, buckets, lut, cut_index, uniform, norm, levels, pp, ii,
                       np_float(torch.cat(weights)), np_float(torch.cat(biases)),
                       np_float(sorted_threshold), row_order.numpy().astype(np.uint8),
                       self.hidden_lookup,
                       *map(np_float, checkpoint["head"]))
        self.stored = sum(a.nbytes for a in self.arrays)

    def prepare(self, batch, raw):
        self.output = np.empty((batch, 10), dtype=np.float32)
        self.moments = np.empty((max(4*WIDTH+2, 2*sum(n+1 for n in self.buckets)),), dtype=np.float32)
        self.work = np.empty((2*WIDTH,), dtype=np.float32)
        self.scratch = self.output.nbytes + self.moments.nbytes + self.work.nbytes
        self.args = [*map(ptr, self.arrays), ptr(self.moments), ptr(self.work), ptr(self.output)]
        if not self.hidden_lookup.size:
            self.args[13] = None

    def __call__(self, x):
        raw = x.dtype == np.uint8
        self.function(None if raw else ptr(x), ptr(x) if raw else None, len(x), self.half, *self.args)
        return self.output[:len(x)]


def benchmark(models, x, batch, rounds):
    names = list(models)
    times = {name: [] for name in names}
    for model in models.values():
        for j in range(5):
            model(x[j*batch:(j+1)*batch])
    for repeat in range(11):
        k = repeat % len(names)
        for name in names[k:] + names[:k]:
            model = models[name]
            batches = [x[((repeat*17+j)*batch) % (len(x)//batch*batch):][:batch] for j in range(rounds)]
            start = time.perf_counter_ns()
            for sample in batches:
                model(sample)
            times[name].append((time.perf_counter_ns()-start)/1e6/rounds)
    return {name: dict(median_ms=median(seq), p10_ms=float(np.quantile(seq, .1)),
                       p90_ms=float(np.quantile(seq, .9)), samples_ms=seq)
            for name, seq in times.items()}


def predictions(model, x, batch):
    return np.concatenate([model(x[i:i+batch]).copy() for i in range(0, len(x), batch)])


def metrics(z, labels):
    return dict(accuracy=float((z.argmax(-1) == labels.numpy()).mean()),
                ce=F.cross_entropy(torch.from_numpy(z), labels).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--out", type=Path, default=HERE / "result.json")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--fidelity-only", action="store_true")
    parser.add_argument("--refit", type=Path)
    parser.add_argument("--refit-seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data, meta = load_data(ROOT / "work/cifar10_shared_scalar/data", HERE, "cpu")
    x, labels = data[1 if args.split == "validation" else 2]
    m, s, level, norm = intensity_data(meta)
    raw = ((x.view(len(x), CHANNELS, -1)*s+m)*255).round().byte().view(len(x), INPUTS).numpy()
    x = np_float(x)
    reconstructed = level[np.arange(3)[None, :, None], raw.reshape(-1, 3, 1024)].reshape(-1, INPUTS)
    np.testing.assert_allclose(reconstructed, x, atol=2e-6, rtol=2e-6)
    old_path = HERE / "compiled_reference.dylib"
    if not old_path.exists():
        old_path = ROOT / "work/compact_routing_followup/compiled_tail.dylib"
    old = library(old_path)
    new = make_library()
    refits = {}
    result = {"platform": platform.platform(), "split": args.split,
              "accuracy": [], "timing": {}, "memory": {}}
    if args.refit:
        sys.path.insert(0, str(PREVIOUS / "sparse_probe"))
        from calibrate import representations
        fitted = torch.load(args.refit, map_location="cpu", weights_only=True)
        refits = {f"refit_fp{16 if half else 32}": RefitRoute(fitted, new, level, norm, half)
                  for half in (False, True)}
        for half in (False, True):
            refits[f"refit_fp{16 if half else 32}_buckets"] = RefitRoute(
                fitted, new, level, norm, half, hidden_buckets=True)
        for model in refits.values():
            model.prepare(128, False)
        if not args.timing_only:
            with torch.no_grad():
                expected = F.linear(representations(torch.from_numpy(x), fitted["layers"])[-1],
                                    *fitted["head"]).numpy()
            for name, model in refits.items():
                predicted = predictions(model, x, 128)
                np.testing.assert_allclose(predicted, expected, atol=1.2e-4, rtol=8e-5)
                raw_predicted = predictions(model, raw, 128)
                np.testing.assert_allclose(raw_predicted, expected, atol=1.2e-4, rtol=8e-5)
                row = dict(seed=args.refit_seed, name=name, **metrics(predicted, labels),
                           max_logit_error=float(np.abs(predicted-expected).max()),
                           max_raw_logit_error=float(np.abs(raw_predicted-expected).max()),
                           changed_predictions_vs_reference=int((predicted.argmax(-1) != expected.argmax(-1)).sum()))
                print("REFIT", row, flush=True)
                result["accuracy"].append(row)
    for seed in (0,) if args.timing_only else range(3):
        teacher, dense = (load_model(PAPER, n, seed) for n in ("logit_pre", "full_dense"))
        dense_old = CompiledDense(dense, old)
        original = {
            "global64": CompiledRoute(teacher, (64, WIDTH, WIDTH, WIDTH), meta, old),
            "global128": CompiledRoute(teacher, (128, WIDTH, WIDTH, WIDTH), meta, old),
            "tile64": CompiledRoute(teacher, (64, WIDTH, WIDTH, WIDTH), meta, old, tile=8),
            "tile96": CompiledRoute(teacher, (96, WIDTH, WIDTH, WIDTH), meta, old, tile=8),
        }
        dense32, dense16 = (Dense(dense, new, level, half) for half in (False, True))
        converted = {name: BucketRoute(model, teacher, new, level, norm) for name, model in original.items()}
        cached = CompiledRoute(teacher, (INPUTS, WIDTH, WIDTH, WIDTH), meta, old)
        for model in [dense32, dense16, *converted.values()]:
            model.prepare(128, True)
        ref_dense = dense_old(x[:128])
        np.testing.assert_allclose(dense32(x[:128]), ref_dense, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(dense32(raw[:128]), ref_dense, atol=4e-5, rtol=4e-5)
        for name, model in converted.items():
            expected = original[name](x[:128])
            np.testing.assert_allclose(model(x[:128]), expected, atol=1.2e-4, rtol=8e-5)
            np.testing.assert_allclose(model(raw[:128]), expected, atol=1.2e-4, rtol=8e-5)
        if not args.timing_only:
            models = {"dense_fp32": dense32, "dense_fp16": dense16, "cached_exact": cached, **converted}
            for name, model in models.items():
                array = x if name == "cached_exact" else raw
                y = predictions(model, array, 128)
                row = dict(seed=seed, name=name, **metrics(y, labels))
                if name in original:
                    expected = predictions(original[name], x, 128)
                    np.testing.assert_allclose(y, expected, atol=1.2e-4, rtol=8e-5)
                    row["max_logit_error"] = float(np.abs(y-expected).max())
                    row["changed_predictions_vs_reference"] = int((y.argmax(-1) != expected.argmax(-1)).sum())
                print("METRIC", row, flush=True)
                result["accuracy"].append(row)
        if seed != 0 or args.fidelity_only:
            continue
        configs = {"dense_fp32": dense32, "dense_fp16": dense16, **converted, **refits}
        for name, model in configs.items():
            if isinstance(model, (BucketRoute, RefitRoute)):
                print("BUCKETS", name, model.buckets, flush=True)
            else:
                print("STORED", name, model.stored, flush=True)
        for name, array in (("normalized_fp32", x), ("raw_uint8", raw)):
            timing, memory = {}, {}
            for batch, repeats in ((1, 320), (128, 12)):
                models = dict(configs)
                if name == "normalized_fp32":
                    models |= {"cached_exact": cached, **{f"{k}_old":v for k,v in original.items()}}
                for model in configs.values():
                    model.prepare(batch, array.dtype == np.uint8)
                measured = benchmark(models, array, batch, repeats)
                timing[str(batch)] = measured
                memory[str(batch)] = {
                    model_name: dict(stored_bytes=model.stored, explicit_scratch_bytes=model.scratch,
                                     normalization_table_bytes=level.nbytes if isinstance(model, Dense) and name == "raw_uint8" else 0)
                    for model_name, model in configs.items()
                }
                print("TIMING", name, batch, {k:round(v["median_ms"],6) for k,v in measured.items()}, flush=True)
            result["timing"][name] = timing
            result["memory"][name] = memory
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
