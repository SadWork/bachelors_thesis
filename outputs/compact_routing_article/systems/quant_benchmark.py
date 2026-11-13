"""Compare compiled routing with Apple BNNS weight-only compressed Dense controls."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import torch

from benchmark import (HERE, ROOT, PAPER, Dense, RefitRoute, benchmark, intensity_data, load_data,
                       load_model, make_library, metrics, np_float, predictions, ptr)


class BnnsDense:
    def __init__(self, model, lib, bytes, all_half=False):
        self.lib = lib
        self.bytes = bytes
        self.all_half = all_half
        layers = (*model.hidden, model.head)
        weights = [np_float(layer.weight) for layer in layers]
        if bytes == 1:
            scale = np.array([np.abs(w).max()/127 for w in weights], np.float32)
            quant = [(w/s).round().clip(-127, 127).astype(np.int8) for w, s in zip(weights, scale)]
        elif bytes == 2:
            scale = np.ones(len(weights), np.float32)
            quant = [w.astype(np.float16) for w in weights]
        else:
            scale = np.ones(len(weights), np.float32)
            quant = weights
        self.scales = scale
        self.arrays = (
            np.concatenate([q.reshape(-1) for q in quant[:-1]]),
            np.concatenate([np_float(layer.bias) for layer in layers[:-1]]).astype(
                np.float16 if all_half else np.float32),
            quant[-1],
            np_float(model.head.bias).astype(np.float16 if all_half else np.float32),
        )
        self.dequantized = [q.astype(np.float32)*s for q,s in zip(quant, scale)]
        self.stored = sum(arr.nbytes for arr in self.arrays) + (scale.nbytes if bytes == 1 else 0)
        self.handle = self.lib.bnns_dense_create(bytes, all_half, *map(ptr,self.arrays), ptr(self.scales))
        if not self.handle:
            raise RuntimeError("BNNS rejected Dense descriptor")

    def prepare(self, batch, raw):
        assert not raw
        self.work = (np.empty((batch*(160*2+3072+10),), dtype=np.float16) if self.all_half
                     else np.empty((batch*160*2,), dtype=np.float32))
        self.output = np.empty((batch,10), dtype=np.float32)
        self.scratch = self.work.nbytes + self.output.nbytes

    def __call__(self, x):
        status = self.lib.bnns_dense_run(self.handle, ptr(x), len(x), ptr(self.work), ptr(self.output))
        if status != 0:
            raise RuntimeError(f"BNNS execution failed: {status}")
        return self.output[:len(x)]

    def close(self):
        self.lib.bnns_dense_free(self.handle)
        self.handle = None


def library():
    p, i = C.c_void_p, C.c_int
    dll = C.CDLL(str(HERE/"quant_dense.dylib"))
    dll.bnns_dense_create.argtypes = [i,i] + [p]*5
    dll.bnns_dense_create.restype = p
    dll.bnns_dense_free.argtypes = [p]
    dll.bnns_dense_free.restype = None
    dll.bnns_dense_run.argtypes = [p,p,i,p,p]
    dll.bnns_dense_run.restype = i
    return dll


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "quant_results.json")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    data, metadata = load_data(ROOT / "work/cifar10_shared_scalar/data", HERE, "cpu")
    x, labels = data[1] if args.timing_only else data[2]
    x = np_float(x)
    _, _, level, norm = intensity_data(metadata)
    lib = library()
    routing_lib = make_library()
    results = {"metrics": [], "failures": {}, "timing": {}, "memory": {}}
    for seed in (0,) if args.timing_only else range(3):
        dense = load_model(PAPER,"full_dense",seed)
        models = {"accelerate_fp32": Dense(dense, routing_lib, level)}
        for bits, all_half in ((8,False),(16,False),(16,True),(32,False)):
            key = f"bnns_{bits}" + ("_all_half" if all_half else "")
            model = None
            try:
                model = BnnsDense(dense, lib, bits//8, all_half)
                model.prepare(128, False)
                layers = (*dense.hidden, dense.head)
                reference = torch.from_numpy(x[:128])
                for i, (w, l) in enumerate(zip(model.dequantized, layers)):
                    if all_half:
                        reference = torch.nn.functional.linear(reference.half(), torch.from_numpy(w).half(), l.bias.half())
                    else:
                        reference = torch.nn.functional.linear(reference, torch.from_numpy(w), l.bias)
                    if i < 4:
                        reference = torch.where(reference > 0, 2*reference, .5*reference)
                np.testing.assert_allclose(model(x[:128]), reference.float().numpy(),
                                           rtol=1.5e-2 if all_half else 1e-4, atol=.1 if all_half else 1e-4)
                del model.dequantized
                models[key] = model
            except (RuntimeError, AssertionError) as error:
                results["failures"][f"{key}_seed{seed}"] = str(error)
                if "model" in locals() and isinstance(model, BnnsDense) and model.handle is not None:
                    model.close()
                print("FAIL",key,seed,str(error),flush=True)
        if seed == 0:
            cp = HERE.parent / "support" / "focus32_topk_refit_ridge_seed0.pt"
            refit = torch.load(cp,weights_only=True,map_location="cpu")
            models["refit_fp16"] = RefitRoute(refit,routing_lib,level,norm,True)
            models["refit_fp16_buckets"] = RefitRoute(refit,routing_lib,level,norm,True,hidden_buckets=True)
        if not args.timing_only:
            for name, model in models.items():
                model.prepare(128,False)
                p = predictions(model,x,128)
                result = dict(seed=seed, name=name, stored_bytes=model.stored, **metrics(p,labels))
                print("METRIC",result,flush=True)
                results["metrics"].append(result)
        if seed == 0:
            for batch, rounds in ((1,360),(128,12)):
                for model in models.values():
                    model.prepare(batch,False)
                measured = benchmark(models,x,batch,rounds)
                results["timing"][str(batch)] = measured
                results["memory"][str(batch)] = {key:dict(stored_bytes=m.stored,scratch_bytes=m.scratch)
                                                 for key,m in models.items()}
                print("TIMING",batch,{k:round(v["median_ms"],6) for k,v in measured.items()},flush=True)
        for model in models.values():
            if isinstance(model,BnnsDense):
                model.close()
    args.out.write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    main()
