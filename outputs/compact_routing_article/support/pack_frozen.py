"""Pack the selected top-K refit model as contiguous, typed deployment buffers."""
from collections import defaultdict
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from activation_support import HERE, SEEDS, representations, validation_data, write_json
from probe import budget

FLOAT32, FLOAT16, UINT16 = "<f4", "<f2", "<u2"


def field_values(payload):
    for layer, (w, b, p, ids, tail) in enumerate(payload["layers"]):
        yield layer, "weight", w, FLOAT32
        yield layer, "bias", b, FLOAT32
        yield layer, "probability", p, FLOAT16
        yield layer, "index", ids, UINT16
        yield layer, "tail_mass", tail, FLOAT32
    yield -1, "head_weight", payload["head"][0], FLOAT32
    yield -1, "head_bias", payload["head"][1], FLOAT32


def decode(raw, fields):
    pieces = defaultdict(dict)
    for field in fields:
        view = np.frombuffer(raw, dtype=field["dtype"], offset=field["offset"], count=field["elements"])
        view = view.reshape(field["shape"]).copy()
        assert view.nbytes == field["bytes"]
        tensor = torch.from_numpy(view)
        tensor = tensor.long() if field["dtype"] == UINT16 else tensor.float()
        pieces[field["layer"]][field["name"]] = tensor
    layers = []
    for idx in range(4):
        row = pieces[idx]
        layers.append(tuple(row[k] for k in ("weight", "bias", "probability", "index", "tail_mass")))
    head = pieces[-1]["head_weight"], pieces[-1]["head_bias"]
    return {"layers": layers, "head": head}


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    out = HERE / "packed"
    out.mkdir(exist_ok=True)
    frozen = json.loads((HERE / "frozen.json").read_text())
    schedule = frozen["selected"]["schedule"]
    assert schedule == "focus32"
    x, _ = validation_data("cifar10")
    x = x[:128]
    rows = []
    for seed in SEEDS:
        name = f"{schedule}_topk_refit_ridge_seed{seed}"
        payload = torch.load(HERE / f"{name}.pt", weights_only=True, map_location="cpu")
        fields, offset, data = [], 0, []
        for layer, key, tensor, dtype in field_values(payload):
            value = np.ascontiguousarray(tensor.numpy(), dtype=dtype)
            fields.append({"layer": layer, "name": key, "dtype": dtype, "shape": list(value.shape),
                           "elements": value.size, "bytes": value.nbytes, "offset": offset})
            offset += value.nbytes
            data.append(value.tobytes())
        raw = b"".join(data)
        ks = [t[2].size(1) for t in payload["layers"]]
        expected = budget(3072, ks, "uniform_tail")["resident_bytes_with_affine_and_head"]
        assert offset == len(raw) == expected == 137000
        path = out / f"{name}.bin"
        path.write_bytes(raw)
        data_read = path.read_bytes()
        unpacked = decode(data_read, fields)
        for (l1, k1, t1, _), (l2, k2, t2, _) in zip(field_values(payload), field_values(unpacked)):
            assert (l1, k1) == (l2, k2)
            torch.testing.assert_close(t1, t2, rtol=0, atol=0)
        before = F.linear(representations(x, payload["layers"])[-1], *payload["head"])
        after = F.linear(representations(x, unpacked["layers"])[-1], *unpacked["head"])
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        row = {"seed": seed, "payload_file": path.name, "payload_bytes": len(raw),
               "sha256": hashlib.sha256(data_read).hexdigest(), "ks": ks,
               "metadata_external": True, "fields": fields, "verified_prediction_rows": len(x),
               "max_logit_error": (before - after).abs().max().item()}
        metadata = out / f"{name}.json"
        write_json(metadata, row)
        rows.append({"seed": seed, "payload_bytes": len(raw), "metadata_json_bytes": metadata.stat().st_size,
                     "max_logit_error": row["max_logit_error"]})
        print(json.dumps(rows[-1]), flush=True)
    write_json(out / "summary.json", rows)


if __name__ == "__main__":
    main()
