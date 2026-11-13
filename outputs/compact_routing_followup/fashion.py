"""Official Fashion-MNIST IDX data with a fixed stratified validation split."""
import gzip
import hashlib
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np
import torch

CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
SOURCES = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
           "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/")
SEED = 20261027


def fetch(directory, name):
    path = directory / name
    checksum = CHECKSUMS[name]
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == checksum:
        return path
    errors = []
    for base in SOURCES:
        try:
            with urlopen(base + name, timeout=90) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != checksum:
                raise ValueError(f"Fashion-MNIST checksum mismatch: {name}")
            temp = path.with_suffix(".partial")
            temp.write_bytes(data)
            temp.replace(path)
            return path
        except (OSError, ValueError) as error:
            errors.append(str(error))
    raise RuntimeError(f"Fashion-MNIST download failed for {name}: {errors}")


def read_data(root, prefix, n):
    with gzip.open(fetch(root, prefix+"-images-idx3-ubyte.gz"), "rb") as stream:
        assert struct.unpack(">IIII", stream.read(16)) == (2051, n, 28, 28)
        pixels = np.frombuffer(stream.read(), dtype=np.uint8).reshape(n, 784)
    with gzip.open(fetch(root, prefix+"-labels-idx1-ubyte.gz"), "rb") as stream:
        assert struct.unpack(">II", stream.read(8)) == (2049, n)
        labels = np.frombuffer(stream.read(), dtype=np.uint8).astype(np.int64)
    assert len(labels) == n
    return pixels, labels


def load_fashion(root, out, device):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    x, y = read_data(root, "train", 60000)
    tx, ty = read_data(root, "t10k", 10000)
    rng = np.random.default_rng(SEED)
    train, val = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(y == label))
        val.extend(ids[:500])
        train.extend(ids[500:])
    train, val = np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)
    np.savez(out/"split.npz", train=train, validation=val)
    normalized = x[train].astype(np.float64) / 255.
    mu, sd = float(normalized.mean()), float(normalized.std())
    del normalized

    def tensor(pixels, target):
        xx = torch.tensor(pixels, dtype=torch.float32, device=device)
        return xx.div_(255.).sub_(mu).div_(sd), torch.as_tensor(target, dtype=torch.long, device=device)

    data = (tensor(x[train],y[train]),tensor(x[val],y[val]),tensor(tx,ty))
    metadata = dict(dataset="Fashion-MNIST", source_md5=CHECKSUMS, split_seed=SEED, sizes=[55000,5000,10000],
                    split_sha256=hashlib.sha256(train.tobytes()+val.tobytes()).hexdigest(),
                    channel_mean=[mu],channel_std=[sd],augmentation=False)
    return data,metadata
