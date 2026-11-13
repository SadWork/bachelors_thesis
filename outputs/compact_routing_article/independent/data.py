"""Load CIFAR-100 with archive verification and a fixed stratified split."""
import hashlib
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch

ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
SPLIT_SEED = 20261027


def load_data(root, out, device):
    archive = Path(root) / "cifar-100-python.tar.gz"
    with archive.open("rb") as f:
        assert hashlib.file_digest(f, "md5").hexdigest() == ARCHIVE_MD5
        f.seek(0)
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    with tarfile.open(archive) as tar:
        def read(name):
            with tar.extractfile(f"cifar-100-python/{name}") as f:
                record = pickle.load(f, encoding="latin1")
                return np.asarray(record["data"], dtype=np.uint8), np.asarray(record["fine_labels"], dtype=np.int64)
        raw, y = read("train")
        tx, ty = read("test")
    assert raw.shape == (50000, 3072) and tx.shape == (10000, 3072)
    assert np.array_equal(np.bincount(y, minlength=100), np.full(100, 500))
    assert np.array_equal(np.bincount(ty, minlength=100), np.full(100, 100))
    rng = np.random.default_rng(SPLIT_SEED)
    parts = [rng.permutation(np.flatnonzero(y == i)) for i in range(100)]
    train = np.concatenate([p[50:] for p in parts]).astype(np.int64)
    val = np.concatenate([p[:50] for p in parts]).astype(np.int64)
    np.savez(out / "split.npz", train=train, validation=val)
    x = raw[train]
    channel_sum, channel_sq = np.zeros(3), np.zeros(3)
    for chunk in np.array_split(x, 9):
        values = chunk.reshape(-1, 3, 1024).astype(np.float64) / 255
        channel_sum += values.sum((0, 2))
        channel_sq += np.square(values).sum((0, 2))
    count = len(train) * 1024
    mean = channel_sum / count
    std = np.sqrt(channel_sq / count - mean**2)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1)

    def tensors(pixels, targets):
        values = torch.as_tensor(pixels, dtype=torch.float32, device=device).reshape(-1, 3, 1024)
        values.div_(255).sub_(center).div_(scale)
        return values.flatten(1), torch.as_tensor(targets, dtype=torch.long, device=device)

    data = tensors(x, y[train]), tensors(raw[val], y[val]), tensors(tx, ty)
    meta = dict(dataset="CIFAR-100", archive_md5=ARCHIVE_MD5, archive_sha256=sha, split_seed=SPLIT_SEED,
                split_sha256=hashlib.sha256(train.tobytes() + val.tobytes()).hexdigest(),
                sizes=[len(pair[0]) for pair in data], channel_mean=mean.tolist(), channel_std=std.tolist(),
                augmentation=False)
    return data, meta
