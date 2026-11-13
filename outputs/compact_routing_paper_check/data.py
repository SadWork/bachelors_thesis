"""Verified CIFAR-10 archive and the split used in Bidirectional NN."""
import hashlib
import pickle
import tarfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
URLS = (
    "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(root):
    archive = Path(root) / "cifar-10-python.tar.gz"
    if archive.exists() and digest(archive) == SHA256:
        return archive
    errors = []
    for url in URLS:
        temporary = archive.with_suffix(".download")
        try:
            with urlopen(url, timeout=120) as source, temporary.open("wb") as out:
                while block := source.read(1024 * 1024):
                    out.write(block)
            if digest(temporary) != SHA256:
                raise ValueError("CIFAR-10 SHA256 mismatch")
            temporary.replace(archive)
            return archive
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("CIFAR-10 download failed. " + " | ".join(errors))


def load_data(root, out, device):
    archive = fetch(root)
    with tarfile.open(archive) as tar:
        def read(name):
            with tar.extractfile("cifar-10-batches-py/" + name) as file:
                record = pickle.load(file, encoding="latin1")
            return np.asarray(record["data"], dtype=np.uint8), np.asarray(record["labels"], dtype=np.int64)
        batches = [read(f"data_batch_{i}") for i in range(1, 6)]
        raw = np.concatenate([x for x, _ in batches])
        labels = np.concatenate([y for _, y in batches])
        test, test_labels = read("test_batch")
    rng = np.random.default_rng(20261027)
    train_ids, val_ids = [], []
    for category in range(10):
        ids = rng.permutation(np.flatnonzero(labels == category))
        val_ids.extend(ids[:500])
        train_ids.extend(ids[500:])
    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)
    np.savez(out / "split.npz", train=train_ids, validation=val_ids)
    train = raw[train_ids]
    channel_sum, channel_sq = np.zeros(3), np.zeros(3)
    for chunk in np.array_split(train, 9):
        channel_values = chunk.reshape(-1, 3, 1024).astype(np.float64) / 255
        channel_sum += channel_values.sum((0, 2))
        channel_sq += np.square(channel_values).sum((0, 2))
    n = len(train) * 1024
    mean = channel_sum / n
    std = np.sqrt(channel_sq / n - mean**2)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1)

    def tensors(pixels, targets):
        x = torch.as_tensor(pixels, dtype=torch.float32, device=device).reshape(-1, 3, 1024)
        x.div_(255).sub_(center).div_(scale)
        return x.flatten(1), torch.as_tensor(targets, dtype=torch.long, device=device)

    data = (tensors(train, labels[train_ids]), tensors(raw[val_ids], labels[val_ids]),
            tensors(test, test_labels))
    meta = dict(dataset="CIFAR-10", flatten_order="CHW", archive_sha256=SHA256, split_seed=20261027,
                split_sha256=hashlib.sha256(train_ids.tobytes() + val_ids.tobytes()).hexdigest(),
                sizes=[len(pair[0]) for pair in data], channel_mean=mean.tolist(),
                channel_std=std.tolist(), augmentation=False)
    return data, meta
