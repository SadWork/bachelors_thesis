"""Exact expansion of fixed-support routing with groupwise mass preservation."""
import torch


def groups_for(kind, random_seed=0):
    if kind == "global":
        return torch.zeros(3072, dtype=torch.long)
    channel = torch.arange(3)[:, None]
    if kind == "channel":
        return channel.expand(3, 1024).flatten()
    r, c = torch.arange(32)[:, None], torch.arange(32)[None, :]
    tiles = ((r // 8) * 4 + c // 8).flatten()
    if kind == "tiles":
        labels = tiles.expand(3, -1)
    elif kind == "random_tiles":
        rng = torch.Generator().manual_seed(random_seed)
        labels = torch.stack([tiles[torch.randperm(1024, generator=rng)] for _ in range(3)])
    else:
        raise ValueError(kind)
    return (channel*16 + labels).flatten()


def expand_tail(p, k, group=None):
    n = p.shape[-1]
    assert 0 < k <= n
    if k == n:
        return p
    if group is None:
        group = torch.zeros(n, dtype=torch.long, device=p.device)
    else:
        group = group.to(p.device)
    g = int(group.max().item()) + 1
    top, index = p.topk(k, dim=-1)
    all_groups = group.expand(len(p), -1)
    top_groups = group[index]
    # Summing many small probabilities in FP32 loses mass when the row is sharp.
    mass_source = p.double()
    masses = mass_source.new_zeros((len(p), g)).scatter_add_(1, all_groups, mass_source)
    masses.scatter_add_(1, top_groups, -top.double())
    counts = p.new_zeros((len(p), g)).scatter_add_(1, all_groups, torch.ones_like(p))
    counts.scatter_add_(1, top_groups, -torch.ones_like(top))
    tail = (masses / counts.clamp_min(1)).clamp_min(0).masked_fill_(counts == 0, 0.).to(p.dtype)
    q = tail[:, group].scatter_(1, index, top)
    torch.testing.assert_close(q.sum(-1), p.sum(-1), atol=2e-6, rtol=2e-6)
    return q
