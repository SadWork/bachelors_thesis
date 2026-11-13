"""Two categorical mean layers with one scalar weight and bias per neuron."""
import math

import torch
from torch import nn


def birelu(x):
    return torch.where(x > 0, 2 * x, 0.5 * x)


def row_mean_kl(logits):
    logp = logits.log_softmax(-1)
    return (logp.exp() * (logp + math.log(logits.shape[-1]))).sum(-1).mean()


def mean_reference(x, weight, probability, bias):
    u = x[:, None, :] * weight[None, :, None] + bias[None, :, None]
    return (birelu(u) * probability).sum(-1)


class MeanLayer(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outputs))
        self.logits = nn.Parameter(torch.zeros(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        p = self.logits.softmax(-1)
        if x.is_cuda:
            from cuda_mean import CudaMean
            return CudaMean.apply(x, self.weight, p, self.bias)
        if x.device.type == "mps":
            from metal_mean import MetalMean
            return MetalMean.apply(x, self.weight, p, self.bias)
        return mean_reference(x, self.weight, p, self.bias)


class Classifier(nn.Module):
    def __init__(self, width1=160, width2=160):
        super().__init__()
        self.first = MeanLayer(3072, width1)
        self.second = MeanLayer(width1, width2)
        self.head = nn.Linear(width2, 10)

    def forward(self, x):
        return self.head(self.second(self.first(x)))

    def layer_kls(self):
        return row_mean_kl(self.first.logits), row_mean_kl(self.second.logits)

    def optimizer(self, lr, hidden_decay, head_decay):
        hidden_weights = [self.first.weight, self.second.weight]
        decayed = {id(p) for p in [*hidden_weights, self.head.weight]}
        return torch.optim.AdamW([
            {"params": hidden_weights, "weight_decay": hidden_decay},
            {"params": [self.head.weight], "weight_decay": head_decay},
            {"params": [p for p in self.parameters() if id(p) not in decayed], "weight_decay": 0.0},
        ], lr=lr)
