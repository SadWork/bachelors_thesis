"""Matched compact four-hidden-layer CIFAR-10 classifiers."""
import math

import torch
from torch import nn


WIDTH = 160
INPUTS = 3072
DEPTH = 4
METHODS = ("narrow_dense", "factor_dense", "mixture_post", "mixture_pre", "logit_pre", "logit_post", "full_dense")


def birelu(x):
    return torch.where(x > 0, 2 * x, .5 * x)


def routing_parameters(rank, inputs=INPUTS):
    return rank * (inputs + WIDTH + (DEPTH - 1) * 2 * WIDTH) + 2 * DEPTH * WIDTH + 10 * (WIDTH + 1)


def narrow_width(rank, inputs=INPUTS):
    def count(w):
        return inputs*w + (DEPTH - 1)*w*w + (DEPTH + 10)*w + 10
    return min(range(1, WIDTH + 1), key=lambda w: abs(count(w) - routing_parameters(rank, inputs)))


class RoutingLayer(nn.Module):
    def __init__(self, inputs, outputs, rank, method):
        super().__init__()
        self.method = method
        self.weight = nn.Parameter(torch.randn(outputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        self.left = nn.Parameter(torch.randn(outputs, rank) * (.5 if method.startswith("mixture") else .45))
        self.right = nn.Parameter(torch.randn(rank, inputs) * (1. if method.startswith("mixture") else .45))
        self.scale = math.sqrt(rank)

    def forward(self, x):
        if self.method.startswith("mixture"):
            mixture = self.left.softmax(-1)
            atoms = self.right.softmax(-1)
            if self.method == "mixture_post":
                return birelu((x @ atoms.T @ mixture.T) * self.weight + self.bias)
            probability = mixture @ atoms
        else:
            probability = ((self.left @ self.right) / self.scale).softmax(-1)
            if self.method == "logit_post":
                return birelu((x @ probability.T) * self.weight + self.bias)
        if x.is_cuda:
            from cuda_mean import CudaMean
            return CudaMean.apply(x, self.weight, probability, self.bias)
        if x.device.type == "mps":
            from metal_mean import MetalMean
            return MetalMean.apply(x, self.weight, probability, self.bias)
        affine = x[:, None, :] * self.weight[None, :, None] + self.bias[None, :, None]
        return (birelu(affine) * probability).sum(-1)


class FactorDense(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.down = nn.Linear(inputs, rank, bias=False)
        self.up = nn.Linear(rank, outputs)
        with torch.no_grad():
            self.up.weight.mul_(math.sqrt(3))

    def forward(self, x):
        return self.up(self.down(x))


class Classifier(nn.Module):
    def __init__(self, method, rank, inputs=INPUTS, classes=10):
        super().__init__()
        self.method = method
        self.width = narrow_width(rank, inputs) if method == "narrow_dense" else WIDTH
        layers = []
        weights = []
        for index in range(DEPTH):
            layer_inputs = inputs if index == 0 else self.width
            if method in ("narrow_dense", "full_dense"):
                layer = nn.Linear(layer_inputs, self.width)
                weights.append(layer.weight)
            elif method == "factor_dense":
                layer = FactorDense(layer_inputs, self.width, rank)
                weights.extend((layer.down.weight, layer.up.weight))
            else:
                layer = RoutingLayer(layer_inputs, self.width, rank, method)
                weights.append(layer.weight)
            layers.append(layer)
        self.hidden = nn.ModuleList(layers)
        self.head = nn.Linear(self.width, classes)
        self.decayed = weights

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
            if self.method.endswith("dense"):
                x = birelu(x)
        return self.head(x)

    def optimizer(self, lr, decay):
        ids = {id(p) for p in self.decayed}
        return torch.optim.AdamW([
            {"params": self.decayed, "weight_decay": decay},
            {"params": [p for p in self.parameters() if id(p) not in ids], "weight_decay": 0.},
        ], lr=lr)
