"""Matched-depth CIFAR-10 models with shared-scalar means or dense hidden layers."""
import torch
from torch import nn

from model import MeanLayer, birelu, row_mean_kl


class DeepClassifier(nn.Module):
    def __init__(self, depth, dense=False, inputs=3072, width=160):
        super().__init__()
        layer = nn.Linear if dense else MeanLayer
        self.dense = dense
        self.hidden = nn.ModuleList(layer(inputs if i == 0 else width, width) for i in range(depth))
        self.head = nn.Linear(width, 10)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
            if self.dense:
                x = birelu(x)
        return self.head(x)

    def layer_kls(self):
        return torch.stack([row_mean_kl(layer.logits) for layer in self.hidden])

    def optimizer(self, lr, hidden_decay, head_decay):
        hidden_weights = [layer.weight for layer in self.hidden]
        decayed = {id(p) for p in [*hidden_weights, self.head.weight]}
        return torch.optim.AdamW([
            {"params": hidden_weights, "weight_decay": hidden_decay},
            {"params": [self.head.weight], "weight_decay": head_decay},
            {"params": [p for p in self.parameters() if id(p) not in decayed], "weight_decay": 0.},
        ], lr=lr)
