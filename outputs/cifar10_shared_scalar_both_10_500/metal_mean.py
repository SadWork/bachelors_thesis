"""Apple GPU implementation of the exact categorical mean and its gradients."""
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def kernels():
    return torch.mps.compile_shader(Path(__file__).with_name("mean.metal").read_text())


class MetalMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, probability, bias):
        x = x.contiguous()
        batch, inputs = x.shape
        outputs = weight.numel()
        out = x.new_empty(batch, outputs)
        kernels().mean_forward(
            x, weight, probability, bias, out, inputs, outputs,
            threads=batch * outputs * 32, group_size=128,
        )
        ctx.save_for_backward(x, weight, probability, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, probability, bias = ctx.saved_tensors
        batch, inputs = x.shape
        outputs = weight.numel()
        dy = grad_output.contiguous()
        dw, dp, db = (torch.empty_like(probability) for _ in range(3))
        kernels().mean_grad_parameters(
            x, weight, probability, bias, dy, dw, dp, db, inputs, outputs, batch,
            threads=inputs * outputs, group_size=128,
        )
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            kernels().mean_grad_inputs(
                x, weight, probability.T.contiguous(), bias, dy, dx, inputs, outputs,
                threads=batch * inputs * 32, group_size=128,
            )
        return dx, dw.sum(-1), dp, db.sum(-1)
