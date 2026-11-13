"""Fused CUDA reduction of categorical mean layers with shared scalar weights."""
import torch
import triton
import triton.language as tl


@triton.jit
def mean_forward(X, W, P, BIAS, Y, B: tl.constexpr, N: tl.constexpr,
                 M: tl.constexpr, BN: tl.constexpr, BB: tl.constexpr):
    row = tl.program_id(0)
    examples = tl.program_id(1) * BB + tl.arange(0, BB)
    inputs = tl.arange(0, BN)
    weight = tl.load(W + row)
    probability = tl.load(P + row * N + inputs, inputs < N, 0)
    bias = tl.load(BIAS + row)
    x = tl.load(X + examples[:, None] * N + inputs[None, :],
                (examples[:, None] < B) & (inputs[None, :] < N), 0)
    u = x * weight + bias
    value = tl.where(u > 0, 2 * u, 0.5 * u)
    y = tl.sum(value * probability[None, :], axis=1)
    tl.store(Y + examples * M + row, y, examples < B)


@triton.jit
def grad_parameters(XT, W, P, BIAS, DYT, DW, DP, DB,
                    B: tl.constexpr, N: tl.constexpr, TILES: tl.constexpr,
                    BI: tl.constexpr, BB: tl.constexpr):
    tile = tl.program_id(0)
    inputs = tile * BI + tl.arange(0, BI)
    row = tl.program_id(1)
    examples = tl.arange(0, BB)
    edge = row * N + inputs
    w = tl.load(W + row)
    p = tl.load(P + edge, inputs < N, 0)
    bias = tl.load(BIAS + row)
    x = tl.load(XT + inputs[:, None] * B + examples[None, :],
                (inputs[:, None] < N) & (examples[None, :] < B), 0)
    dy = tl.load(DYT + row * B + examples, examples < B, 0)
    u = w * x + bias
    slope = tl.where(u > 0, 2.0, 0.5)
    du = dy[None, :] * slope
    tl.store(DP + edge, tl.sum(du * u, axis=1), inputs < N)
    partial = row * TILES + tile
    tl.store(DW + partial, tl.sum(p * tl.sum(du * x, axis=1), axis=0))
    tl.store(DB + partial, tl.sum(p * tl.sum(du, axis=1), axis=0))


@triton.jit
def grad_inputs(X, W, PT, BIAS, DY, DX, B: tl.constexpr, N: tl.constexpr,
                M: tl.constexpr, BM: tl.constexpr, BB: tl.constexpr):
    i = tl.program_id(0)
    examples = tl.program_id(1) * BB + tl.arange(0, BB)
    rows = tl.arange(0, BM)
    w = tl.load(W + rows, rows < M, 0)
    p = tl.load(PT + i * M + rows, rows < M, 0)
    bias = tl.load(BIAS + rows, rows < M, 0)
    x = tl.load(X + examples * N + i, examples < B, 0)
    dy = tl.load(DY + examples[:, None] * M + rows[None, :],
                 (examples[:, None] < B) & (rows[None, :] < M), 0)
    u = x[:, None] * w[None, :] + bias[None, :]
    slope = tl.where(u > 0, 2.0, 0.5)
    dx = tl.sum(dy * slope * (w * p)[None, :], axis=1)
    tl.store(DX + examples * N + i, dx, examples < B)


class CudaMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, probability, bias):
        x = x.contiguous()
        batch, inputs = x.shape
        outputs = weight.shape[0]
        out = x.new_empty(batch, outputs)
        block_batch = 2 if inputs > 512 else 8
        mean_forward[(outputs, triton.cdiv(batch, block_batch))](
            x, weight, probability, bias, out, batch, inputs, outputs,
            triton.next_power_of_2(inputs), block_batch, num_warps=8 if inputs > 512 else 4,
        )
        ctx.save_for_backward(x, weight, probability, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, probability, bias = ctx.saved_tensors
        batch, inputs = x.shape
        outputs = weight.shape[0]
        dy = grad_output.contiguous()
        tiles = triton.cdiv(inputs, 16)
        dw = x.new_empty(outputs, tiles)
        db = torch.empty_like(dw)
        dp = torch.empty_like(probability)
        grad_parameters[(tiles, outputs)](
            x.T.contiguous(), weight, probability, bias, dy.T.contiguous(), dw, dp, db,
            batch, inputs, tiles, 16, triton.next_power_of_2(batch), num_warps=8,
        )
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            grad_inputs[(inputs, triton.cdiv(batch, 8))](
                x, weight, probability.T.contiguous(), bias, dy, dx,
                batch, inputs, outputs, triton.next_power_of_2(outputs), 8, num_warps=4,
            )
        return dx, dw.sum(-1), dp, db.sum(-1)
