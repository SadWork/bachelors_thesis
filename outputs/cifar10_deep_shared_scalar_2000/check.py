"""Checks for KL normalization, optimizer groups, and the CUDA forward/backward."""
import math
import torch

from model import Classifier, MeanLayer, birelu, mean_reference, row_mean_kl


def check(device):
    torch.manual_seed(19)
    model = Classifier(7, 11)
    for layer, inputs, outputs in ((model.first, 3072, 7), (model.second, 7, 11)):
        assert layer.weight.shape == layer.bias.shape == (outputs,)
        assert layer.logits.shape == (outputs, inputs)
        assert sum(p.numel() for p in layer.parameters()) == outputs * (inputs + 2)
    full = Classifier()
    assert sum(p.numel() for p in full.parameters()) == 519370
    tiny = MeanLayer(4, 3).to(device)
    x = torch.tensor([[-2., -1., 0., 1.], [2., 3., 4., 5.]], device=device)
    with torch.no_grad():
        tiny.weight.copy_(x.new_tensor([2., -1., 0.]))
        tiny.bias.copy_(x.new_tensor([0., 1., -.5]))
        tiny.logits.copy_(torch.randn_like(tiny.logits))
        manual = torch.stack([
            sum(p[i] * birelu(w * x[:, i] + b) for i in range(x.shape[1]))
            for w, b, p in zip(tiny.weight, tiny.bias, tiny.logits.softmax(-1))
        ], dim=1)
        torch.testing.assert_close(tiny(x), manual)
    opt = model.optimizer(0.003, hidden_decay=0.01, head_decay=0.03)
    assert [group["weight_decay"] for group in opt.param_groups] == [0.01, 0.03, 0]
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(model.first.weight), id(model.second.weight)]
    assert [id(p) for p in opt.param_groups[1]["params"]] == [id(model.head.weight)]
    assert {id(p) for p in opt.param_groups[2]["params"]} == {
        id(model.first.bias), id(model.second.bias), id(model.head.bias),
        id(model.first.logits), id(model.second.logits),
    }
    with torch.no_grad():
        model.first.logits.copy_(torch.randn_like(model.first.logits))
        model.second.logits.copy_(torch.randn_like(model.second.logits) * 2)
    first, second = model.layer_kls()
    def reference(logits):
        p = logits.double().softmax(-1)
        return (p * p.log()).sum(-1).mean() + math.log(p.shape[1])
    expected = 0.03 * (reference(model.first.logits) + reference(model.second.logits)) / 2
    torch.testing.assert_close((0.03 * (first + second) / 2).double(), expected)
    torch.testing.assert_close(row_mean_kl(torch.zeros(13, 3072)), torch.zeros(()), atol=1e-6, rtol=0)

    fused = None
    if device == "cuda":
        from cuda_mean import CudaMean
        fused = CudaMean
    if device == "mps":
        from metal_mean import MetalMean
        fused = MetalMean
    if fused is not None:
        for batch, n, m in [(9, 17, 11), (17, 3072, 160), (27, 160, 160)]:
            x = torch.randn(batch, n, device=device)
            w = torch.randn(m, device=device)
            p = torch.randn(m, n, device=device).softmax(-1)
            b = torch.randn(m, device=device)
            params = tuple(t.requires_grad_() for t in (x, w, p, b))
            a, ref = fused.apply(*params), mean_reference(*params)
            torch.testing.assert_close(a, ref, atol=2e-5, rtol=2e-5)
            upstream = torch.randn_like(ref)
            for got, want in zip(torch.autograd.grad(a, params, upstream), torch.autograd.grad(ref, params, upstream)):
                torch.testing.assert_close(got, want, atol=5e-5, rtol=5e-5)
    print("CHECKS_OK scalar parameters and routing, normalized KL, AdamW groups, and fused gradients", flush=True)
