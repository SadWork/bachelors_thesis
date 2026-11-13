"""Sparse stochastic linear MNIST experiment. Python 3.11+, torch, numpy.

All training forwards gather only K input features per class. Routing is an
input-independent categorical tree. REINFORCE uses an independent counterfactual
class-logit baseline; no dense surrogate or pretrained teacher is used.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import struct
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_data(root, out):
    arrays, hashes = {}, {}
    for name in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
        p = root / name
        raw = p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())
        magic, n = struct.unpack('>II', raw[:8])
        assert magic in (2049, 2051)
        offset = 16 if magic == 2051 else 8
        arrays[name] = torch.from_numpy(np.frombuffer(raw[offset:], dtype=np.uint8).copy()).reshape(n, -1)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    x = arrays['train-images-idx3-ubyte'].float() / 255
    y = arrays['train-labels-idx1-ubyte'].flatten().long()
    tx = arrays['t10k-images-idx3-ubyte'].float() / 255
    ty = arrays['t10k-labels-idx1-ubyte'].flatten().long()
    assert x.shape == (60000, 784) and tx.shape == (10000, 784)
    g = torch.Generator().manual_seed(20261110)
    train, val = torch.randperm(len(x), generator=g).split([55000, 5000])
    np.savez(out/'split.npz', train=train.numpy(), validation=val.numpy())
    (out/'data_manifest.json').write_text(json.dumps({'sha256_uncompressed_idx': hashes,
        'normalization': 'uint8 / 255; no centering or augmentation',
        'split_seed': 20261110, 'train': len(train), 'validation': len(val), 'test':len(tx)}, indent=2))
    return x[train], y[train], x[val], y[val], tx, ty


class TreeSampler(nn.Module):
    def __init__(self, k, base=4, inputs=784, classes=10):
        super().__init__()
        self.k, self.base, self.inputs, self.classes = k, base, inputs, classes
        self.depth = math.ceil(math.log(inputs, base))
        self.nodes = (base**self.depth-1)//(base-1)
        # Each branch starts proportional to its number of valid descendant
        # leaves. This yields exactly uniform leaf probabilities on 0..783.
        init = torch.zeros(self.nodes, base)
        mask = torch.zeros(self.nodes, base, dtype=torch.bool)
        for depth in range(self.depth):
            offset = (base**depth-1)//(base-1)
            span = base**(self.depth-depth-1)
            for prefix in range(base**depth):
                counts = torch.tensor([max(0, min(span, inputs-(prefix*base+c)*span)) for c in range(base)])
                mask[offset+prefix] = counts > 0
                # Unreachable nodes receive a safe dummy distribution.
                if not (counts > 0).any():
                    mask[offset+prefix, 0] = True
                init[offset+prefix] = counts.clamp_min(1).float().log()
        self.logits = nn.Parameter(init.expand(classes, k, -1, -1).clone())
        self.register_buffer('valid', mask)
        self.weight = nn.Parameter(torch.randn(classes, k) / math.sqrt(k))
        self.bias = nn.Parameter(torch.zeros(classes))

    def sample(self, samples, greedy=False):
        shape = (samples, self.classes, self.k)
        node = torch.zeros(shape, dtype=torch.long)
        index = torch.zeros_like(node)
        logp = torch.zeros(shape)
        m = torch.arange(self.classes)[None, :, None]
        r = torch.arange(self.k)[None, None, :]
        for _ in range(self.depth):
            z = self.logits[m, r, node].masked_fill(~self.valid[node], -torch.inf)
            lp = z.log_softmax(-1)
            c = z.argmax(-1) if greedy else torch.multinomial(lp.exp().reshape(-1, self.base), 1).reshape(shape)
            logp = logp + lp.gather(-1, c[..., None]).squeeze(-1)
            index = self.base*index+c
            node = self.base*node+1+c
        return index, logp.sum(-1)

    def forward_indices(self, x, indices):
        return (x[:, indices] * self.weight).sum(-1).permute(1, 0, 2) + self.bias

    def forward(self, x, samples=8):
        indices, logp = self.sample(samples)
        return self.forward_indices(x, indices), logp

    @torch.no_grad()
    def leaf_probabilities(self):
        p = torch.ones(self.classes, self.k, 1)
        for depth in range(self.depth):
            offset = (self.base**depth-1)//(self.base-1)
            n = self.base**depth
            z = self.logits[:, :, offset:offset+n].masked_fill(~self.valid[offset:offset+n], -torch.inf)
            p = (p[..., None] * z.softmax(-1)).flatten(-2)
        return p[..., :self.inputs]


class FixedSampler(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.register_buffer('indices', torch.randint(784, (1, 10, k)))
        self.weight = nn.Parameter(torch.randn(10, k)/math.sqrt(k))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        return (x[:, self.indices[0]]*self.weight).sum(-1)+self.bias


def reinforce_objective(logits, logp, target, entropy_coef):
    """Unbiased score estimator of expected CE; class-specific baseline.

    For sample s,class m, replace just that class logit by the corresponding
    logit from sample (s+1)%S. Conditional on all other classes this baseline
    is independent of the current class routes. Reduces global-loss noise.
    """
    s, b, m = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, m), target.repeat(s))
    with torch.no_grad():
        z = logits.detach()
        other_sample = z.roll(1, 0)
        log_z = z.logsumexp(-1, keepdim=True)
        p = (z-log_z).exp().clamp(max=1-1e-7)
        log_z_without_m = log_z + torch.log1p(-p)
        counterfactual_log_z = torch.logaddexp(log_z_without_m, other_sample)
        advantage = (log_z-counterfactual_log_z - F.one_hot(target, m)[None]*(z-other_sample)).mean(1)
        # Gradient of -lambda H(p) is E[lambda log p * grad log p].
        advantage += entropy_coef*(logp.detach()-logp.detach().roll(1, 0))
    return ce+(advantage*logp).sum(-1).mean(), ce.detach()


@torch.no_grad()
def accuracy(model, x, y, stochastic=False, repeats=8, seed=9401):
    # Independent routing per example. Reusing independent pre-sampled paths
    # across chunks avoids making a single topology decide an entire test set.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        count, loss, var, total = 0., 0., 0., 0
        for start in range(0, len(x), 512):
            xx, yy = x[start:start+512], y[start:start+512]
            if stochastic:
                b = len(xx)
                ix, _ = model.sample(b*repeats)
                # Each image has a fresh, independent set of routes.
                ix = ix.reshape(b, repeats, 10, model.k)
                z = (xx[torch.arange(b)[:, None, None, None], ix]*model.weight).sum(-1)+model.bias
                count += (z.argmax(-1) == yy[:, None]).float().sum().item()/repeats
                loss += F.cross_entropy(z.reshape(-1, 10), yy.repeat_interleave(repeats), reduction='sum').item()/repeats
                var += z.var(1, unbiased=False).mean(-1).sum().item()
            else:
                z = model(xx)
                count += (z.argmax(-1)==yy).sum().item()
                loss += F.cross_entropy(z, yy, reduction='sum').item()
            total += len(xx)
        return {'accuracy': count/total, 'ce':loss/total, 'logit_variance': var/total}


def train_one(kind, k, seed, data, args):
    name = f'{kind}_k{k}_s{seed}'
    result_path = args.out/'runs'/f'{name}.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    torch.manual_seed(seed)
    x, y, vx, vy, tx, ty = data
    model = nn.Linear(784, 10) if kind == 'dense' else FixedSampler(k) if kind == 'random' else TreeSampler(k, args.base)
    if kind == 'learned':
        opt = torch.optim.Adam([{'params':[model.logits], 'lr':args.route_lr},
                                {'params':[model.weight, model.bias], 'lr':args.lr}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(123000+seed)
    history, best, best_epoch, state = [], -1, 0, None
    significant_best, significant_epoch = -1., 0
    stopped_on_plateau = False
    resume_path = args.out/'checkpoints'/f'{name}_resume.pt'
    start_epoch = 0
    previous_wall = 0.
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['model'])
        opt.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng'])
        g.set_state(resume['shuffle_rng'])
        history, best, best_epoch, state = resume['history'], resume['best'], resume['best_epoch'], resume['best_state']
        significant_best, significant_epoch = resume['significant_best'], resume['significant_epoch']
        start_epoch, previous_wall = resume['epoch'], resume['wall_seconds']
    begin = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        perm = torch.randperm(len(x), generator=g)
        total_ce = 0.
        train_start = time.perf_counter()
        model.train()
        for ix in perm.split(args.batch):
            opt.zero_grad(set_to_none=True)
            if kind == 'learned':
                logits, logp = model(x[ix], args.samples)
                objective, ce = reinforce_objective(logits, logp, y[ix], args.entropy*max(0.,1-epoch/args.entropy_epochs))
            else:
                ce = F.cross_entropy(model(x[ix]), y[ix])
                objective = ce
            objective.backward()
            opt.step()
            total_ce += ce.item()*len(ix)
        train_seconds = time.perf_counter()-train_start
        if args.until_plateau and epoch+1>20 and (epoch+1)%args.eval_every:
            history.append({'epoch':epoch+1,'train_ce':total_ce/len(x),'train_seconds':train_seconds})
            continue
        model.eval()
        val = accuracy(model, vx, vy, kind=='learned', repeats=4)
        history.append({'epoch':epoch+1, 'train_ce': total_ce/len(x), 'validation':val, 'train_seconds':train_seconds})
        if val['accuracy'] > best:
            best, best_epoch = val['accuracy'], epoch+1
            state = copy.deepcopy(model.state_dict())
        if val['accuracy'] > significant_best+args.min_delta:
            significant_best, significant_epoch = val['accuracy'], epoch+1
        print(json.dumps({'run':name, **history[-1]}), flush=True)
        if args.until_plateau:
            evaluations = [h for h in history if 'validation' in h]
            recent = [h['validation']['accuracy'] for h in evaluations[-6:]]
            mean_gain = float(np.mean(recent[-3:])-np.mean(recent[-6:-3])) if len(recent)==6 else float('inf')
            stopped_on_plateau = (epoch+1 >= args.min_epochs and epoch+1-significant_epoch >= args.patience
                                  and mean_gain <= args.min_delta)
            torch.save({'model':model.state_dict(), 'optimizer':opt.state_dict(),
                'torch_rng':torch.get_rng_state(),'shuffle_rng':g.get_state(),
                'history':history,'best':best,'best_epoch':best_epoch,'best_state':state,
                'significant_best':significant_best,'significant_epoch':significant_epoch,
                'epoch':epoch+1,'wall_seconds':previous_wall+time.perf_counter()-begin},resume_path)
            if stopped_on_plateau:
                print(f'PLATEAU {name} epoch={epoch+1} last_significant_epoch={significant_epoch} mean_gain={mean_gain:.6f}',flush=True)
                break
    if args.until_plateau and not stopped_on_plateau:
        raise RuntimeError(f'{name}: reached {args.epochs} epochs without satisfying plateau; resume checkpoint saved')
    model.load_state_dict(state)
    torch.save({'state':state, 'kind':kind,'k':k,'seed':seed, 'base':args.base}, args.out/'checkpoints'/f'{name}.pt')
    result = {'name':name, 'kind':kind, 'k':k,'seed':seed, 'base':args.base,
        'best_epoch':best_epoch, 'best_validation_accuracy':best,
        'parameters':sum(p.numel() for p in model.parameters()),
        'wall_seconds':previous_wall+time.perf_counter()-begin, 'history':history,
        'epochs_trained':epoch+1,'stopped_on_plateau':stopped_on_plateau,
        'last_significant_improvement_epoch':significant_epoch,
        'test':accuracy(model, tx, ty, kind=='learned', repeats=16)}
    if kind == 'learned':
        with torch.random.fork_rng():
            torch.manual_seed(9901)
            prob = model.leaf_probabilities()
            assert torch.allclose(prob.sum(-1), torch.ones(10,k), atol=1e-5)
            result['route_entropy_nats'] = (-(prob*prob.clamp_min(1e-30).log()).sum(-1)).mean().item()
            result['max_leaf_probability'] = prob.max(-1).values.mean().item()
            ix, _ = model.sample(1024)
            result['unique_pixels_per_image'] = float(np.mean([len(np.unique(i)) for i in ix.numpy()]))
            result['pixel_reads_with_duplicates'] = 10*k
            with torch.no_grad():
                greedy, _ = model.sample(1, greedy=True)
                result['test_greedy_path'] = accuracy(lambda xx: model.forward_indices(xx,greedy)[0],tx,ty)
                dense_weight = (prob*model.weight[...,None]).sum(1)
                result['test_exact_mean_logits_dense_cost'] = accuracy(lambda xx: F.linear(xx,dense_weight,model.bias),tx,ty)
                # Global MAP differs from greedy selection of each branch.
                map_ix = prob.argmax(-1)[None]
                result['test_map_leaf'] = accuracy(lambda xx: model.forward_indices(xx,map_ix)[0],tx,ty)
                # S independent logit samples per example, then average logits.
                result['test_mc_mean_logits'] = {}
                for samples in [4,16]:
                    correct = 0
                    for start in range(0,len(tx),256):
                        xx, yy = tx[start:start+256], ty[start:start+256]
                        b = len(xx)
                        ix,_ = model.sample(b*samples)
                        ix = ix.reshape(b,samples,10,k)
                        z = (xx[torch.arange(b)[:,None,None,None],ix]*model.weight).sum(-1)+model.bias
                        correct += (z.mean(1).argmax(-1)==yy).sum().item()
                    result['test_mc_mean_logits'][str(samples)] = correct/len(tx)
        result['latency'] = benchmark(model, kind)
    else:
        result['latency'] = benchmark(model, kind)
    result_path.write_text(json.dumps(result, indent=2))
    if resume_path.exists(): resume_path.unlink()
    print('RESULT '+json.dumps({a:b for a,b in result.items() if a!='history'}),flush=True)
    return result


@torch.no_grad()
def benchmark(model, kind):
    times = {}
    for batch in [1,256]:
        xx = torch.rand(batch,784)
        def forward():
            if kind == 'learned':
                ix,_ = model.sample(batch)
                return (xx[torch.arange(batch)[:,None,None],ix]*model.weight).sum(-1)+model.bias
            return model(xx)
        for _ in range(10): forward()
        reps = 50
        measurements = []
        for _ in range(5):
            t = time.perf_counter()
            for __ in range(reps): forward()
            measurements.append((time.perf_counter()-t)/reps*1e6)
        times[str(batch)] = {'median_us_per_batch': float(np.median(measurements)), 'reps':reps,'rounds':5}
    return times


def self_check():
    torch.manual_seed(23)
    model = TreeSampler(2)
    p = model.leaf_probabilities()
    assert torch.allclose(p,torch.full_like(p,1/784),atol=1e-7)
    ix, lp = model.sample(128)
    assert ix.min()>=0 and ix.max()<784 and lp.shape==(128,10)
    x = torch.rand(7,784)
    z = model.forward_indices(x,ix)
    assert z.shape==(128,7,10)
    # Verify the counterfactual score estimator against exact enumeration on
    # a genuinely stochastic toy classifier. Independent sample-pair baseline.
    q_logits = torch.tensor([[.3,-.4],[.2,.6]],requires_grad=True)
    choices = torch.tensor([[0.,1.],[.4,-.8]])
    target = torch.tensor([1])
    exact = 0.
    for a in range(2):
        for b in range(2):
            pp = q_logits.softmax(-1)[0,a]*q_logits.softmax(-1)[1,b]
            exact = exact+pp*F.cross_entropy(torch.stack([choices[0,a],choices[1,b]])[None],target)
    expected_grad = torch.autograd.grad(exact,q_logits)[0]
    estimate = torch.zeros_like(q_logits)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    ids = torch.tensor([[a,b],[c,d]])
                    lp = q_logits.log_softmax(-1).expand(2,-1,-1).gather(-1,ids[...,None]).squeeze(-1)
                    z = choices.expand(2,-1,-1).gather(-1,ids[...,None]).squeeze(-1)[:,None,:]
                    obj,_ = reinforce_objective(z,lp,target,0.)
                    grad = torch.autograd.grad(obj,q_logits)[0]
                    pair_prob = lp.detach().sum().exp()
                    estimate += pair_prob*grad
    assert torch.allclose(expected_grad,estimate,atol=1e-6), (expected_grad,estimate)
    print('SELF_CHECK_OK: uniform leaves, valid paths, gather shape, unbiased routing gradient',flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--base',type=int,default=4)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--route-lr',type=float,default=.03)
    p.add_argument('--entropy',type=float,default=.0001)
    p.add_argument('--entropy-epochs',type=int,default=20)
    p.add_argument('--until-plateau',action='store_true')
    p.add_argument('--min-epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=30)
    p.add_argument('--min-delta',type=float,default=.001)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--ks',type=int,nargs='+',default=[1,2,4,8,16])
    p.add_argument('--kinds',nargs='+',default=['dense','random','learned'])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--check-only',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    self_check()
    if args.check_only: return
    for folder in [args.out,args.out/'runs',args.out/'checkpoints']: folder.mkdir(parents=True,exist_ok=True)
    (args.out/'config.json').write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'torch':torch.__version__, 'numpy':np.__version__,'platform':platform.platform(),
        'processor':platform.processor(),'device':'cpu'},indent=2))
    data = load_data(args.data,args.out)
    results = []
    for seed in args.seeds:
        for kind in args.kinds:
            for k in ([0] if kind=='dense' else args.ks):
                results.append(train_one(kind,k,seed,data,args))
                (args.out/'results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
