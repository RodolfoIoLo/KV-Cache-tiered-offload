"""CPU learning model: manual attention, two-layer decoder and one training loop."""
import argparse
from dataclasses import dataclass
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

DTYPE = torch.float64


@dataclass(frozen=True)
class Config:
    vocab: int = 16 
    width: int = 8  
    heads: int = 2  
    layers: int = 2 
    ffn: int = 32 
    max_length : int = 64 


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


def attention(q, k, v, casual=True, offset=0):
    scores = q @ k.transpose(-2, -1)/ math.sqrt(q.shape[-1])
    if casual:
        qpos = offset + torch.arange(q.shape[-2], device=q.device)
        kpos = torch.arange(k.shape[-2], device=q.device)
        allowed = kpos[None, :] <= qpos[:, None]
        scores = scores.masked_fill(~allowed, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.dim = cfg.width // cfg.heads
        self.q = nn.Linear(cfg.width, cfg.width)
        self.k = nn.Linear(cfg.width, cfg.width)
        self.v = nn.Linear(cfg.width, cfg.width)
        self.out = nn.Linear(cfg.width, cfg.width)

    def split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.dim).transpose(1, 2)

    def merge(self, x):
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, t, -1)

    def forward(self, x):
        q, k, v = (self.split(proj(x)) for proj in (self.q, self.k, self.v))
        out, _ = attention(q, k, v)
        return self.out(self.merge(out))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = SelfAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.ffn), nn.ReLU(),
                                nn.Linear(cfg.ffn, cfg.width))

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ff(x))


class TinyDecoder(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        pos = torch.arange(cfg.max_length, dtype=DTYPE)[:, None]
        scale = torch.exp(torch.arange(0, cfg.width, 2, dtype=DTYPE)
                          * (-math.log(10000.0) / cfg.width))
        pe = torch.zeros(cfg.max_length, cfg.width, dtype=DTYPE)
        pe[:, 0::2] = torch.sin(pos * scale)
        pe[:, 1::2] = torch.cos(pos * scale)
        self.register_buffer('position', pe)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.lm_head = nn.Linear(cfg.width, cfg.vocab, bias=False)

    def embed(self, ids, offset=0):
        end = offset + ids.shape[1]
        if ids.ndim != 2 or ids.shape[1] == 0 or end > self.cfg.max_length:
            raise ValueError('Expected nonempty [batch, tokens] within context limit')
        return self.embedding(ids) * math.sqrt(self.cfg.width) + self.position[offset:end]

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def make_model(cls=TinyDecoder):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return cls().to(dtype=DTYPE, device='cpu').eval()


@torch.no_grad()
def generate_full(model, prompt, count=8):
    ids = prompt.clone()
    generated = []
    for _ in range(count):
        token = model(ids)[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(token[0, 0]))
        ids = torch.cat((ids, token), dim=1)
    return generated


@torch.no_grad
def demo_attention():
    model = make_model()
    torch.manual_seed(11)
    x = torch.randn(1, 4, 8, dtype=DTYPE)
    layer = model.blocks[0].attn 
    q, k, v = (layer.split(proj(x)) for proj in (layer.q, layer.k, layer.v))
    out, weights = attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    close(out, ref)
    close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    assert torch.isfinite(out).all()
    changed = x.clone()
    changed[:, -1] += 3.0
    close(layer(x)[:, :-1], layer(changed)[:, :-1])
    q2 = torch.randn(1, 2, 3, 4, dtype=DTYPE)
    k2 = torch.randn(1, 2, 5, 4, dtype=DTYPE)
    v2 = torch.randn_like(k2)
    cross, cross_weights = attention(q2, k2, v2, casual=False)
    close(cross, F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0))
    assert cross_weights.shape == (1, 2, 3, 5)
    merged = layer.out(layer.merge(out))
    return dict(passed=True, input_shape=list(x.shape), q_shape=list(q.shape),
                weights_first_head=weights[0, 0].tolist(),
                output_shape=list(merged.shape),
                reference_max_error=float((out-ref).abs().max()),
                future_change_does_not_affect_past=True,
                cross_weights_shape=list(cross_weights.shape))


def demo_decoder():
    model = make_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = model(prompt)
        close(logits, model(prompt))
    return dict(passed=True, logits_shape=list(logits.shape),
                generated_ids=generate_full(model, prompt),
                note='Untrained teaching model, not language quality')


def demo_basics():
    model = make_model()
    x = torch.arange(12, dtype=DTYPE).reshape(3, 4)
    w = torch.arange(8, dtype=DTYPE).reshape(4, 2)
    result = x @ w
    scores = torch.tensor([2.0, 1.0, 0.0], dtype=DTYPE)
    probs = torch.softmax(scores, dim=-1)
    close(probs, torch.softmax(scores+100, dim=-1))
    assert int(torch.tensor([1, 2]) @ torch.tensor([3, 4])) == 11
    assert result.shape == (3, 2)
    value = torch.tensor([1.0, 2.0], dtype=DTYPE, requires_grad=True)
    value.square().sum().backward()
    close(value.grad, 2*value.detach())
    return dict(passed=True, dot_product=11, matrix_product=result.tolist(),
                reshape_4_by_3=x.reshape(4, 3).tolist(),
                transpose_4_by_3=x.transpose(0, 1).tolist(),
                softmax=probs.tolist(), softmax_sum=float(probs.sum()),
                gradient_of_sum_squares=value.grad.tolist())


def demo_training():
    model = make_model().train()
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1]])
    x, target = ids[:, :-1], ids[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.embedding.weight.detach().clone()
    losses = []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab), target.reshape(-1))
        loss.backward()
        assert model.embedding.weight.grad is not None
        assert torch.isfinite(model.embedding.weight.grad).all()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert not torch.equal(before, model.embedding.weight)
    assert losses[-1] < losses[0]
    model.eval()
    with torch.no_grad():
        close(model(x), model(x))
    return dict(passed=True, first_loss=losses[0], last_loss=losses[-1],
                parameter_changed=True, steps=len(losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['basics', 'attention', 'decoder', 'train-step'])
    args = parser.parse_args()
    result = {'basics': demo_basics, 'attention': demo_attention, 'decoder': demo_decoder, 'train-step': demo_training}[args.mode]()
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
