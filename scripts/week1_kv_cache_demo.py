"""Incremental KV, compatible prefix reuse, compressed radix index and payload size."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from week1_transformer_demo import TinyDecoder, attention, close, generate_full, make_model

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local' / 'week1'
BUNDLE = WORK / 'transfer_bundle.pt'

class KVDecoder(TinyDecoder):
    def cached(self, ids, past=None):
        offset = 0
        if past is not None:
            if len(past) != self.cfg.layers:
                raise ValueError("Wrong number of cache layers")
            offset = past[0][0].shape[2]
            expected = (ids.shape[0], self.cfg.heads, offset,
                        self.cfg.width // self.cfg.heads)
            for k, v in past:
                if k.shape != expected or v.shape != expected:
                    raise ValueError('Incompatible KV shape')
                if k.dtype != self.embedding.weight.dtype or v.dtype != k.dtype:
                    raise ValueError('Incompatible KV dtype')
        x = self.embed(ids, offset)
        new_cache = []
        for index, block in enumerate(self.blocks):
            layer = block.attn
            q, k, v = (layer.split(proj(x)) for proj in (layer.q, layer.k, layer.v))
            if past is not None:
                k = torch.cat((past[index][0], k), dim=2)
                v = torch.cat((past[index][1], v), dim=2)
            out, _ = attention(q, k, v, offset=offset)
            x = block.norm1(x + layer.out(layer.merge(out)))
            x = block.norm2(x + block.ff(x))
            new_cache.append((k, v))
        return self.lm_head(x), new_cache


def clone_cache(cache):
    return [(k.clone(), v.clone()) for k, v in cache]


def model_id(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(bytes(value.detach().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


@torch.no_grad()
def generate_cached(model, prompt, count=8):
    logits, cache = model.cached(prompt)
    generated = []
    for index in range(count):
        token = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(token[0, 0]))
        if index + 1 < count:
            logits, cache = model.cached(token, cache)
    return generated, cache


@torch.no_grad()
def export_bundle():
    WORK.mkdir(parents=True, exist_ok=True)
    model = make_model(KVDecoder)
    prefix = torch.tensor([[1, 2, 3]])
    suffix = torch.tensor([[4, 5]])
    _, cache = model.cached(prefix)
    expected = model(torch.cat((prefix, suffix), dim=1))[:, -suffix.shape[1]:]
    torch.save({'model_state': model.state_dict(), 'model_id': model_id(model),
                'prefix': prefix, 'suffix': suffix, 'cache': cache,
                'expected_logits': expected}, BUNDLE)
    return dict(passed=True, bundle=str(BUNDLE), bytes=BUNDLE.stat().st_size)


@torch.no_grad()
def demo_equivalence():
    model = make_model(KVDecoder)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cache = None
    errors = []
    lengths = []
    for position in range(ids.shape[1]):
        logits, cache = model.cached(ids[:, position:position+1], cache)
        ref = model(ids[:, :position+1])[:, -1:]
        close(logits, ref)
        errors.append(float((logits-ref).abs().max()))
        lengths.append(cache[0][0].shape[2])
    _, prefix_cache = model.cached(ids[:, :3])
    chunk, _ = model.cached(ids[:, 3:5], prefix_cache)
    close(chunk, model(ids[:, :5])[:, -2:])
    full = generate_full(model, ids[:, :4])
    incremental, end_cache = generate_cached(model, ids[:, :4])
    assert full == incremental
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / 'kv-roundtrip.pt'
    torch.save(prefix_cache, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    for (k1, v1), (k2, v2) in zip(prefix_cache, restored):
        assert torch.equal(k1, k2) and torch.equal(v1, v2)
    close(model.cached(ids[:, 3:5], restored)[0], chunk)
    export_bundle()
    return dict(passed=True, maximum_logit_error=max(errors), cache_lengths=lengths,
                generated_ids=full, restored_cache_equal=True,
                generation_end_cache_length=end_cache[0][0].shape[2],
                expected_end_length=4+8-1, two_token_chunk_equal=True)


class PrefixStore:
    def __init__(self):
        self.entries = {}

    def put(self, identity, tokens, cache, last_logits):
        self.entries[(identity, tuple(tokens))] = (clone_cache(cache), last_logits.clone())

    def lookup(self, identity, tokens):
        for length in range(len(tokens), 0, -1):
            item = self.entries.get((identity, tuple(tokens[:length])))
            if item is not None:
                cache, logits = item
                return length, clone_cache(cache), logits.clone()
        return 0, None, None


@torch.no_grad()
def demo_prefix():
    model = make_model(KVDecoder)
    identity = model_id(model) + ':sin-pos-v1:float64:MHA'
    prefix = torch.tensor([[1, 2, 3]])
    logits, cache = model.cached(prefix)
    snapshot = clone_cache(cache)
    store = PrefixStore()
    store.put(identity, prefix[0].tolist(), cache, logits[:, -1:])
    matches = []
    for request in ([1, 2, 3, 4], [1, 2, 3, 5]):
        length, restored, _ = store.lookup(identity, request)
        assert length == 3
        result, _ = model.cached(torch.tensor([request[length:]]), restored)
        close(result[:, -1:], model(torch.tensor([request]))[:, -1:])
        matches.append(length)
    assert store.lookup(identity, [9, 2, 3, 4])[0] == 0
    assert store.lookup(identity+'-different-model', [1, 2, 3])[0] == 0
    length, _, saved_logits = store.lookup(identity, [1, 2, 3])
    assert length == 3
    close(saved_logits, model(prefix)[:, -1:])
    for (k, v), (old_k, old_v) in zip(cache, snapshot):
        assert torch.equal(k, old_k) and torch.equal(v, old_v)
    return dict(passed=True, partial_hits=matches, wrong_prefix_hit=0,
                wrong_identity_hit=0, branches_isolated=True,
                full_hit_uses_saved_last_logits=True)


def common_length(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


class RadixNode:
    def __init__(self, label=(), terminal=False):
        self.label, self.terminal, self.children = tuple(label), terminal, {}

    def insert(self, tokens):
        node, rest = self, tuple(tokens)
        while rest:
            child = node.children.get(rest[0])
            if child is None:
                node.children[rest[0]] = RadixNode(rest, True)
                return
            n = common_length(rest, child.label)
            if n < len(child.label):
                split = RadixNode(child.label[:n])
                node.children[rest[0]] = split
                child.label = child.label[n:]
                split.children[child.label[0]] = child
                rest = rest[n:]
                if rest:
                    split.children[rest[0]] = RadixNode(rest, True)
                else:
                    split.terminal = True
                return
            rest, node = rest[n:], child
        node.terminal = True

    def match(self, tokens):
        node, rest, matched = self, tuple(tokens), 0
        while rest and rest[0] in node.children:
            child = node.children[rest[0]]
            n = common_length(rest, child.label)
            matched += n
            if n < len(child.label):
                break
            rest, node = rest[n:], child
        return matched

    def edges(self):
        return [{'edge': list(child.label), 'terminal': child.terminal,
                'children': child.edges()} for child in self.children.values()]


@torch.no_grad()
def demo_radix():
    tree = RadixNode()
    tree.insert([1, 2, 3, 4])
    tree.insert([1, 2, 5])
    assert tree.match([1, 2, 3, 6]) == 3
    assert tree.match([9, 2, 3]) == 0
    assert tree.match([1, 2, 3, 4]) == 4
    model = make_model(KVDecoder)
    _, long_cache = model.cached(torch.tensor([[1, 2, 3, 4]]))
    sliced = [(k[:, :, :3].clone(), v[:, :, :3].clone()) for k, v in long_cache]
    _, direct = model.cached(torch.tensor([[1, 2, 3]]))
    for pair1, pair2 in zip(sliced, direct):
        close(pair1[0], pair2[0]); close(pair1[1], pair2[1])
    return dict(passed=True, tree=tree.edges(), partial_edge_match=3,
                matching_prefix_can_slice_compatible_cache=True)


@torch.no_grad()
def demo_size():
    measured = []
    for dtype in (torch.float64, torch.float32):
        for batch, length in ((1, 6), (2, 6), (1, 12)):
            model = make_model(KVDecoder).to(dtype=dtype)
            _, cache = model.cached(torch.ones(batch, length, dtype=torch.long))
            actual = sum(t.numel()*t.element_size() for pair in cache for t in pair)
            expected = 2*model.cfg.layers*batch*length*model.cfg.width*cache[0][0].element_size()
            assert actual == expected
            measured.append(dict(dtype=str(dtype), batch=batch, tokens=length, bytes=actual))
    per_token = 2*32*32*128*2
    assert per_token*32768 == 16*2**30
    assert per_token*32768*5 == 80*2**30
    assert (2*32*8*128*2)*32768 == 4*2**30
    return dict(passed=True, measured_payloads=measured,
                analytical_gib=[16, 80, 4], note='payload only, no weights or allocator metadata')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = {'equivalence': demo_equivalence, 'prefix': demo_prefix,
            'radix': demo_radix, 'size': demo_size, 'export': export_bundle}
    parser.add_argument('--mode', required=True, choices=list(modes))
    result = modes[parser.parse_args().mode]()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()