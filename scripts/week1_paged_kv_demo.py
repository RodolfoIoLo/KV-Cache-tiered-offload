"""Teaching-only paged KV: block mapping, references, append and copy-on-write."""
import json
import math
from dataclasses import dataclass

import torch
from week1_transformer_demo import attention, close, make_model
from week1_kv_cache_demo import KVDecoder


@dataclass
class Request:
    blocks: list[int]
    length: int


class PagePool:
    def __init__(self, template, block_tokens=4, capacity=16):
        if block_tokens <= 0 or capacity <= 0:
            raise ValueError('Block length and pool capacity must be positive')
        layers = len(template)
        _, heads, _, dim = template[0][0].shape
        self.block_tokens = block_tokens
        shape = (capacity, layers, heads, block_tokens, dim)
        self.k = template[0][0].new_zeros(shape)
        self.v = torch.zeros_like(self.k)
        self.refs = [0] * capacity
        self.free = list(range(capacity-1, -1, -1))
        self.requests = {}

    def allocate(self):
        if not self.free:
            raise MemoryError('Teaching pool is full')
        block = self.free.pop()
        assert self.refs[block] == 0
        self.refs[block] = 1
        self.k[block].zero_()
        self.v[block].zero_()
        return block

    @property
    def used(self):
        return len(self.refs) - len(self.free)

    def create(self, name, cache, share=None, shared_blocks=0):
        if name in self.requests:
            raise ValueError('Request already exists')
        length = cache[0][0].shape[2]
        _, layers, heads, _, dim = self.k.shape
        expected_shape = (1, heads, length, dim)
        if (length == 0 or len(cache) != layers
                or any(t.shape != expected_shape or t.dtype != self.k.dtype
                       or t.device != self.k.device for pair in cache for t in pair)):
            raise ValueError('Expected one nonempty request in the pool KV layout')
        total_blocks = math.ceil(length / self.block_tokens)
        source = self.requests[share] if share is not None else None
        shareable_length = min(length, source.length) if source else 0
        if not 0 <= shared_blocks <= shareable_length // self.block_tokens:
            raise ValueError('Only complete compatible prefix blocks may be shared')
        if total_blocks - shared_blocks > len(self.free):
            raise MemoryError('Teaching pool is full')
        ks = torch.stack([k[0] for k, _ in cache])
        vs = torch.stack([v[0] for _, v in cache])
        ids = source.blocks[:shared_blocks] if source else []
        # Check the whole prefix before changing any ownership counts.
        for logical, block in enumerate(ids):
            start = logical * self.block_tokens
            end = start + self.block_tokens
            close(self.k[block], ks[:, :, start:end])
            close(self.v[block], vs[:, :, start:end])
        for block in ids:
            self.refs[block] += 1
        for logical in range(shared_blocks, total_blocks):
            start = logical * self.block_tokens
            count = min(self.block_tokens, length - start)
            block = self.allocate()
            self.k[block, :, :, :count] = ks[:, :, start:start+count]
            self.v[block, :, :, :count] = vs[:, :, start:start+count]
            ids.append(block)
        self.requests[name] = Request(ids, length)
        self.check()

    def fork(self, parent, child):
        if child in self.requests:
            raise ValueError('Request already exists')
        old = self.requests[parent]
        for block in old.blocks:
            self.refs[block] += 1
        self.requests[child] = Request(list(old.blocks), old.length)
        self.check()

    def append(self, name, new_k, new_v):
        request = self.requests[name]
        expected = self.k[0, :, :, :1]
        if any(t.shape != expected.shape or t.dtype != expected.dtype
               or t.device != expected.device for t in (new_k, new_v)):
            raise ValueError('Append requires one token in the pool KV layout')
        offset = request.length % self.block_tokens
        if offset == 0:
            request.blocks.append(self.allocate())
        elif self.refs[request.blocks[-1]] > 1:
            old = request.blocks[-1]
            new = self.allocate()
            self.k[new].copy_(self.k[old])
            self.v[new].copy_(self.v[old])
            self.refs[old] -= 1
            request.blocks[-1] = new
        block = request.blocks[-1]
        self.k[block, :, :, offset:offset+1] = new_k
        self.v[block, :, :, offset:offset+1] = new_v
        request.length += 1
        self.check()

    def gather(self, name):
        request = self.requests[name]
        keys = torch.cat([self.k[b] for b in request.blocks], dim=2)[:, :, :request.length]
        values = torch.cat([self.v[b] for b in request.blocks], dim=2)[:, :, :request.length]
        return [(keys[layer:layer+1], values[layer:layer+1])
                for layer in range(keys.shape[0])]

    def release(self, name):
        request = self.requests.pop(name)
        for block in request.blocks:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)
        self.check()

    def check(self):
        counts = [0] * len(self.refs)
        for request in self.requests.values():
            assert len(request.blocks) == math.ceil(request.length/self.block_tokens)
            for block in request.blocks:
                counts[block] += 1
        assert counts == self.refs
        assert len(self.free) == len(set(self.free))
        assert set(self.free) == {i for i, count in enumerate(counts) if count == 0}


def compare_cache(actual, expected):
    assert len(actual) == len(expected)
    for (ak, av), (ek, ev) in zip(actual, expected):
        close(ak, ek)
        close(av, ev)
        q = torch.ones(1, ak.shape[1], 1, ak.shape[-1], dtype=ak.dtype)
        # Place the single query at the last position so it sees the whole prefix.
        offset = ak.shape[2] - 1
        close(attention(q, ak, av, offset=offset)[0],
              attention(q, ek, ev, offset=offset)[0])


def last_column(cache):
    return (torch.stack([k[0, :, -1:] for k, _ in cache]),
            torch.stack([v[0, :, -1:] for _, v in cache]))


@torch.no_grad()
def main():
    model = make_model(KVDecoder)
    _, a = model.cached(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    _, b = model.cached(torch.tensor([[1, 2, 3, 4, 9]]))
    separate = PagePool(a)
    separate.create('A', a)
    separate.create('B', b)
    unshared_count = separate.used
    assert unshared_count == 4
    separate.release('A')
    separate.release('B')
    pool = PagePool(a)
    pool.create('A', a)
    pool.create('B', b, share='A', shared_blocks=1)
    shared_count = pool.used
    assert shared_count == 3
    table = {name: dict(blocks=list(r.blocks), tokens=r.length)
             for name, r in pool.requests.items()}
    compare_cache(pool.gather('A'), a)
    compare_cache(pool.gather('B'), b)
    pool.release('A')
    compare_cache(pool.gather('B'), b)
    pool.release('B')
    assert pool.used == 0

    boundaries = {}
    for length in (3, 4, 5):
        _, cache = model.cached(torch.arange(1, length+1)[None])
        pool.create('test', cache)
        compare_cache(pool.gather('test'), cache)
        boundaries[length] = pool.used
        assert pool.used == math.ceil(length/4)
        pool.release('test')

    _, prefix = model.cached(torch.tensor([[1, 2, 3]]))
    pool.create('A', prefix)
    pool.fork('A', 'B')
    shared_id = pool.requests['B'].blocks[-1]
    _, a4 = model.cached(torch.tensor([[4]]), prefix)
    pool.append('A', *last_column(a4))
    new_id = pool.requests['A'].blocks[-1]
    assert new_id != shared_id and pool.requests['B'].blocks[-1] == shared_id
    compare_cache(pool.gather('B'), prefix)
    _, b5 = model.cached(torch.tensor([[5]]), prefix)
    pool.append('B', *last_column(b5))
    compare_cache(pool.gather('A'), a4)
    compare_cache(pool.gather('B'), b5)
    pool.release('A')
    pool.release('B')
    assert pool.used == 0
    print(json.dumps(dict(passed=True, block_size_tokens=pool.block_tokens,
                          shared_block_count=shared_count, unshared_block_count=unshared_count,
                          request_tables=table, boundary_block_counts=boundaries,
                          copy_on_write=(new_id != shared_id),
                          fork_tail_before=shared_id, tail_after_append={'A': new_id, 'B': shared_id},
                          all_blocks_reclaimed=(pool.used == 0 and separate.used == 0),
                          note='CPU gather and attention compared with contiguous KV'), indent=2))


if __name__ == '__main__':
    main()
