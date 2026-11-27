"""Opt-in, untimed worker diagnostics for the pinned Task A deployment."""

import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

import torch
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker
from vllm.v1.attention.backends.flash_v100 import routing


class DiagnosticExtension:
    pass


ROOT = Path(os.environ['TASK_A_DIAGNOSTIC_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
COUNTS = {}
ROUTES = {}
original_route = routing.record_route


def route(name):
    ROUTES[name] = ROUTES.get(name, 0) + 1
    return original_route(name)


routing.record_route = route


def marker():
    path = ROOT / 'current.json'
    return json.loads(path.read_text()) if path.exists() else None


def record(row):
    with LOCK:
        with (ROOT / f'worker-{os.getpid()}.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')


original_forward = Qwen2ForCausalLM.forward
original_logits = Qwen2ForCausalLM.compute_logits


def forward(self, *args, **kwargs):
    info = marker()
    if info is None:
        return original_forward(self, *args, **kwargs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ROUTES.clear()
    start.record()
    output = original_forward(self, *args, **kwargs)
    end.record()
    end.synchronize()
    input_ids = kwargs.get('input_ids', args[0] if args else None)
    record(dict(event='model_forward', marker=info, gpu_ms=start.elapsed_time(end),
                input_tokens=input_ids.numel() if input_ids is not None else None,
                attention_routes=dict(ROUTES)))
    return output


def logits(self, *args, **kwargs):
    result = original_logits(self, *args, **kwargs)
    info = marker()
    if info is not None and result is not None:
        tag = info['tag']
        ordinal = COUNTS.get(tag, 0)
        COUNTS[tag] = ordinal + 1
        path = ROOT / f'{tag}-logits-{ordinal}.pt'
        torch.save(result.detach().cpu(), path)
        record(dict(event='logits', marker=info, ordinal=ordinal, file=path.name))
    return result


Qwen2ForCausalLM.forward = forward
Qwen2ForCausalLM.compute_logits = logits


def capture_buffers(keys, addresses, sizes, operation):
    info = marker()
    if info is None or info['kind'] not in ('build', 'revisit'):
        return
    torch.cuda.set_device(0)
    runtime = ctypes.CDLL('libcudart.so.12')
    copy = runtime.cudaMemcpy
    copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    copy.restype = ctypes.c_int
    for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
        segments = []
        for pointer, length in zip(pointers, lengths, strict=True):
            buffer = ctypes.create_string_buffer(length)
            result = copy(buffer, ctypes.c_void_p(pointer), length, 2)
            if result:
                raise RuntimeError(f'Diagnostic cudaMemcpy failed: {result}')
            segments.append(buffer.raw)
        data = b''.join(segments)
        name = hashlib.sha256(key.encode()).hexdigest()
        file = f'{info["tag"]}-{operation}-{name}.bin'
        (ROOT / file).write_bytes(data)
        record(dict(event=operation, marker=info, key=key, bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), file=file,
                    segment_bytes=list(lengths)))


class ObservedStore:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *args, **kwargs):
        capture_buffers(keys, addresses, sizes, 'before_put')
        return self.inner.batch_put_from_multi_buffers(keys, addresses, sizes, *args, **kwargs)

    def batch_get_into_multi_buffers(self, keys, addresses, sizes, *args, **kwargs):
        result = self.inner.batch_get_into_multi_buffers(keys, addresses, sizes, *args, **kwargs)
        capture_buffers(keys, addresses, sizes, 'after_get')
        return result


def instrument(thread_type):
    original = thread_type._handle_request

    def handle(self, request):
        if not isinstance(self.store, ObservedStore):
            self.store = ObservedStore(self.store)
        return original(self, request)

    thread_type._handle_request = handle


instrument(worker.KVCacheStoreSendingThread)
instrument(worker.KVCacheStoreRecvingThread)
