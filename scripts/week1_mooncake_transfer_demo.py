"""Linux CPU/TCP KV handoff through the real Mooncake Transfer Engine."""
import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys

import torch
from week1_transformer_demo import close, make_model
from week1_kv_cache_demo import BUNDLE, KVDecoder, WORK, model_id


def send_json(conn, value):
    conn.sendall(json.dumps(value).encode('utf-8') + b'\n')


def recv_json(conn):
    # Read exactly one frame; do not accidentally consume the next message.
    data = bytearray()
    while not data.endswith(b'\n'):
        char = conn.recv(1)
        if not char:
            raise ConnectionError('Control connection closed before message completed')
        data.extend(char)
        if len(data) > 65536:
            raise ValueError('Control message too long')
    return json.loads(data.decode('utf-8'))


def load_case(path):
    case = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(KVDecoder)
    model.load_state_dict(case['model_state'])
    if model_id(model) != case['model_id']:
        raise ValueError('Model identity mismatch')
    return case, model


def payload_for(case):
    buf = io.BytesIO()
    torch.save({'model_id': case['model_id'], 'prefix': case['prefix'],
                'suffix': case['suffix'], 'cache': case['cache']}, buf)
    return buf.getvalue()


def engine_new():
    # Mooncake can auto-discover other transports even when TCP is requested.
    os.environ['MC_FORCE_TCP'] = '1'
    from mooncake.engine import TransferEngine
    engine = TransferEngine()
    ret = engine.initialize('127.0.0.1', 'P2PHANDSHAKE', 'tcp', '')
    if ret is not None and ret != 0:
        raise RuntimeError(f'Mooncake initialization failed: {ret}')
    return engine


def register(engine, pointer, size):
    ret = engine.register_memory(pointer, size)
    if ret != 0:
        raise RuntimeError(f'Memory registration failed: {ret}')


def finish(role, result):
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f'mooncake-{role}.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print(f'Result file: {path}', flush=True)


@torch.no_grad()
def receiver(args):
    case, model = load_case(args.bundle)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', args.port))
        listener.listen(1)
        listener.settimeout(300)
        print(f'READY: receiver on 127.0.0.1:{args.port}; now start sender', flush=True)
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(300)
            hello = recv_json(conn)
            size = hello['bytes']
            if not isinstance(size, int) or not 0 < size <= 64*1024*1024:
                raise ValueError('Invalid payload size')
            engine = engine_new()
            buffer = (ctypes.c_uint8 * size)()
            pointer = ctypes.addressof(buffer)
            register(engine, pointer, size)
            try:
                send_json(conn, dict(session=f'127.0.0.1:{engine.get_rpc_port()}',
                                     pointer=pointer, bytes=size))
                if recv_json(conn).get('state') != 'written':
                    raise RuntimeError('Sender did not complete transfer')
                data = bytes(buffer)
                digest = hashlib.sha256(data).hexdigest()
                if digest != hello['sha256']:
                    raise ValueError('Transferred payload checksum mismatch')
                received = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                if received['model_id'] != case['model_id']:
                    raise ValueError('Received cache belongs to another model')
                assert torch.equal(received['prefix'], case['prefix'])
                assert torch.equal(received['suffix'], case['suffix'])
                for pair1, pair2 in zip(received['cache'], case['cache'], strict=True):
                    for a, b in zip(pair1, pair2, strict=True):
                        if a.dtype != b.dtype or not torch.equal(a, b):
                            raise ValueError('Received KV differs from the reference')
                actual, _ = model.cached(received['suffix'], received['cache'])
                full = model(torch.cat((case['prefix'], case['suffix']), dim=1))[:, -case['suffix'].shape[1]:]
                close(actual, full)
                close(actual, case['expected_logits'])
                assert torch.equal(actual.argmax(-1), full.argmax(-1))
                result = dict(passed=True, transport='Mooncake TransferEngine TCP',
                              bytes=size, sha256=digest, kv_equal=True,
                              kv_tensor_bytes=sum(t.nbytes for kv in received['cache'] for t in kv),
                              prefix_tokens=received['prefix'].shape[1],
                              continuation_tokens=received['suffix'].shape[1],
                              logits_max_error=float((actual-full).abs().max()),
                              next_ids=actual[:, -1].argmax(-1).tolist())
            finally:
                ret = engine.unregister_memory(pointer)
                if ret != 0:
                    raise RuntimeError(f'Memory deregistration failed: {ret}')
            send_json(conn, dict(passed=True, sha256=digest))
            finish('receiver', result)


def sender(args):
    case, _ = load_case(args.bundle)
    data = payload_for(case)
    digest = hashlib.sha256(data).hexdigest()
    # Keep the ctypes object alive until write completion and receiver acknowledgment.
    buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    pointer = ctypes.addressof(buffer)
    engine = engine_new()
    register(engine, pointer, len(data))
    try:
        with socket.create_connection(('127.0.0.1', args.port), timeout=30) as conn:
            conn.settimeout(300)
            send_json(conn, dict(bytes=len(data), sha256=digest))
            target = recv_json(conn)
            assert target['bytes'] == len(data)
            ret = engine.transfer_sync_write(target['session'], pointer,
                                             target['pointer'], len(data))
            if ret < 0:
                raise RuntimeError(f'Mooncake transfer failed: {ret}')
            send_json(conn, dict(state='written'))
            ack = recv_json(conn)
            if not ack.get('passed') or ack.get('sha256') != digest:
                raise RuntimeError(f'Receiver validation failed: {ack}')
    finally:
        ret = engine.unregister_memory(pointer)
        if ret != 0:
            raise RuntimeError(f'Memory deregistration failed: {ret}')
    finish('sender', dict(passed=True, bytes=len(data), sha256=digest,
                          payload_via='Mooncake TransferEngine',
                          control_via='local TCP socket'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=['receiver', 'sender'])
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        raise SystemExit('Run this demo on Linux; Windows runs the first three demos.')
    torch.set_num_threads(1)
    if not args.bundle.is_file():
        raise SystemExit('Missing bundle: first run week1_kv_cache_demo.py --mode export')
    {'receiver': receiver, 'sender': sender}[args.role](args)


if __name__ == '__main__':
    main()
