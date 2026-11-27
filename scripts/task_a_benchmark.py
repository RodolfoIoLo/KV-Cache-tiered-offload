"""Measure streaming TTFT after verified GPU prefix eviction on a dedicated vLLM endpoint."""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


BLOCK = 16
KV_BYTES_PER_TOKEN = 28672  # Qwen2.5-1.5B, TP=1, 28 layers, 2 KV heads, FP16.
SUFFIX = 8
METRIC = re.compile(r'^vllm:(\w+)\{([^}]*)\}\s+([-\d.eE+]+)$')
LABEL = re.compile(r'(\w+)="([^"]*)"')


def tokens(seed, repetition, length, role, count):
    key = f'{seed}/{repetition}/{length}/{role}'
    rng = random.Random(key)
    return [rng.randrange(256, 16000) for _ in range(count)]


def metrics(api, model):
    with urlopen(api + '/metrics', timeout=20) as response:
        lines = response.read().decode().splitlines()
    values, config = {}, {}
    for line in lines:
        match = METRIC.match(line)
        if not match:
            continue
        name, fields, value = match.groups()
        labels = dict(LABEL.findall(fields))
        if name == 'cache_config_info':
            if labels.get('engine') == '0':
                config = labels
            continue
        if labels.get('model_name') != model or labels.get('engine') != '0':
            continue
        key = None
        if name in ('prefix_cache_queries_total', 'prefix_cache_hits_total',
                    'external_prefix_cache_hits_total', 'time_to_first_token_seconds_sum',
                    'time_to_first_token_seconds_count'):
            key = name
        elif name == 'prompt_tokens_by_source_total':
            key = labels['source']
        elif name == 'request_success_total':
            key = 'finish_' + labels['finished_reason']
        elif name.startswith('mooncake_store_operation') and not name.endswith('_created'):
            if name.endswith('_bucket'):
                continue
            key = '/'.join((name.removeprefix('mooncake_store_operation').lstrip('_'),
                            labels['operation'], labels['status']))
        if key:
            values[key] = values.get(key, 0) + float(value)
    mandatory = ('local_compute', 'local_cache_hit', 'external_kv_transfer',
                 'prefix_cache_hits_total', 'prefix_cache_queries_total', 'finish_length')
    missing = [key for key in mandatory if key not in values]
    if missing or not config:
        raise ValueError(f'Missing engine/model counters {missing} or cache config for {model}')
    return values, config


def completion(api, model, prompt, output_tokens, read_lengths=None, logprobs=None):
    payload = dict(model=model, prompt=prompt, temperature=0, seed=0,
                   min_tokens=output_tokens, max_tokens=output_tokens, ignore_eos=True,
                   stream=True, return_token_ids=True, add_special_tokens=False)
    if read_lengths is not None:
        payload['kv_transfer_params'] = {'read_prefix_lengths': read_lengths}
    if logprobs is not None:
        payload['logprobs'] = logprobs
    req = Request(api + '/v1/completions', data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    ids, first, reason, request_id, done = [], None, None, None, False
    score_chunks = []
    start = time.perf_counter_ns()
    with urlopen(req, timeout=180) as response:
        for line in response:
            arrived = time.perf_counter_ns()
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(event['error'])
            request_id = event['id']
            for choice in event['choices']:
                chunk = choice.get('token_ids') or []
                if chunk and first is None:
                    first = arrived
                ids.extend(chunk)
                if choice.get('logprobs'):
                    score_chunks.append(choice['logprobs'])
                reason = choice.get('finish_reason') or reason
    if not done or first is None or reason != 'length' or len(ids) != output_tokens:
        raise RuntimeError(f'Incomplete stream {request_id}: {reason=}, {ids=}, {done=}')
    result = dict(request_id=request_id, ttft_ms=(first-start)/1e6, token_ids=ids)
    if logprobs is not None:
        result['logprobs'] = score_chunks
    return result


def observe(args, prompt, kind, length, repetition, index=None):
    diagnostic = os.environ.get('TASK_A_DIAGNOSTIC_DIR')
    if diagnostic:
        marker = dict(tag=f'{kind}-{length}-{repetition}-{index}',
                      kind=kind, length=length, repetition=repetition, index=index)
        path = Path(diagnostic) / 'current.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(marker), encoding='utf-8')
        temp.replace(path)
    before, _ = metrics(args.api, args.model)
    policy = args.read_prefix_lengths if kind == 'revisit' else None
    result = completion(args.api, args.model, prompt, args.output_tokens, policy, args.logprobs)
    skip_read = isinstance(policy, list) and length not in policy
    get_tokens = length if kind == 'revisit' and args.mode == 'store' and not skip_read else 0
    get_bytes = get_tokens * KV_BYTES_PER_TOKEN
    # All new full blocks are asynchronously saved. A short divergent suffix plus
    # generated tokens fits in one partial block and is not eligible for saving.
    # Revisit prefix keys already exist even if the policy chooses recomputation.
    put_bytes = (len(prompt) // BLOCK * BLOCK * KV_BYTES_PER_TOKEN
                 if args.mode == 'store' and kind != 'revisit' else 0)
    deadline = time.monotonic() + args.metrics_timeout
    while True:
        after, _ = metrics(args.api, args.model)
        delta = {key: val - before.get(key, 0) for key, val in after.items()}
        if (delta['finish_length'] >= 1 and delta.get('bytes_total/load_get/ok', 0) >= get_bytes
                and delta.get('bytes_total/save_put/ok', 0) >= put_bytes):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'{kind}: counters did not settle for {result["request_id"]}: '
                               f'expected GET {get_bytes}, PUT {put_bytes}, observed {delta}')
        time.sleep(.2)
    expected = {
        'finish_length': 1, 'finish_stop': 0, 'finish_error': 0, 'finish_abort': 0,
        'prefix_cache_queries_total': len(prompt), 'prefix_cache_hits_total': 0,
        'local_cache_hit': 0, 'external_kv_transfer': get_tokens,
        'external_prefix_cache_hits_total': get_tokens, 'local_compute': len(prompt)-get_tokens,
        'bytes_total/load_get/ok': get_bytes, 'keys_total/load_get/ok': get_tokens//BLOCK,
        'bytes_total/save_put/ok': put_bytes, 'time_to_first_token_seconds_count': 1,
    }
    for key, target in expected.items():
        if delta.get(key, 0) != target:
            raise ValueError(f'{kind}, {result["request_id"]}, {key}: '
                             f'expected {target}, observed {delta.get(key, 0)}, counters {delta}')
    if any(val for key, val in delta.items()
           if key.startswith('failed_keys_total/') or key.endswith(('/error', '/partial_failure'))):
        raise ValueError(f'Mooncake operation failed: {delta}')
    return dict(**result, kind=kind, length=length, repetition=repetition, index=index,
                seed=args.seed, mode=args.mode, prompt_tokens=len(prompt),
                prompt_digest=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                prefix_hits=int(delta['prefix_cache_hits_total']),
                external_tokens=int(delta['external_kv_transfer']),
                compute_tokens=int(delta['local_compute']), get_bytes=get_bytes, put_bytes=put_bytes,
                get_ms=delta.get('time_seconds_sum/load_get/ok', 0)*1000,
                lookup_ms=delta.get('time_seconds_sum/lookup_exists/ok', 0)*1000,
                server_ttft_ms=delta['time_to_first_token_seconds_sum']*1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['store', 'recompute'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--api', default='http://127.0.0.1:8102')
    parser.add_argument('--model', default='week1-qwen')
    parser.add_argument('--lengths', default='512')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--repetition-start', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=1, help='Excluded rounds per length')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--read-prefix-lengths', help='Optional routing connector: all, none, or a comma-separated set of token lengths')
    parser.add_argument('--pressure-count', type=int, default=3)
    parser.add_argument('--pressure-length', type=int, default=1024)
    parser.add_argument('--output-tokens', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--metrics-timeout', type=float, default=35)
    parser.add_argument('--logprobs', type=int, choices=range(1, 21), help='Diagnostic token scores')
    args = parser.parse_args()
    lengths = [int(s) for s in args.lengths.split(',')]
    if args.read_prefix_lengths is not None:
        if args.mode != 'store':
            parser.error('A read policy requires the Store routing connector')
        if args.read_prefix_lengths != 'all':
            args.read_prefix_lengths = ([] if args.read_prefix_lengths == 'none' else
                                        [int(n) for n in args.read_prefix_lengths.split(',')])
    if (not lengths or len(set(lengths)) != len(lengths) or any(n <= 0 or n % BLOCK for n in lengths)
            or args.pressure_length < BLOCK or args.pressure_length % BLOCK
            or args.warmups < 0 or args.repetitions < 1 or args.pressure_count < 1):
        parser.error('Use distinct positive, block-aligned lengths, and at least one trial and pressure request')
    _, config = metrics(args.api, args.model)
    blocks = int(config['num_gpu_blocks'])
    if config['enable_prefix_caching'] != 'True' or int(config['block_size']) != BLOCK:
        raise RuntimeError(f'Both arms require GPU prefix caching and block size {BLOCK}: {config}')
    if args.pressure_count*args.pressure_length < blocks*BLOCK:
        raise ValueError(f'Pressure must cover the {blocks} GPU blocks')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        def write(record):
            print(json.dumps(record, allow_nan=False), file=file, flush=True)

        write(dict(kind='config', gpu_blocks=blocks, block_size=BLOCK, lengths=lengths,
                   **{k: v for k,v in vars(args).items() if k not in ('lengths','output')}))
        repetitions = list(range(-args.warmups, 0)) + list(range(args.repetition_start, args.repetition_start+args.repetitions))
        for rep in repetitions:
            order = list(lengths)
            random.Random(f'{args.seed}/order/{rep}').shuffle(order)
            for length in order:
                prefix = tokens(args.seed, rep, length, 'prefix', length)
                for kind in ('build', 'pressure', 'revisit'):
                    count = args.pressure_count if kind == 'pressure' else 1
                    for idx in range(count):
                        if kind == 'pressure':
                            prompt = tokens(args.seed, rep, length, f'pressure-{idx}', args.pressure_length)
                        else:
                            # Different first suffix ID makes the common prefix exactly length.
                            marker = 100 if kind == 'build' else 101
                            prompt = prefix + [marker] + tokens(args.seed, rep, length, kind, SUFFIX-1)
                        row = observe(args, prompt, kind, length, rep, idx)
                        row['warmup'] = rep < 0
                        write(row)
                        if kind == 'revisit':
                            print(f'{args.mode} rep={rep} length={length} TTFT={row["ttft_ms"]:.3f} ms '
                                  f'local={row["prefix_hits"]} external={row["external_tokens"]}', flush=True)
        write(dict(kind='complete', measured_revisits=len(lengths)*args.repetitions))


if __name__ == '__main__':
    main()
