"""Pair completed Task A runs, validate cache paths, and summarize client TTFT."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles


def load_arm(folder):
    status = json.loads((folder / 'status.json').read_text())
    if status != {'client_or_startup_exit': 0, 'cleanup_exit': 0}:
        raise ValueError(f'Incomplete run: {folder}')
    with (folder / 'measurements.jsonl').open() as f:
        rows = [json.loads(line) for line in f]
    measured = sum(row['kind'] == 'revisit' and not row['warmup']
                   for row in rows if row['kind'] == 'revisit')
    if rows[0]['kind'] != 'config' or rows[-1] != dict(kind='complete', measured_revisits=measured):
        raise ValueError(f'Incomplete measurements: {folder}')
    return rows[0], rows[1:-1]


def paired_samples(first, second):
    cfg_a, a = load_arm(first)
    cfg_b, b = load_arm(second)
    for name in ('gpu_blocks', 'block_size', 'model', 'seed', 'lengths', 'repetitions', 'warmups',
                 'pressure_count', 'pressure_length', 'output_tokens'):
        if cfg_a[name] != cfg_b[name]:
            raise ValueError(f'Unmatched {name}: {first}, {second}')
    if cfg_a['mode'] != 'recompute' or cfg_b['mode'] != 'store':
        raise ValueError('Pair order must be recompute, then store')
    run_a,run_b = [json.loads((p/'run.json').read_text()) for p in (first,second)]
    for key in ('gpu_blocks','block_tokens','max_model_len','max_num_seqs','dtype',
                'gpu_memory_utilization','enable_prefix_caching','hash_algorithm','model'):
        if run_a[key] != run_b[key]:
            raise ValueError(f'Unmatched runtime {key}')
    if run_a.get('max_num_batched_tokens', 2048) != run_b.get('max_num_batched_tokens', 2048):
        raise ValueError('Unmatched prefill scheduling granularity')
    tiers = (second/'serve.log').read_text(encoding='utf-8',errors='replace')
    for x, y in zip(a, b, strict=True):
        for name in ('kind', 'index', 'repetition', 'length', 'prompt_digest', 'warmup', 'prompt_tokens'):
            if x[name] != y[name]:
                raise ValueError(f'Unmatched workload: {name}')
        if x['kind'] != 'revisit':
            continue
        n = x['length']
        if (x['prefix_hits'] or y['prefix_hits'] or x['external_tokens'] or x['get_bytes']
            or y['external_tokens'] != n or y['get_bytes'] != n * 28672
            or x['compute_tokens'] != x['prompt_tokens'] or y['compute_tokens'] != y['prompt_tokens'] - n):
            raise ValueError(f'Invalid path for {x["request_id"]}')
        if x['warmup']:
            continue
        tier = next((line for line in tiers.splitlines()
                     if 'Mooncake load tier summary:' in line and f"req_id={y['request_id']}-" in line), None)
        expected = {'memory_keys':n//16,'disk_keys':0,'unknown_keys':0,'success_keys':n//16,'failed_keys':0}
        if not tier or any(not re.search(rf'\b{k}={v}\b',tier) for k,v in expected.items()):
            raise ValueError(f'Missing successful DRAM tier proof for {y["request_id"]}')
        yield dict(length=n, seed=x['seed'], repetition=x['repetition'],
                   recompute_ttft_ms=x['ttft_ms'], read_ttft_ms=y['ttft_ms'],
                   saved_ms=x['ttft_ms']-y['ttft_ms'], local_hit_tokens=y['prefix_hits'],
                   recompute_server_ttft_ms=x['server_ttft_ms'], read_server_ttft_ms=y['server_ttft_ms'],
                   read_tokens=y['external_tokens'], read_bytes=y['get_bytes'], read_get_ms=y['get_ms'],
                   read_lookup_ms=y['lookup_ms'], recompute_tokens=x['compute_tokens'],
                   read_server_minus_get_lookup_ms=y['server_ttft_ms']-y['get_ms']-y['lookup_ms'],
                   read_compute_tokens=y['compute_tokens'], token_ids_equal=x['token_ids']==y['token_ids'],
                   recompute_ids=' '.join(map(str,x['token_ids'])), read_ids=' '.join(map(str,y['token_ids'])))


def quartiles(xs):
    if len(xs) == 1:
        return xs[0], xs[0], xs[0]
    return tuple(quantiles(xs,n=4,method='inclusive'))


def save_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',action='append',required=True,nargs=2,type=Path,metavar=('COMPUTE','READ'))
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--plot',action='store_true',help='Render TTFT medians and IQR with Matplotlib')
    args = parser.parse_args()
    rows = [row for pair in args.pair for row in paired_samples(*pair)]
    if not rows or len({(r['seed'],r['repetition'],r['length']) for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate measurements')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['length']].append(row)
    summary = []
    for n, group in sorted(grouped.items()):
        s = dict(length=n, pairs=len(group))
        for mode, col in [('recompute','recompute_ttft_ms'),('read','read_ttft_ms')]:
            q1,m,q3 = quartiles([r[col] for r in group])
            s.update({f'{mode}_{k}_ms':v for k,v in zip(('q1','median','q3'),(q1,m,q3))})
        margin = median(r['saved_ms'] for r in group)
        wins = sum(r['saved_ms']>0 for r in group)
        correct = sum(r['token_ids_equal'] for r in group)
        s.update(median_saved_ms=margin, read_wins=wins, matching_outputs=correct,
                 median_get_ms=median(r['read_get_ms'] for r in group),
                 median_lookup_ms=median(r['read_lookup_ms'] for r in group))
        s['median_read_server_minus_get_lookup_ms'] = median(
            r['read_server_minus_get_lookup_ms'] for r in group)
        summary.append(s)
    args.out.mkdir(parents=True,exist_ok=True)
    save_csv(args.out/'samples.csv',rows)
    save_csv(args.out/'summary.csv',summary)
    mismatches = [r for r in rows if not r['token_ids_equal']]
    checks = dict(path_checks_passed=True, paired_revisits=len(rows),
                  matching_output_pairs=len(rows)-len(mismatches),
                  all_output_token_ids_equal=not mismatches,
                  warmups_excluded=True, gpu_local_hit_tokens=0,
                  memory_tier_verified=True, per_request_successful_get_bytes_verified=True,
                  scope='Qwen2.5-1.5B FP16, V100, single-host TCP, synthetic token IDs',
                  limitations=['Four generated tokens per request',
                               'Selected lengths 64 to 1536 tokens',
                               'Generation compares IDs; full-vocabulary logits are not captured'],
                  mismatches=[dict(length=r['length'],seed=r['seed'],repetition=r['repetition'],
                                   recompute_ids=r['recompute_ids'],read_ids=r['read_ids'])
                              for r in mismatches])
    (args.out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    if args.plot:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(7,4.1),layout='constrained')
        ns=[s['length'] for s in summary]
        for mode,label,marker in [('recompute','GPU recompute','o'),('read','DRAM read','s')]:
            m=[s[f'{mode}_median_ms'] for s in summary]
            low=[v-s[f'{mode}_q1_ms'] for v,s in zip(m,summary)]
            high=[s[f'{mode}_q3_ms']-v for v,s in zip(m,summary)]
            ax.errorbar(ns,m,yerr=(low,high),marker=marker,capsize=3,label=label)
        ax.set(xlabel='Reusable prefix (tokens)',ylabel='Client TTFT (ms)',title='V100 prefix revisit · median and interquartile range')
        ax.grid(axis='y',alpha=.2)
        ax.legend()
        fig.savefig(args.out/'ttft.png',dpi=180)
        plt.close(fig)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
