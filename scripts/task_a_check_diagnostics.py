"""Check untimed Task A GPU-buffer snapshots, full logits and forward spans."""

import argparse
import json
from pathlib import Path

import torch


def records(folder):
    return [json.loads(line) for path in (folder / 'diagnostics').glob('worker-*.jsonl')
            for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read', required=True, type=Path)
    parser.add_argument('--compute', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for folder in (args.read, args.compute):
        assert json.loads((folder/'status.json').read_text()) == {
            'client_or_startup_exit': 0, 'cleanup_exit': 0}
    read, compute = records(args.read), records(args.compute)
    puts = {row['key']: row for row in read if row['event'] == 'before_put'}
    gets = [row for row in read if row['event'] == 'after_get']
    assert gets
    for row in gets:
        before = puts[row['key']]
        assert before['bytes'] == row['bytes'] and before['segment_bytes'] == row['segment_bytes']
        assert before['sha256'] == row['sha256']
        assert (args.read/'diagnostics'/before['file']).read_bytes() == (
            args.read/'diagnostics'/row['file']).read_bytes()
    comparisons = []
    for first in read:
        if first['event'] != 'logits' or first['marker']['kind'] != 'revisit' or first['ordinal'] != 0:
            continue
        tag = first['marker']['tag']
        reference = next(row for row in compute if row['event']=='logits'
                         and row['marker']['tag']==tag and row['ordinal']==0)
        a = torch.load(args.read/'diagnostics'/first['file'], weights_only=True).float()
        b = torch.load(args.compute/'diagnostics'/reference['file'], weights_only=True).float()
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        difference = (a-b).abs()
        entry = dict(marker=first['marker'], vocabulary=a.shape[-1],
                     max_absolute_logit_error=difference.max().item(),
                     mean_absolute_logit_error=difference.mean().item(),
                     read_argmax=a[0].argmax().item(), recompute_argmax=b[0].argmax().item())
        for label, tensor, rows in [('read',a,read),('recompute',b,compute)]:
            values, ids = tensor[0].topk(3)
            entry[label+'_top_logits'] = values.tolist()
            entry[label+'_top_ids'] = ids.tolist()
            spans = [row for row in rows if row['event']=='model_forward' and row['marker']['tag']==tag]
            entry[label+'_first_forward'] = {key: val for key,val in spans[0].items()
                                             if key not in ('marker','event')}
        comparisons.append(entry)
    result = dict(copied_kv_bitwise_equal=True, verified_get_blocks=len(gets),
                  verified_get_bytes=sum(row['bytes'] for row in gets),
                  comparisons=comparisons,
                  timing_scope='Untimed diagnostics. CUDA event forward span includes CPU launch gaps.',
                  excluded_from_primary_ttft_statistics=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
