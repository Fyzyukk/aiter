#!/usr/bin/env python3
"""Create a short-K-first copy of the complete 201-winner Event queue on CPU."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = OUT / 'compute_prologue_all_winners_event_queue.json'
    source_sha = sha(source)
    queue = json.loads(source.read_text())
    decorated = []
    for command in queue['commands']:
        argv = command['argv']
        plan_path = Path(argv[argv.index('--plan') + 1])
        index = int(argv[argv.index('--target-index') + 1])
        target = json.loads(plan_path.read_text())['targets'][index]
        assert target['same_parent_current_winner'] and '--event-confirm' in argv
        assert argv[argv.index('--iters') + 1] == '51' and argv[argv.index('--repetitions') + 1] == '8'
        decorated.append((target['shape'][2], target['kid'], target['shape'][0], target['shape'][1], command, target))
    # Strong initial short-K effects motivate prioritization; all 201 remain.
    decorated.sort(key=lambda x: x[:4])
    ordered = copy.deepcopy(queue)
    ordered['commands'] = [entry[4] for entry in decorated]
    assert len(ordered['commands']) == 201
    assert {c['name']: c for c in ordered['commands']} == {c['name']: c for c in queue['commands']}
    output = OUT / 'compute_prologue_all_winners_event_short_k_queue.json'
    output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + '\n')
    assert sha(source) == source_sha
    manifest = {'status': 'cpu_prepared_priority_copy_not_executed', 'new_gpu_execution': False,
                'original_queue': str(source.relative_to(ROOT)), 'original_queue_sha256': source_sha,
                'priority_queue': str(output.relative_to(ROOT)), 'priority_queue_sha256': sha(output),
                'command_count': 201, 'original_command_content_and_coverage_unchanged': True,
                'ordering': 'K ascending, then parent 9021/9022, M ascending, N ascending',
                'parent_counts': dict(Counter(entry[1] for entry in decorated)),
                'K_counts': dict(sorted(Counter(entry[0] for entry in decorated).items())),
                'first_commands': [{'name': c['name'], 'parent_id': t['kid'], 'shape': t['shape']}
                                   for _, _, _, _, c, t in decorated[:24]],
                'note': 'Short-K representatives had larger observed prologue effects. Ordering is scheduling only, not a selection or adoption gate.'}
    path = OUT / 'compute_prologue_event_priority_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'queue': str(output), 'command_count': 201,
                      'parent_counts': manifest['parent_counts'], 'K_counts': manifest['K_counts'],
                      'original_queue_unchanged': True}))


if __name__ == '__main__':
    main()
