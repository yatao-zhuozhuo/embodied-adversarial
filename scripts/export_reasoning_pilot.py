"""Publish a frozen, benchmark-balanced one-third reasoning pilot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.annotate_unified_expert_sft import completed_entries, select_entries
from scripts.collect_unified_expert_sft import read, write
from scripts.replay_expert_two_tool_sft import _export_samples

QUOTAS = {'libero': 28, 'metaworld': 33, 'maniskill': 16}


def select_diverse(entries, quota):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['case_id'].rsplit('/', 2)[0]].append(entry)
    selected = []
    while len(selected) < quota and groups:
        for task in sorted(list(groups)):
            selected.append(groups[task].pop(0))
            if not groups[task]:
                del groups[task]
            if len(selected) == quota:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.source / 'annotation_plan.json')
    if (args.output / 'pilot_plan.json').exists():
        return
    while True:
        available = {bench: completed_entries(select_entries(plan, bench), args.source) for bench in QUOTAS}
        counts = {bench: len(entries) for bench, entries in available.items()}
        write(args.output / 'status.json', dict(available=counts, quotas=QUOTAS, complete=False, checked_at=time.time()))
        if all(counts[bench] >= quota for bench, quota in QUOTAS.items()):
            break
        time.sleep(30)
    selected = [entry for bench, quota in QUOTAS.items() for entry in select_diverse(available[bench], quota)]
    for entry in selected:
        episode = Path(entry['episode'])
        relative = Path('episodes') / Path(*episode.parts[-4:-1])
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((args.source / relative).resolve(), target_is_directory=True)
    exported = _export_samples(args.output)
    write(args.output / 'pilot_plan.json', dict(episodes=selected, quotas=QUOTAS, episode_count=len(selected),
          source=str(args.source.resolve()), exported_turns=exported))
    write(args.output / 'status.json', dict(available=counts, quotas=QUOTAS, complete=True, episode_count=len(selected),
          exported_turns=exported, checked_at=time.time()))
    print(json.dumps({'episodes': len(selected), 'exported_turns': exported}), flush=True)


if __name__ == '__main__':
    main()
