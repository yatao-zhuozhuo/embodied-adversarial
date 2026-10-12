"""Follow successful frozen trajectories into a new split without rewriting sources."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_unified_expert_sft import load_cases, read, write
from scripts.replay_expert_two_tool_sft import _export_samples


def sync(source, destination):
    previous = {case['case_id']: (case, reference) for case, _, reference in load_cases(source / 'snapshot')}
    counts = {}
    changed = False
    for case, _, reference in load_cases(destination / 'snapshot'):
        if previous.get(case['case_id']) != (case, reference):
            raise ValueError('Cannot reuse a changed case or reference: ' + case['case_id'])
        outcome = source / 'results' / (case['case_id'] + '.json')
        if not outcome.exists():
            counts['pending'] = counts.get('pending', 0) + 1
            continue
        result = read(outcome)
        target = destination / 'results' / (case['case_id'] + '.json')
        relative = Path('episodes') / case['bench'] / case['task_id'] / f"seed-{case['seed']}"
        if result.get('success') is True:
            episode = source / relative
            if read(episode / 'episode.json').get('unified_case') != case:
                raise ValueError('Episode does not match frozen case')
            resolved = episode.resolve()
            link = destination / relative
            if link.exists() and not link.is_symlink():
                raise ValueError('Refusing to overwrite episode directory')
            if not link.is_symlink() or link.resolve() != resolved:
                link.parent.mkdir(parents=True, exist_ok=True)
                temporary = link.with_name(link.name + '.next')
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(resolved, target_is_directory=True)
                temporary.replace(link)
                changed = True
        if not target.exists() or read(target) != result:
            write(target, result)
            changed = True
        state = 'success' if result.get('success') is True else 'failed'
        counts[state] = counts.get(state, 0) + 1
    if changed:
        _export_samples(destination)
    write(destination / 'sync_status.json', dict(source=str(source), split_revision=6,
          counts=counts, checked_at=time.time(), original_metadata_preserved=True))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        print(json.dumps(sync(args.source.resolve(), args.destination.resolve())), flush=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
