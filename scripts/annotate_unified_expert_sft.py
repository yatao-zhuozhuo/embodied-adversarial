"""Freeze and annotate verified short unified SFT trajectories with Gemini."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.annotate_expert_sft_reasoning import MODEL, annotate_episode
from scripts.collect_unified_expert_sft import load_cases, read, write
from scripts.replay_expert_two_tool_sft import _export_samples


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(input_root, output_root):
    plan_path = output_root / 'annotation_plan.json'
    if plan_path.exists():
        return read(plan_path)
    entries = []
    for case, _, _ in load_cases(input_root / 'snapshot'):
        outcome = input_root / 'results' / (case['case_id'] + '.json')
        if not outcome.exists():
            continue
        result = read(outcome)
        if result.get('success') is not True or not 0 < result.get('compressed_turns', 101) <= 100:
            continue
        episode = (input_root / 'episodes' / case['bench'] / case['task_id'] / f"seed-{case['seed']}" / 'episode.json').resolve()
        record = read(episode)
        if record.get('unified_case') != case or record.get('stop_reason') != 'official_success':
            raise ValueError('Frozen success evidence mismatch: ' + case['case_id'])
        samples = episode.parent / 'samples.jsonl'
        if not record.get('terminal_frames') or not 0 < record.get('samples', 0) <= 100:
            raise ValueError('Missing terminal evidence or invalid turn count')
        entries.append(dict(case_id=case['case_id'], episode=str(episode),
                            episode_sha256=digest(episode), samples_sha256=digest(samples), turns=record['samples']))
    if not entries:
        raise ValueError('No verified short SFT trajectories')
    plan = dict(split_revision=6, model=MODEL, input_root=str(input_root.resolve()),
                episodes=entries, episode_count=len(entries), turn_count=sum(entry['turns'] for entry in entries))
    write(plan_path, plan)
    return plan


def select_entries(plan, bench):
    return [entry for entry in plan['episodes'] if not bench or entry['case_id'].split('/')[0] == bench]


def completed_entries(entries, output_root):
    completed = []
    for entry in entries:
        episode = Path(entry['episode'])
        directory = output_root / 'episodes' / Path(*episode.parts[-4:-1])
        report = directory / 'report.json'
        if report.exists() and (directory / 'samples.jsonl').exists():
            value = read(report)
            if value.get('success') is True and value.get('source_samples_sha256') == entry['samples_sha256']:
                completed.append(entry)
    return completed


def publish_status(output_root, plan, status, bench):
    if not bench:
        write(output_root / 'status.json', status)
        return
    with (output_root / '.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write(output_root / f'status.{bench}.json', status)
        reports = list(output_root.glob('episodes/*/*/seed-*/report.json'))
        benches = {path.stem.split('.', 1)[1]: read(path) for path in output_root.glob('status.*.json')}
        write(output_root / 'status.json', dict(model=status['model'], mode='parallel_benchmarks',
              completed_episodes=len(reports), planned_episodes=plan['episode_count'], planned_turns=plan['turn_count'],
              benches=benches, failures=[failure for row in benches.values() for failure in row.get('failures', [])],
              complete=len(reports) == plan['episode_count'], checked_at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--bench', choices=['libero', 'metaworld', 'maniskill'])
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--retry-passes', type=int, default=5)
    parser.add_argument('--success-target', type=int, default=0)
    args = parser.parse_args()
    key = os.environ.get('CODINGROUTER_API_KEY')
    if not key:
        parser.error('CODINGROUTER_API_KEY is required')
    plan = freeze(args.input_root, args.output_root)
    entries = select_entries(plan, args.bench)
    worker_lock = (args.output_root / f".annotation-{args.bench or 'all'}.lock").open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for attempt in range(args.retry_passes):
        completed = len(completed_entries(entries, args.output_root))
        failed = []
        for entry in entries:
            if args.success_target and completed >= args.success_target:
                break
            episode = Path(entry['episode'])
            samples = episode.parent / 'samples.jsonl'
            if digest(episode) != entry['episode_sha256'] or digest(samples) != entry['samples_sha256']:
                raise ValueError('Frozen source changed: ' + entry['case_id'])
            relative = Path('episodes') / Path(*episode.parts[-4:-1])
            directory = args.output_root / relative
            report_path = directory / 'report.json'
            try:
                if report_path.exists() and read(report_path).get('source_samples_sha256') == entry['samples_sha256'] and (directory / 'samples.jsonl').exists():
                    continue
                annotate_episode(samples, args.output_root, api_key=key, timeout=args.timeout,
                                 verbose=False, workers=args.workers, model=args.model)
                for name in ['episode.json', 'frames']:
                    target = directory / name
                    if not target.exists():
                        target.symlink_to(episode.parent / name, target_is_directory=name == 'frames')
                completed += 1
                with (args.output_root / '.export.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    _export_samples(args.output_root)
            except Exception as error:
                failed.append(dict(case_id=entry['case_id'], error=str(error).replace(key, '[REDACTED]')))
            status = dict(model=args.model, pass_index=attempt + 1, completed_episodes=completed,
                          planned_episodes=len(entries), planned_turns=sum(entry['turns'] for entry in entries), failures=failed,
                          checked_at=time.time())
            publish_status(args.output_root, plan, status, args.bench)
            print(json.dumps(status), flush=True)
        publish_status(args.output_root, plan, dict(model=args.model, completed_episodes=completed,
              planned_episodes=len(entries), planned_turns=sum(entry['turns'] for entry in entries), failures=failed,
              complete=completed >= (args.success_target or len(entries)), success_target=args.success_target,
              checked_at=time.time()), args.bench)
        if not failed or (args.success_target and completed >= args.success_target):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
