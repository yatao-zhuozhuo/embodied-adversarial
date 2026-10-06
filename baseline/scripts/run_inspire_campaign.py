"""Gate an Inspire campaign on real environment, image and action checks.

Run in the baseline checkout after preparing models and starting its services.
This does not submit platform resources or start model servers.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'baseline/.runtime/coordinator/bin/python'
BENCHES = ('libero', 'metaworld', 'maniskill3', 'robocasa')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--preflight-wait-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not args.campaign or Path(args.campaign).name != args.campaign or args.campaign in {'.', '..'}:
        parser.error('campaign must be one directory name')
    campaign = ROOT / 'baseline/results' / args.campaign
    campaign.mkdir(exist_ok=False)
    logs = campaign / 'logs'
    logs.mkdir()
    os.chdir(ROOT)

    def status(stage, **details):
        record = {'stage': stage, 'updated_at': time.time(), **details}
        temp = campaign / 'status.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(campaign / 'status.json')
        print(json.dumps(record), flush=True)

    def run(label, script, *arguments):
        status(label)
        with (logs / f'{label}.log').open('w') as log:
            subprocess.run([str(PYTHON), str(ROOT / script), *arguments],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        expected = [ROOT / f'baseline/results/{phase}/{bench}-{i}.json'
                    for phase in ('environment-preflight', 'environment-repeat')
                    for bench in BENCHES for i in range(10)]
        deadline = time.monotonic() + args.preflight_wait_seconds
        status('waiting_for_80_environment_checks')
        while True:
            ready = 0
            for path in expected:
                if not path.exists():
                    continue
                try:
                    record = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                if not record.get('passed'):
                    raise RuntimeError(f'Environment check failed: {path}')
                ready += 1
            if ready == len(expected):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Only {ready}/80 environment checks completed')
            time.sleep(10)
        run('freeze_questions', 'baseline/scripts/freeze_questions.py')
        run('freeze_protocol', 'baseline/scripts/freeze_protocol.py')
        for bench in BENCHES:
            run(f'native_{bench}', 'baseline/scripts/native_chain_smoke.py',
                '--bench', bench, '--output', str(campaign / f'native-{bench}.json'))
        models = ('qwen35_4b', 'qwen38_27b', 'qwen35_9b')
        # All three models must pass before any formal result is collected.
        for model in models:
            run(f'vision_{model}', 'baseline/scripts/inference_smoke.py',
                '--models', model, '--require-vision', '--output', str(campaign / f'vision-{model}.json'))
            run(f'canary_{model}', 'baseline/scripts/launch_model.py', '--model', model,
                '--root', str(campaign / 'canary'), '--benchmark', 'metaworld',
                '--limit', '1', '--turns', '2', '--timeout', '300')
            run(f'check_canary_{model}', 'baseline/scripts/check_canary.py',
                str(campaign / 'canary' / model / 'baseline-metaworld-00/final.json'))
        for model in models:
            run(f'formal_{model}', 'baseline/scripts/launch_model.py',
                '--model', model, '--root', str(campaign / 'formal'))
        status('complete', models=list(models), planned_episodes=120)
    except Exception as exc:
        status('blocked', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
