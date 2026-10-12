#!/usr/bin/env python3
"""Refresh installed task metadata without creating environments or copying data."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from scripts.generate_two_tool_sft_manifest import PYTHONS, _catalog


def worker(bench, code):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['LIBERO_DIR'] = str(ROOT / 'sim/venvs/_sources/LIBERO')
    env['LIBERO_CONFIG_PATH'] = str(ROOT / 'SFT_data/configs/libero')
    if bench == 'libero':
        env['PYTHONPATH'] += os.pathsep + env['LIBERO_DIR']
    result = subprocess.run([str(PYTHONS[bench]), '-c', code], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'{bench}: {result.stderr[-2000:]}')
    marker = next((line[len('UNIFIED_JSON='):] for line in result.stdout.splitlines()
                   if line.startswith('UNIFIED_JSON=')), None)
    if marker is None:
        raise RuntimeError(f'{bench}: no metadata marker')
    return json.loads(marker)


def write(name, data):
    (HERE / 'catalogs').mkdir(exist_ok=True)
    (HERE / 'catalogs' / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def discover(bench):
    os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / 'SFT_data/configs/libero')
    rows = _catalog(bench)
    if bench == 'libero':
        extra = worker(bench, '''
import json
from pathlib import Path
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark_dict
result = {}
for suite, cls in get_benchmark_dict().items():
    if suite not in ('libero_10', 'libero_90', 'libero_spatial', 'libero_object', 'libero_goal'):
        continue
    b = cls()
    for i in range(b.get_num_tasks()):
        t = b.get_task(i)
        result[f'{suite}_task{i}'] = dict(task_name=t.name,
            bddl_path=b.get_task_bddl_file_path(i),
            init_states_path=str(Path(get_libero_path('init_states')) / t.problem_folder / t.init_states_file),
            demonstration_path=str(Path(get_libero_path('datasets')) / b.get_task_demonstration(i)))
print('UNIFIED_JSON=' + json.dumps(result))
''')
        for row in rows:
            row.update(extra[row['task_slug']])
    elif bench == 'maniskill':
        extra = worker(bench, '''
import inspect, hashlib, json
from pathlib import Path
import mani_skill.envs
from mani_skill.utils.registration import REGISTERED_ENVS
result = {}
for task, spec in REGISTERED_ENVS.items():
    cls = spec.cls
    robot = inspect.signature(cls.__init__).parameters.get('robot_uids')
    p = Path(inspect.getfile(cls))
    result[task] = dict(native_default_robot=robot.default if robot and robot.default is not inspect.Parameter.empty else None,
        implementation_path=str(p), implementation_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        asset_download_ids=spec.asset_download_ids)
print('UNIFIED_JSON=' + json.dumps(result))
''')
        for row in rows:
            row.update(extra[row['task_slug']])
    elif bench == 'robocasa':
        registry = worker(bench, '''
import json
from pathlib import Path
from robocasa.utils.dataset_registry_utils import get_ds_meta
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
result = {}
for task in TASK_SET_REGISTRY['all_tasks']:
    m = get_ds_meta(task=task, source='human', split='pretrain')
    if m:
        p = Path(m['path'])
        result[task] = dict(**m, installed=p.is_dir(), resolved_path=str(p.resolve()))
print('UNIFIED_JSON=' + json.dumps(result))
''')
        write('robocasa_demo_registry.json', registry)
    package = {'libero': 'libero', 'metaworld': 'metaworld', 'maniskill': 'mani_skill', 'robocasa': 'robocasa'}[bench]
    versions = worker(bench, f'''
import json, sys
from importlib.metadata import version, PackageNotFoundError
result = dict(python=sys.version.split()[0])
for name in [{package!r}, 'gymnasium', 'numpy', 'mujoco', 'robosuite', 'torch', 'sapien']:
    try: result[name] = version(name)
    except PackageNotFoundError: pass
print('UNIFIED_JSON=' + json.dumps(result))
''')
    write(f'{bench}.json', {'bench': bench, 'python': str(PYTHONS[bench]), 'versions': versions, 'tasks': rows})
    print(f'{bench}: {len(rows)} registered environment IDs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benches', nargs='+', choices=tuple(PYTHONS), default=list(PYTHONS))
    args = parser.parse_args()
    for bench in args.benches:
        discover(bench)
