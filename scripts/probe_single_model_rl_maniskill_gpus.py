"""Check native ManiSkill rendering on every job GPU before loading model weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def child():
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    sys.path.insert(0, str(ROOT))
    import sapien
    from sim.env_registry import _make_maniskill_direct
    print(json.dumps({'visible_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
                      'render_pci': sapien.Device('cuda').pci_string}), flush=True)

    def check():
        import numpy as np
        start = time.monotonic()
        env = _make_maniskill_direct('PullCube-v1', image_width=256, image_height=256)
        try:
            for _ in range(3):
                env.reset(seed=20481500)
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                action[-1] = 1.
                env.step(action)
            print(json.dumps({'status': 'passed', 'seconds': time.monotonic()-start}), flush=True)
        finally:
            env.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(check).result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--processes-per-gpu', type=int, default=2)
    args = parser.parse_args()
    if args.child:
        child()
        return
    if args.output is None:
        parser.error('--output is required')
    if not 1 <= args.processes_per_gpu <= 4:
        parser.error('processes-per-gpu must be between 1 and 4')
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,driver_version',
                                         '--format=csv,noheader'], text=True)
    (args.output/'gpu_inventory.csv').write_text(inventory)
    gpus = [line.split(',')[0].strip() for line in inventory.splitlines()]
    if len(gpus) != 8:
        raise RuntimeError('Expected eight GPUs on this job node')

    def probe(item):
        gpu, replica = item
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        env.pop('MUJOCO_EGL_DEVICE_ID', None)
        with (args.output/f'gpu-{gpu}-{replica}.log').open('w') as log:
            process = subprocess.Popen([os.environ['OPENETA_MANISKILL_PYTHON'], str(Path(__file__).resolve()),
                                        '--child'], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=180)
                record = {'gpu': gpu, 'replica': replica, 'exit_code': code,
                          'status': 'passed' if code == 0 else 'failed'}
            except subprocess.TimeoutExpired:
                record = {'gpu': gpu, 'replica': replica, 'status': 'failed', 'error': 'render timeout'}
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record['cleanup_error'] = 'GPU process did not exit after SIGKILL'
        print(json.dumps(record), flush=True)
        return record

    items = [(gpu, i) for gpu in gpus for i in range(args.processes_per_gpu)]
    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        records = list(executor.map(probe, items))
    (args.output/'summary.json').write_text(json.dumps({'host': os.uname().nodename, 'checks': records}, indent=2)+'\n')
    if any(record['status'] != 'passed' for record in records):
        raise SystemExit('ManiSkill GPU rendering check failed; model training was not started')


if __name__ == '__main__':
    main()
