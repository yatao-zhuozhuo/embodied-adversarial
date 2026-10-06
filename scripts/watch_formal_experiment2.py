#!/usr/bin/env python3
"""Hourly, single-instance supervisor for the explicitly authorized experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / 'runs/formal_experiment2'
INTERVAL = 3600
RUNNERS = {'run_embodied_swift_staged_formal5.sh', 'run_embodied_swift_formal10.sh', 'launch.sh'}


def emit(status, **fields):
    item = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status': status, **fields}
    (RUN / 'watchdog_status.json').write_text(json.dumps(item, indent=2) + '\n')
    with (RUN / 'watchdog_events.jsonl').open('a') as out:
        out.write(json.dumps(item) + '\n')
    print(json.dumps(item), flush=True)


def completed():
    for i in range(5):
        path = RUN / f'round-{i:04d}/summary.json'
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text())
            if not all(Path(data['checkpoints'][role]).joinpath('adapter_model.safetensors').is_file()
                       for role in ('alice', 'bob')):
                return False
        except (ValueError, KeyError, TypeError):
            return False
    return True


def processes():
    owned, runners = [], []
    for proc in psutil.process_iter():
        if proc.pid == os.getpid():
            continue
        try:
            if proc.status() == psutil.STATUS_ZOMBIE:
                continue
            env = proc.environ()
            if env.get('RUN_ROOT') != str(RUN):
                continue
            args = proc.cmdline()
            owned.append(proc)
            if any(Path(arg).name in RUNNERS for arg in args):
                runners.append(proc)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            # Unknown ownership must not be mistaken for an idle machine.
            raise RuntimeError(f'cannot verify process ownership: {proc.pid}')
    return owned, runners


def stop_orphans(owned):
    for proc in owned:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        alive = [p for p in owned if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
        if not alive:
            return
        time.sleep(.25)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    time.sleep(2)


def check():
    if (RUN / 'watchdog.pause').exists():
        emit('paused')
        return False
    if completed():
        emit('completed', rounds=5)
        return True
    owned, runners = processes()
    if runners:
        log = RUN / 'train.log'
        emit('running', runner_pids=[p.pid for p in runners],
             log_age_seconds=round(time.time() - log.stat().st_mtime) if log.exists() else None,
             completed_rounds=len(list(RUN.glob('round-*/summary.json'))))
        return False
    if owned:
        emit('cleaning_orphans', pids=[p.pid for p in owned])
        stop_orphans(owned)
    # Recheck before starting, and never run another experiment's GPU workload over it.
    if processes()[0]:
        raise RuntimeError('experiment processes remain; restart deferred')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                         text=True, capture_output=True, check=True, timeout=30)
    if gpu.stdout.strip():
        emit('restart_deferred_gpu_busy', gpu_pids=gpu.stdout.strip())
        return False
    env = os.environ.copy()
    env['RUN_ROOT'] = str(RUN)
    env['DRY_RUN'] = 'false'
    with (RUN / 'train.log').open('a') as log:
        log.write('\n=== hourly watchdog: resume from existing checkpoints ===\n')
        log.flush()
        child = subprocess.Popen(['bash', str(RUN / 'launch.sh')], cwd=REPO,
                                 env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    (RUN / 'launcher.pid').write_text(str(child.pid) + '\n')
    emit('restarted', pid=child.pid)
    time.sleep(5)
    if child.poll() is not None:
        emit('restart_exited', pid=child.pid, returncode=child.returncode)
    return False


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--run-root', type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run_root.resolve()
    if RUN.parent != REPO / 'runs' or not (RUN / 'launch.sh').is_file():
        raise SystemExit('run-root must be an existing experiment under runs with launch.sh')
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'watchdog.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('watchdog already active')
        (RUN / 'watchdog.pid').write_text(str(os.getpid()) + '\n')
        while True:
            start = time.monotonic()
            try:
                done = check()
            except Exception as exc:
                emit('check_error', error=f'{type(exc).__name__}: {exc}')
                done = False
            if done or args.once:
                return
            time.sleep(max(1, INTERVAL - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
